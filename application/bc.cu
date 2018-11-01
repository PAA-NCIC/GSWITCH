#include <iostream>
#include <queue>
#include <stack>
#include "gswitch.h"

using G = device_graph_t<CSR, Empty>;

// actors
inspector_t inspector; 
selector_t selector;
executor_t executor;
feature_t fets;
config_t conf;
stat_t stats;

///////////////////////////////////////////
//Forward:
// - prop : NULL
// - msg  : num_of_spath(root,v), level

//Backward:
// - prop : level
// - msg  : num_of_spath(root,v), dependency(r,v)

struct fwdata_t{
  int level; // dft = -1
  int nsp; // dft = 0
};

struct Forward:Functor<VC,fwdata_t,Empty,Empty>{
  __device__ Status filter(int vid, G g){
    fwdata_t* v = wa_of(vid);
    if(v->level == g.get_level()) return Active;
    else if (v->level < 0) return Inactive;
    else return Fixed;
  }
  __device__ fwdata_t emit(int vid, Empty* em, G g){
	  fwdata_t vdata = *wa_of(vid);
	  vdata.level++;
	  return vdata;
  }
  __device__ bool cond(int v, fwdata_t u, G g){
    return wa_of(v)->level < 0 || wa_of(v)->level == g.get_level();
  }
  __device__ bool comp(fwdata_t* v, fwdata_t u, G g){
    v->level = u.level; 
    v->nsp  += u.nsp;
    return true;
  }
  __device__ bool compAtomic(fwdata_t* v, fwdata_t u, G g){
    v->level = u.level;
    atomicAdd(&(v->nsp), u.nsp);
    return true;
  }
};

struct bwdata_t{
  int nsp;
  float dep;
};

//float tmp = (0.0+v->nsp)* ((1+newv.dep)/newv.nsp);
struct Backward:Functor<VC,bwdata_t,int,Empty>{
  __device__ Status filter(int vid, G g){
    if(*ra_of(vid) == g.get_level()){
      bwdata_t* v = wa_of(vid);
      v->dep = v->nsp*v->dep;
      return Active;
    }else if(*ra_of(vid) < g.get_level()) return Inactive;
    else return Fixed;
  }
  __device__ bwdata_t emit(int vid, Empty* em, G g){
    bwdata_t vdata = *wa_of(vid);
    vdata.dep = (vdata.dep+1)/vdata.nsp;
    return vdata;
  }
  __device__ bool cond(int v, bwdata_t newv, G g){
    return *ra_of(v) == g.get_level();
  }
  __device__ bool comp(bwdata_t* v, bwdata_t newv, G g){
    v->dep += newv.dep;
    return true;
  }
  __device__ bool compAtomic(bwdata_t* v, bwdata_t newv, G g){
    atomicAdd(&(v->dep), newv.dep);
    return true;
  }
};
///////////////////////////////////////////

template<typename G, typename F, typename H>
double run_bc(G &g, F& f, H& h, int root){
  double elapsed_time=0;
  // step 1: initializing
  LOG(" -- Initializing\n");
  active_set_t as = build_active_set(g.nvertexs, conf);
  as.init(root);

  {// step 2: Forward phase
    double s = mwtime();
    for(int level=0;;level++){
      inspector.inspect(as, g, f, stats, fets, conf);
      if(as.finish(g,f,conf)){break;}
      selector.select(stats, fets, conf);
      executor.filter(as, g, f, stats, fets, conf);
      g.update_level();
      executor.expand(as, g, f, stats, fets, conf);
      //fets.record();// for training
      if(as.finish(g,f,conf)){break;}
    }
    double e = mwtime();
    LOG("GPU Forward time: %.3f ms\n", e-s);
    f.data.sync_wa();
    elapsed_time += e-s;
  }

  // step 3: prepare next phase, data copy
  set_functor(f, h);
  as.reset(true);
  conf.reset();
  fets.reset();
  stats.reset();
  conf.conf_compensation=true;
  g.update_level(-1);
  if(conf.conf_fuse_inspect) g.update_level(-1); // this is ugly
  as.init(ALL_INACTIVE);

  {// step 4: Backward phase
    double s = mwtime();
    for(int level=0;;level++){
      inspector.inspect(as, g, h, stats, fets, conf);
      if(as.finish(g,h,conf)) break;
      selector.select(stats, fets, conf);
      executor.filter(as, g, h, stats, fets, conf);
      g.update_level(-1);
      executor.expand(as, g, h, stats, fets, conf);
      //fets.record();// for training
      if(as.finish(g,h,conf)) break;
    }
    double e = mwtime();
    LOG("GPU Backward time: %.3f ms\n", e-s);
    h.data.sync_wa();
    elapsed_time += e-s;
  }

  return elapsed_time;
}

bwdata_t* bc_cpu(host_graph_t<CSR,Empty> hg, int root){
  LOG("generate CPU BC reference\n");
  double ms = mwtime();
  bwdata_t* ret = (bwdata_t*)malloc(sizeof(bwdata_t)*hg.nvertexs);
  int* vis = (int*)malloc(sizeof(int)*hg.nvertexs);
  std::queue<int> q;
  std::stack<int> st;
  for(int i=0;i<hg.nvertexs;i++) ret[i].nsp=0, vis[i]=-1, ret[i].dep=0;
  ret[root].nsp = 1;
  vis[root] = 1;
  q.push(root);

  // Forward
  while(!q.empty()){
    int v = q.front();
    q.pop();
    st.push(v);
    int s = hg.start_pos[v];
    int e = (v==(hg.nvertexs-1)?hg.nedges:hg.start_pos[v+1]);
    for(int j=s; j<e; ++j){
      int u = hg.adj_list[j];
      if(vis[u]==-1){
        q.push(u);
        vis[u] = vis[v]+1;
      }
      if(vis[u]==vis[v]+1){
        ret[u].nsp += ret[v].nsp;
      }
    }
  }

  // Backward
  while(!st.empty()){
    int v = st.top();
    st.pop();
    ret[v].dep *= ret[v].nsp;
    int s = hg.start_pos[v];
    int e = (v==(hg.nvertexs-1)?hg.nedges:hg.start_pos[v+1]);
    for(int j=s; j<e; ++j){
      int u = hg.adj_list[j];
      if(vis[u]+1 == vis[v]){
        float tmp = (1+ret[v].dep)/ret[v].nsp;
        ret[u].dep += tmp;
      }
    }
  }

  free(vis);
  double me = mwtime();
  LOG("CPU BC: %.3f ms\n",me-ms);
  return ret;
}

void validation(bwdata_t* A, bwdata_t* B, int N){
  bool flag=true;
  for(int i=0;i<N;++i){
    if(!isAcceptable(A[i].nsp, B[i].nsp)){
      flag = false;
      puts("failed");
      std::cout << i << " " << A[i].nsp << " " << B[i].nsp << std::endl;
      break;
    }
  }
  if(flag) puts("Forward passed");

  flag=true;
  for(int i=0;i<N;++i){
    if(!isAcceptable(A[i].dep, B[i].dep)){
      flag = false;
      puts("failed");
      std::cout << i << " " << A[i].dep << " " << B[i].dep << std::endl;
      break;
    }
  }
  if(flag) puts("Backward passed");
}

template<typename F, typename H>
void set_functor(F& f, H& h){
  fwdata_t* fd = f.data.h_wa;
  h.data.init_ra([fd](int i){return fd[i].level;});
  h.data.init_wa([fd](int i){return bwdata_t{fd[i].nsp,0.0};});
}

//"/home/mengke/data/com-orkut/com-orkut.ungraph.txt"
int main(int argc, char* argv[]){
  parse_cmd(argc, argv, "BC");

  // step 1 : set features
  fets.centric = VC;
  fets.pattern = ASSO;
  fets.use_root = true;
  fets.fromall = false;
  fets.toall = false;
  conf.conf_fuse_inspect = false;
  conf.conf_qmode = Cached;
  //conf.conf_qmode = Normal;


  // step 2 : init Graph & Algorithm
  auto g = build_graph<VC>(cmd_opt.path, fets, cmd_opt.with_header, cmd_opt.with_weight, cmd_opt.directed);
  Forward f;
  Backward h;
  f.data.build(g.hg.nvertexs);
  h.data.build(g.hg.nvertexs);

  // step 3 : choose root vertex
  int root = cmd_opt.src;
  if(root < 0) root = g.hg.random_root();
  LOG(" -- Root is: %d\n", root);
  fets.use_root = root;
  f.data.init_wa([root](int i){
    return i==root?fwdata_t{0,1}:fwdata_t{-100,0};
  });
  init_conf(stats, fets, conf, g, f);

  // step 4 : execute Algorithm
  LOG(" -- Launching BC\n");
  double elapsed_time = run_bc(g.dg,f,h,root);

  // step 5 : validation
  if(cmd_opt.validation){
    bwdata_t* A = bc_cpu(g.hg, root);
    validation(A, h.data.h_wa, g.hg.nvertexs);
  }

  LOG("GPU BC time: %.3f ms\n", elapsed_time);
	std::cout << elapsed_time << std::endl;

  return 0;
}

