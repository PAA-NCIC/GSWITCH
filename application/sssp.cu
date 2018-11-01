#include <iostream>
#include <queue>
#include <limits>
#include "gswitch.h"

using G = device_graph_t<CSR, int>;
const int maxi = std::numeric_limits<int>::max();

// actors
inspector_t inspector; 
selector_t selector;
executor_t executor;
feature_t fets;
config_t conf;
stat_t stats;

struct SSSP:Functor<VC,int,int,int>{

 
  __device__ Status filter(int vid, G g){
    int* newd = wa_of(vid);
    int* oldd = ra_of(vid);
    if(*newd == *oldd) return Inactive;
    if(data.window.in(*newd)){
      *oldd = *newd;
      return Active;
    }
    return Inactive;
  }
  __device__ int emit(int vid, int* w, G g){
    return *wa_of(vid)+*w;
  }

  __device__ bool comp(int* disu, int newv, G g){
    if(*disu > newv){*disu = newv;return data.window.in(newv);}
    return false;
  }
  __device__ bool compAtomic(int* v, int newv, G g){
    int old = atomicMin(v, newv);
    if(old > newv) return data.window.in(newv);
    else return false;
  } 

  __device__ bool cond(int vid, int msg, G g){
    int* newd = wa_of(vid);
    return *newd > msg;
  }

  __device__ bool exit(int v, G g){
    return *wa_of(v) != *ra_of(v);
  }
};

int* spfa(host_graph_t<CSR,int> hg, int root){
  std::cout << "generate CPU SSSP reference" << std::endl;
  double ms = mwtime();
  int* dis = (int*)malloc(sizeof(int)*hg.nvertexs);
  int* vis = (int*)malloc(sizeof(int)*hg.nvertexs);
  memset(vis,0,sizeof(int)*hg.nvertexs);
  std::queue<int> q;
  for(int i=0;i<hg.nvertexs;i++) dis[i]=maxi;
  dis[root] = 0;
  vis[root] = 1;
  q.push(root);
  while(!q.empty()){
    int v = q.front();
    q.pop();
    vis[v] = 0;
    int s = hg.start_pos[v];
    int e = (v==(hg.nvertexs-1)?hg.nedges:hg.start_pos[v+1]);
    for(int j=s; j<e; ++j){
      int u = hg.adj_list[j];
      int w = hg.edge_weights[j];
      if(dis[u]>dis[v]+w){
        dis[u]=dis[v]+w;
        if(!vis[u]){
          q.push(u);
          vis[u]=1;
        }
      }
    }
  }
  double me = mwtime();
  LOG("CPU SSSP: %.3f ms\n",me-ms);
  free(vis);
  return dis;
}


void validation(int* dGPU, int* dCPU, int N){
  bool flag=true;
  //const float eps=1e-5;
  for(int i=0;i<N;++i){
    if(dGPU[i]-dCPU[i] != 0){
      flag = false;
      puts("failed");
      std::cout << i << " " << dGPU[i] << " " << dCPU[i] << std::endl;
      break;
    }
  }
  if(flag) puts("passed");
}

template<typename G, typename F>
double run_sssp(G g, F f, int root){
  // step 1: initializing
  LOG(" -- Initializing\n");
  active_set_t as = build_active_set(g.nvertexs, conf);
  as.init(root);

  // step 2: Execute Algorithm
  double s = mwtime();
  for(int level=0;;level++){
    inspector.inspect(as, g, f, stats, fets, conf);
    selector.select(stats, fets, conf);
    executor.filter(as, g, f, stats, fets, conf);
    g.update_level();
    executor.expand(as, g, f, stats, fets, conf);
    if(as.finish(g,f,conf)) break;
  }
  double e = mwtime();

  return e-s;
}


int main(int argc, char* argv[]){
  parse_cmd(argc, argv, "SSSP");

  // step 1 : set features
  fets.centric = VC;
  fets.pattern = Mono;
  fets.fromall = false;
  fets.toall = true;
  conf.conf_window = true;
  //conf.conf_compensation = true;

  // step 2 : init Graph & Algorithm
  edgelist_t<int> el;
  el.read_mtx(cmd_opt.path, cmd_opt.directed, cmd_opt.with_weight, cmd_opt.with_header);
  el.gen_weight(64);
  auto g = build_graph<VC,int>(el, fets);
  SSSP f;
  f.data.build(g.hg.nvertexs);

  // step 3 : choose root vertex
  int root = cmd_opt.src;
  if(root < 0) root = g.hg.random_root();
  LOG(" -- Root is: %d\n", root);
  fets.use_root = root;
  f.data.init_wa([root](int i){return i==root?0:maxi;});
  f.data.init_ra([root](int i){return i==root?0:maxi;});
  init_conf(stats, fets, conf, g, f);

  // step 3 : execute Algorithm
  LOG(" -- Launching SSSP\n");
  double time = run_sssp(g.dg,f,root);
    
  // step 4 : validation
  f.data.sync_wa();
  if(cmd_opt.validation){
    int* dCPU=spfa(g.hg, root);
    validation(f.data.h_wa,dCPU,g.hg.nvertexs);
  }

  LOG("GPU SSSP time: %.3f ms\n", time);
  std::cout << time << std::endl;
	//std::cout << fets.nvertexs << " " 
        //<< fets.nedges << " "
        //<< fets.avg_deg << " "
        //<< fets.std_deg << " "
        //<< fets.range << " "
        //<< fets.GI << " "
        //<< fets.Her << " "
        //<< time << std::endl;

  return 0;
}


