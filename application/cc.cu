#include <iostream>
#include <queue>
#include "gswitch.h"

using G = device_graph_t<COO, Empty>;

// actors
inspector_t inspector; 
selector_t selector;
executor_t executor;
feature_t fets;
config_t conf;
stat_t stats;

struct CC:Functor<EC,int,Empty,Empty>{
  __device__ Status filter(int v, int u, Empty* e){
    if(*wa_of(v)==*wa_of(u)) return Fixed;
    else return Active;
  }

  __device__ void update(int v, int u, Empty* edata){
    int* rv = wa_of(*wa_of(v));
    int* ru = wa_of(*wa_of(u));

    if(*rv<*ru) atomicMin(ru, *rv);
    else if(*rv>*ru) atomicMin(rv, *ru);
  }
};

int* CC_CPU(edgelist_t<Empty> el){
  host_graph_t<CSR,Empty> hg;
  hg.build(el, true);
  LOG("generate CPU CC reference\n");
  double ms = mwtime();

  int* id = (int*)malloc(sizeof(int)*hg.nvertexs);
  for(int i=0 ; i<hg.nvertexs; i++) id[i]=i;
  std::queue<int> q;
  for(int i=0,c=0; i<hg.nvertexs; i++){
    if(id[i]!=i) continue;
    c=id[i];
    q.push(i);
    while(!q.empty()){
      int v = q.front();
      q.pop();
      int s = hg.start_pos[v];
      int e = (v==(hg.nvertexs-1)?hg.nedges:hg.start_pos[v+1]);
      for(int j=s; j<e; ++j){
        int u = hg.adj_list[j];
        if(id[u]==u){
          id[u] = c;
          q.push(u);
        }
      }
    }
  }
  double me = mwtime();
  LOG("CPU CC: %.3f ms\n", me-ms);
  return id;
}

void validation(int* lCPU, int* lGPU, int N){
  bool flag = true;
  for(int i=0; i<N; ++i){
    if(lCPU[i] != lGPU[i]){
      flag = false;
      puts("failed");
      std::cout << i << " " << lCPU[i] << " " << lGPU[i] << std::endl;
      break;
    }
  }
  if(flag) puts("passed");
}

template<typename G, typename F>
double run_cc(G g, F f){
  // step 1: initializing
  LOG(" -- Initializing\n");
  active_set_t as = build_active_set(g.nedges, conf);
  as.init(ALL_ACTIVE);
  int* dg_flag, *h_flag;
  cudaMalloc((void**)&dg_flag, sizeof(int));
  cudaMallocHost((void**)&h_flag, sizeof(int));

  // step 2: Execute Algorithm
  double s = mwtime();
  for(int level=0;;level++){
    inspector.inspect(as, g, f, stats, fets, conf);
    if(as.finish(g,f,conf)) break;
    selector.select(stats, fets, conf);
    executor.filter(as, g, f, stats, fets, conf);
    g.update_level();
    executor.expand(as, g, f, stats, fets, conf);
    p_jump(f, g.nvertexs, dg_flag, h_flag);
    if(as.finish(g,f,conf)) break;
  }
  double e = mwtime();

  return e-s;
}

int main(int argc, char* argv[]){
  //query_device_prop();
  parse_cmd(argc, argv, "CC");

  // step 1 : set features
  fets.centric = EC;
  fets.pattern = ASSO;

  // step 2 : init Graph & Algorithm
  auto g = build_graph<EC>(cmd_opt.path, fets, cmd_opt.with_header, cmd_opt.with_weight, cmd_opt.directed);
  CC f;
  f.data.build(g.hg.nvertexs);

  // step 3 : init data
  f.data.init_wa([](int i){return i;});
  init_conf(stats, fets, conf, g, f);

  // step 4 : execute Algorithm
  LOG(" -- Launching CC\n");
  double time = run_cc(g.dg, f);
    
  // step 5 : validation
  f.data.sync_wa();
  if(cmd_opt.validation){
    int* id = CC_CPU(g.el);
    validation(id, f.data.h_wa, g.hg.nvertexs);
  }

  LOG("GPU CC time: %.3f ms\n", time);
	std::cout << time << std::endl;

  return 0;
}
