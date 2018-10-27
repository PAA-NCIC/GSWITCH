#include <iostream>
#include <algorithm>
#include "gswitch.h"

const float damp = 0.85f;
const float eps = 0.01f;

using G = device_graph_t<CSR, Empty>;

// actors
inspector_t inspector; 
selector_t selector;
executor_t executor;
feature_t fets;
config_t conf;
stat_t stats;

struct PR:Functor<VC,float,float,Empty>{
  __device__ Status filter(int vid, G g){
    float* cache = wa_of(vid);
    float* pr    = ra_of(vid);
    int od = g.get_out_degree(vid);
    float newpr   = (*cache*damp + 1-damp);
    if(od) newpr /= od;
    float err     = (newpr-*pr)/(*pr);
    *cache = 0;
    *pr    = newpr;
    if(err > -eps && err < eps) return Inactive;
    return Active;
  }
  __device__ float emit(int vid, Empty *em, G g) { return *ra_of(vid); }
  __device__ bool comp(float* v, float newv, G g) { *v += newv;return true; }
  __device__ bool compAtomic(float* v, float newv, G g) { atomicAdd(v, newv);return true; } 
};

void validation(host_graph_t<CSR,Empty> hg, float* pr){
  for(int i = 0; i<hg.nvertexs; ++i) if(hg.odegrees[i]) pr[i] *= hg.odegrees[i];
  bool flag = true;
  for(int i=0;i<hg.nvertexs;++i){
    int s = hg.start_pos[i];
    int e = (i==hg.nvertexs-1)?hg.nedges:hg.start_pos[i+1];
    float sum = 0;
    for(int j = s; j < e; ++ j){
      int u = hg.adj_list[j];
      sum += pr[u]/hg.odegrees[u];
    }
    sum = sum*damp + 1-damp;
    float diff = (sum - pr[i])/sum;
    if(fabs(diff)<eps) flag&=true;
    else flag&=false;
    if(!flag){
      puts("failed");
      std::cout << "id: " << i << "  "<< sum << " " << pr[i] << " " << hg.odegrees[i] << std::endl;
      break;
    }
  }
  if(flag) puts("passed");
  if(cmd_opt.verbose){
    std::sort(pr,pr+hg.nvertexs);
    std::cout << "top 10 pagerank:" << std::endl;
    for(int i=1;i<=10;++i){
      std::cout << pr[hg.nvertexs-i] << std::endl;
    }
  }
}


template<typename G, typename F>
double run_pagerank(G g, F f){
  // step 1: initializing
  LOG(" -- Initializing\n");
  active_set_t as = build_active_set(g.nvertexs, conf);
  as.init(ALL_ACTIVE);

  double s = mwtime();
  // step 2: Execute Algorithm
  for(int level=0;;level++){
    inspector.inspect(as, g, f, stats, fets, conf);
    if(as.finish(g,f,conf)) break;
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
  parse_cmd(argc, argv, "PageRank");

  // step 1 : set features
  fets.centric = VC;
  fets.pattern = ASSO;
  fets.fromall = true;
  fets.toall = true;

  // step 2 : init Graph & Algorithm
  auto g = build_graph<VC>(cmd_opt.path, fets, cmd_opt.with_header, cmd_opt.with_weight, cmd_opt.directed);
  PR f;
  f.data.build(g.hg.nvertexs);
  f.data.init_wa([](int i){ return 0; });
  f.data.init_ra([g](int i){ 
    if(g.hg.odegrees[i]) return (1-damp)/g.hg.odegrees[i];
    else return (1-damp);
  });
  f.data.set_zero(0.0f);
  init_conf(stats, fets, conf, g, f);

  // step 3 : execute Algorithm
  LOG(" -- Launching PageRank\n");
  double time = run_pagerank(g.dg, f);
    
  // step 4 : validation
  f.data.sync_ra();
  if(cmd_opt.validation){
    validation(g.hg, f.data.h_ra);
  }
  LOG("GPU PageRank time: %.3f ms\n", time);
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


