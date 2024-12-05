#include "gswitch.h"
#include <iostream>
#include <queue>

using G = device_graph_t<CSR, Empty>;

// actors
inspector_t inspector;
selector_t selector;
executor_t executor;
feature_t fets;
config_t conf;
stat_t stats;

struct GC : Functor<VC, int, int, Empty> {
  __device__ Status filter(int vid, G g) {
    int *color = ra_of(vid);
    int *maxid = wa_of(vid);
    if (*color != -1)
      return Fixed;
    else if (*maxid < vid) {
      *color = g.get_level();
      return Fixed;
    } else {
      *maxid = -1;
      return Active;
    }
  }
  __device__ int emit(int vid, Empty *w, G g) {
    int color = *ra_of(vid);
    return (color == -1 ? vid : -1);
  }
  __device__ bool cond(int v, int maxid, G g) { return *ra_of(v) == -1; }
  __device__ bool comp(int *v, int maxid, G g) {
    if (*v < maxid)
      *v = maxid;
    return true;
  }
  __device__ bool compAtomic(int *v, int maxid, G g) {
    atomicMax(v, maxid);
    return true;
  }
};

void validation(host_graph_t<CSR, Empty> hg, int *A, int N) {
  bool flag = true;
  for (int i = 0; i < N; ++i) {
    int vcolor = A[i];
    int s = hg.start_pos[i];
    int e = (i == hg.nvertexs - 1) ? hg.nedges : hg.start_pos[i + 1];
    for (int j = s; j < e; ++j) {
      int u = hg.adj_list[j];
      int ucolor = A[u];
      if (ucolor == vcolor && ucolor != -1) {
        flag = false;
        puts("failed");
        std::cout << i << " " << u << "  have same color " << ucolor
                  << std::endl;
        break;
      }
      if (ucolor == -1 || vcolor == -1) {
        flag = false;
        puts("failed");
        std::cout << i << " " << vcolor << " or " << u << " " << ucolor
                  << "  is not colored " << std::endl;
        break;
      }
    }
    if (!flag)
      break;
  }
  if (flag)
    puts("passed");
}

template <typename G, typename F> double run_gc(G g, F f, int N) {
  // step 1: initializing
  LOG(" -- Initializing\n");
  active_set_t as = build_active_set(g.nvertexs, conf);
  as.init(ALL_ACTIVE);

  // step 2: Execute Algorithm
  double s = mwtime();
  int level;
  for (level = 0;; level++) {
    inspector.inspect(as, g, f, stats, fets, conf);
    if (as.finish(g, f, conf))
      break;
    selector.select(stats, fets, conf);
    executor.filter(as, g, f, stats, fets, conf);
    g.update_level();
    executor.expand(as, g, f, stats, fets, conf);
    // fets.record(); // for training
    if (as.finish(g, f, conf))
      break;
  }
  double e = mwtime();
  return e - s;
}

int main(int argc, char *argv[]) {
  parse_cmd(argc, argv, "GC");

  // step 1 : set features
  fets.centric = VC;
  fets.pattern = ASSO;
  fets.fromall = true;
  fets.toall = true;
  conf.conf_fixed = true;

  // step 2 : init Graph & Algorithm
  auto g = build_graph<VC>(cmd_opt.path, fets, cmd_opt.with_header,
                           cmd_opt.with_weight, cmd_opt.directed);
  // if (g.hg.nedges == 0) return 1;
  GC f;
  f.data.build(g.hg.nvertexs);

  // step 3 : choose root vertex
  f.data.init_ra([](int i) { return -1; });
  f.data.init_wa([](int i) { return -1; });
  init_conf(stats, fets, conf, g, f);

  // step 4 : execute Algorithm
  LOG(" -- Launching GC\n");
  double time = run_gc(g.dg, f, g.hg.nvertexs);

  // step 5 : validation
  if (cmd_opt.validation) {
    f.data.sync_ra();
    validation(g.hg, f.data.h_ra, g.hg.nvertexs);
  }

  LOG("GPU GC time: %.3f ms\n", time);
  std::cout << time << std::endl;
  // std::cout << fets.nvertexs << " "
  //<< fets.nedges << " "
  //<< fets.avg_deg << " "
  //<< fets.std_deg << " "
  //<< fets.range << " "8
  //<< fets.GI << " "
  //<< fets.Her << " "
  //<< time << std::endl;
  return 0;
}
