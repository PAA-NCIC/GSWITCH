#include <iostream>
#include <queue>

#include "gswitch.h"

using G = device_graph_t<CSR, Empty>;

// actors
inspector_t inspector;
selector_t selector;
executor_t executor;
feature_t fets;
config_t conf;
stat_t stats;

struct BFS : Functor<VC, int, Empty, Empty> {
  __device__ Status filter(int vid, G g) {
    int lvl = *wa_of(vid);
    if (lvl == g.get_level())
      return Active;
    else if (lvl < 0)
      return Inactive;
    else
      return Fixed;
  }
  __device__ int emit(int vid, Empty *w, G g) { return g.get_level(); }
  __device__ bool cond(int v, int newv, G g) { return *wa_of(v) == -1; }
  __device__ bool comp(int *v, int newv, G g) {
    *v = newv;
    return true;
  }
  __device__ bool compAtomic(int *v, int newv, G g) {
    *v = newv;
    return true;
  }
};

int *bfs_cpu(host_graph_t<CSR, Empty> hg, int root) {
  LOG("generate CPU BFS reference\n");
  double ms = mwtime();
  int *lvl = (int *)malloc(sizeof(int) * hg.nvertexs);
  memset(lvl, -1, sizeof(int) * hg.nvertexs);
  std::queue<int> q;
  lvl[root] = 0;
  q.push(root);
  while (!q.empty()) {
    int v = q.front();
    q.pop();
    int s = hg.start_pos[v];
    int e = (v == (hg.nvertexs - 1) ? hg.nedges : hg.start_pos[v + 1]);
    for (int j = s; j < e; ++j) {
      int u = hg.adj_list[j];
      if (lvl[u] == -1) {
        lvl[u] = lvl[v] + 1;
        q.push(u);
      }
    }
  }
  double me = mwtime();
  LOG("CPU BFS: %.3f ms\n", me - ms);
  return lvl;
}

void validation(int *lCPU, int *lGPU, int N) {
  bool flag = true;
  for (int i = 0; i < N; ++i) {
    if (lGPU[i] - lCPU[i] != 0) {
      flag = false;
      puts("failed");
      std::cout << i << " " << lGPU[i] << " " << lCPU[i] << std::endl;
      break;
    }
  }
  if (flag)
    puts("passed");
}

template <typename G, typename F>
double run_bfs(G g, F f, active_set_t &as, int root) {
  // step 1: initializing
  LOG(" -- Initializing\n");
  as.init(root);

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
  parse_cmd(argc, argv, "BFS");

  // step 1 : set features
  fets.centric = VC;
  fets.pattern = Idem;
  fets.fromall = false;
  fets.toall = false;
  conf.conf_pruning = true;

  // step 2 : init Graph & Algorithm
  auto g = build_graph<VC>(cmd_opt.path, fets, cmd_opt.with_header,
                           cmd_opt.with_weight, cmd_opt.directed);
  if (g.hg.nedges == 0)
    return 1;
  BFS f;
  f.data.build(g.hg.nvertexs);

  init_conf(stats, fets, conf, g, f);
  active_set_t as = build_active_set(g.dg.nvertexs, conf);

  // step 3 : choose root vertex
  int root = cmd_opt.src;
  if (root < 0)
    root = g.hg.random_root();
  LOG(" -- Root is: %d\n", root);
  fets.use_root = root;

  for (int iteration = 0; iteration < 3; ++iteration) {
    f.data.init_wa([root](int i) { return i == root ? 0 : -1; });

    // step 4 : execute Algorithm
    LOG(" -- Launching BFS\n");
    double time = run_bfs(g.dg, f, as, root);

    // reset
    reset_conf(stats, fets, conf, g, f);
    as.reset();
    g.dg.reset_level();

    // step 5 : validation
    f.data.sync_wa();
    if (cmd_opt.validation) {
      int *lvl = bfs_cpu(g.hg, root);
      validation(lvl, f.data.h_wa, g.hg.nvertexs);
    }

    LOG("GPU BFS time: %.3f ms\n", time);
    std::cout << time << std::endl;
  }
  // std::cout << fets.nvertexs << " "
  //<< fets.nedges << " "
  //<< fets.avg_deg << " "
  //<< fets.std_deg << " "
  //<< fets.range << " "
  //<< fets.GI << " "
  //<< fets.Her << " "
  //<< time << std::endl;
  return 0;
}
