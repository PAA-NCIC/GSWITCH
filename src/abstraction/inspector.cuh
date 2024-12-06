#ifndef __INSPECTOR_CUH
#define __INSPECTOR_CUH

#include "abstraction/config.cuh"
#include "abstraction/features.cuh"
#include "abstraction/statistics.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

template <typename G, typename F>
__global__ void __inspect_EC(active_set_t as, G g, F f, stat_t stat,
                             config_t conf) {
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  const int lane = threadIdx.x & 31;
  int active_num = 0;
  for (int idx = gtid; idx < as.size; idx += STRIDE) {
    bool tag_previous = as.bitmap.is_active(idx);
    bool tag_next = false;
    if (tag_previous) {
      packed_t vv = g.dg_coo[idx];
      int v0 = vv >> 32;
      int v1 = vv & ((1ll << 32) - 1);
      tag_next = (f.filter(v0, v1, NULL) == Active);
    }
    unsigned int active = _ballot(tag_next);
    if (lane == 0)
      as.bitmap.active.store_word_as_int(idx, active);
    if (tag_previous & tag_next)
      active_num++;
  }

  __syncthreads();
  active_num = blockReduceSum(active_num);
  if (!threadIdx.x)
    atomicAdd(&stat.dg_runtime_info[0], active_num);
}

template <typename G, typename F>
__global__ void __inspect_VC(active_set_t as, G g, F f, stat_t stat,
                             config_t conf) {
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  const int lane = threadIdx.x & 31;
  const int *__restrict__ odegree = g.dg_odegree;
  // const int* __restrict__ idegree = g.directed? g.dgr_odegree : g.dg_odegree;
  int active_num = 0;
  int inactive_num = 0;
  int push_workload = 0;
  int pull_workload = 0;
  int max_push_deg = 0;
  int max_pull_deg = 0;

  for (int idx = gtid; idx < as.size; idx += STRIDE) {
    int od = __ldg(odegree + idx);
    int id = od;

    Status s = f.filter(idx, g);
    bool tag_active = (s == Active);
    bool tag_inactive = (s != Inactive); // only inactive lane hold 0.
    unsigned int active = _ballot(tag_active);
    unsigned int inactive = _ballot(tag_inactive);
    if (lane == 0)
      as.bitmap.active.store_word_as_int(idx, active);
    if (lane == 0)
      as.bitmap.inactive.store_word_as_int(idx, inactive);

    if (tag_active) {
      active_num++;
      push_workload += od;
      max_push_deg = MAX(max_push_deg, od);
    }

    if (!tag_inactive) { // yes, it's inversed
      inactive_num++;
      pull_workload += id;
      max_pull_deg = MAX(max_pull_deg, id);
    }
  }

  __syncthreads();
  active_num = blockReduceSum(active_num);
  if (!threadIdx.x)
    atomicAdd(&stat.dg_runtime_info[0], active_num);

  __syncthreads();
  inactive_num = blockReduceSum(inactive_num);
  if (!threadIdx.x)
    atomicAdd(&stat.dg_runtime_info[1], inactive_num);

  __syncthreads();
  push_workload = blockReduceSum(push_workload);
  if (!threadIdx.x)
    atomicAdd(&stat.dg_runtime_info[2], push_workload);

  __syncthreads();
  pull_workload = blockReduceSum(pull_workload);
  if (!threadIdx.x)
    atomicAdd(&stat.dg_runtime_info[3], pull_workload);

  __syncthreads();
  max_push_deg = blockReduceSum(max_push_deg);
  if (!threadIdx.x)
    atomicMax(&stat.dg_runtime_info[4], max_push_deg);

  __syncthreads();
  max_pull_deg = blockReduceSum(max_pull_deg);
  if (!threadIdx.x)
    atomicAdd(&stat.dg_runtime_info[5], max_pull_deg);
}

struct inspector_t {

  // ################ VC
  template <typename E, typename F>
  void inspect(active_set_t &as, device_graph_t<CSR, E> g, F &f, stat_t &stat,
               feature_t &fets, config_t &conf) {
    TRACE();
    stat.last_filter_time = mwtime();

    f.data.window.adjust(fets, conf);

    bool need_inspect = true;
    if (conf.conf_fuse_inspect)
      need_inspect = false;
    init_fets(stat, fets, g.nvertexs, g.nedges);

    // the first level don't need insecpt
    // normal mode will enter this ! kernel-fusion mode will not enter this.
    if (need_inspect && conf.conf_first_round) {
      set_first_fets(as, g, f, stat, fets, conf, g.nvertexs, g.nedges);
      need_inspect = false;
    }

    if (need_inspect) {
      __inspect_VC<<<CTANUM, THDNUM>>>(as, g, f, stat, conf); // to bitmap
      // LOG("%d ", as.queue.get_current_queue().debug());
      set_fets(as, g, f, stat, fets, conf, g.nvertexs, g.nedges);
    }
    fets.flatten(); // this will contaminate the origin data
  }

  // ################ EC
  template <typename E, typename F>
  void inspect(active_set_t &as, device_graph_t<COO, E> g, F &f, stat_t &stat,
               feature_t &fets, config_t &conf) {
    TRACE();
    stat.last_filter_time = mwtime();
    init_fets(stat, fets, g.nvertexs, g.nedges);
    // the first level don't need insecpt
    if (conf.conf_first_round) {
      conf.conf_first_round = false;
      fets.active_vertex = g.nedges;
      fets.active_vertex_ratio = 1;
      return;
    }
    __inspect_EC<<<CTANUM, THDNUM>>>(as, g, f, stat, conf);
    set_fets(as, g, f, stat, fets, conf, g.nvertexs, g.nedges);
    fets.flatten();
  }

  void init_fets(stat_t &stat, feature_t &fets, int nvertexs, int edges) {
    fets.level = stat.level;

    fets.active_vertex = 0;
    fets.active_vertex_ratio = 0;
    fets.unactive_vertex = nvertexs;
    fets.unactive_vertex_ratio = 1;

    fets.growing_rate = fets.push_workload;
    fets.push_workload = 0;
    fets.pull_workload = edges;
    fets.push_workload_ratio = 0;
    fets.pull_workload_ratio = 1;

    fets.cur_avg_deg_active = 0;
    fets.cur_avg_deg_unactive = fets.avg_deg;
  }

  template <typename G, typename F>
  void set_first_fets(active_set_t &as, G g, F &f, stat_t &stat,
                      feature_t &fets, config_t conf, int nvertexs,
                      int nedges) {
    if (fets.use_root < 0) {
      fets.active_vertex = nvertexs;
      fets.unactive_vertex = 0;
      fets.push_workload = nedges;
      fets.pull_workload = 0;
    } else {
      fets.active_vertex = 1;
      fets.unactive_vertex = nvertexs - 1;
      fets.push_workload = fets.first_workload;
      fets.pull_workload = nedges - fets.first_workload;
    }
    fets.active_vertex_ratio = (double)fets.active_vertex / nvertexs;
    fets.unactive_vertex_ratio = (double)fets.unactive_vertex / nvertexs;
    fets.push_workload_ratio = (double)fets.push_workload / nedges;
    fets.pull_workload_ratio = (double)fets.pull_workload / nedges;
    fets.cur_avg_deg_active = DIV(fets.push_workload, fets.active_vertex);
    fets.cur_avg_deg_unactive = DIV(fets.pull_workload, fets.unactive_vertex);
    fets.cur_max_deg_active = fets.push_workload;
    fets.cur_max_deg_unactive = fets.avg_deg;
  }

  template <typename G, typename F>
  void set_fets(active_set_t &as, G g, F &f, stat_t &stat, feature_t &fets,
                config_t conf, int nvertexs, int nedges) {
    stat.collect(6);
    // cudaThreadSynchronize();
    stat.clean(); // for next one;

    fets.active_vertex = stat.active_num();
    fets.unactive_vertex = stat.unactive_num();
    fets.active_vertex_ratio = (double)stat.active_num() / nvertexs;
    fets.unactive_vertex_ratio = (double)stat.unactive_num() / nvertexs;

    bool done = false;
    if (fets.active_vertex <= 0)
      done = true;
    if (fets.pattern == Idem && fets.unactive_vertex <= 0)
      done = true;

    if (done) {
      if (conf.conf_qmode == Cached)
        as.queue.swap(); // to make the cursor of cached queue behave the same
                         // as the fusion mode.
      as.halt_host();
    } else {
      LOG("[%d] ", fets.active_vertex);
    }

    fets.growing_rate = (0.0 + stat.push_workload()) / fets.growing_rate;
    fets.push_workload = stat.push_workload();
    fets.pull_workload = stat.pull_workload();
    fets.push_workload_ratio = (double)stat.push_workload() / nedges;
    fets.pull_workload_ratio = (double)stat.pull_workload() / nedges;
    fets.cur_avg_deg_active = DIV(fets.push_workload, fets.active_vertex);
    fets.cur_avg_deg_unactive = DIV(fets.pull_workload, fets.unactive_vertex);
    fets.cur_max_deg_active = stat.max_deg_active();
    fets.cur_max_deg_unactive = stat.max_deg_unactive();
  }

  bool fromall, toall;
};

#endif
