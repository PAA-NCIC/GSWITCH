#ifndef __expand_VC_WM_CUH
#define __expand_VC_WM_CUH

#include "abstraction/config.cuh"
#include "data_structures/active_set.cuh"
#include "data_structures/functor.cuh"
#include "data_structures/graph.cuh"
#include "kernel_libs/kernel_fusion.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

template <ASFmt fmt, QueueMode M, typename G, typename F>
__global__ void __expand_VC_WM_fused_wtf(active_set_t as, G g, F f,
                                         config_t conf) {
  const int *__restrict__ strict_adj_list = g.dg_adj_list;

  // used for local storage
  __shared__ int tmp[3 * THDNUM_EXPAND];

  const int assize = ASProxy<fmt, M>::get_size_hard(as);
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  if (assize == 0) {
    if (gtid == 0)
      as.halt_device();
    return;
  }
  const int cosize = 32;
  const int phase = gtid & (cosize - 1);
  const int warp_id = threadIdx.x >> 5;
  const int OFFSET_warp = 3 * cosize * warp_id;
  const int OFFSET_start_pos = OFFSET_warp + cosize;
  const int OFFSET_odegree = OFFSET_warp + 2 * cosize;
  const int assize_align =
      (assize & (cosize - 1)) ? (((assize >> 5) + 1) << 5) : assize;
  Status want = conf.want();

  for (int idx = gtid; idx < assize_align; idx += STRIDE) {
    // step 1: load vertexs into share memory;
    int v;
    if (idx < assize)
      v = ASProxy<fmt, M>::fetch(as, idx, want);
    else
      v = -1;
    if (v >= 0) {
      tmp[OFFSET_warp + phase] = v;
      tmp[OFFSET_start_pos + phase] = tex1Dfetch<int>(g.dt_start_pos, v);
      tmp[OFFSET_odegree + phase] = tex1Dfetch<int>(g.dt_odegree, v);
    } else {
      tmp[OFFSET_warp + phase] = -1;
      tmp[OFFSET_odegree + phase] = 0;
    }

    // step 2: get sum of edges for these 32 vertexs and scan odegree;
    int nedges_warp = 0;
    int offset = 1;
    for (int d = cosize >> 1; d > 0; d >>= 1) {
      if (phase < d) {
        int ai = offset * (2 * phase + 1) - 1;
        int bi = offset * (2 * phase + 2) - 1;
        tmp[OFFSET_odegree + bi] += tmp[OFFSET_odegree + ai];
      }
      offset <<= 1;
    }

    nedges_warp = tmp[OFFSET_odegree + cosize - 1];
    if (!phase)
      tmp[OFFSET_odegree + cosize - 1] = 0;

    for (int d = 1; d < cosize; d <<= 1) {
      offset >>= 1;
      if (phase < d) {
        int ai = offset * (2 * phase + 1) - 1;
        int bi = offset * (2 * phase + 2) - 1;

        int t = tmp[OFFSET_odegree + ai];
        tmp[OFFSET_odegree + ai] = tmp[OFFSET_odegree + bi];
        tmp[OFFSET_odegree + bi] += t;
      }
    }

    int full_tier = assize_align - cosize;
    int width = idx < (full_tier) ? cosize : (assize - full_tier);

    // step 3: process 32 edges in parallel
    for (int i = phase; i < nedges_warp; i += cosize) {
      int id = __upper_bound(&tmp[OFFSET_odegree], width, i) - 1;
      // if(tmp[OFFSET_warp+id] < 0) continue;
      int ei = tmp[OFFSET_start_pos + id] + i - tmp[OFFSET_odegree + id];
      int u = __ldg(strict_adj_list + ei);
      int v = tmp[OFFSET_warp + id];
      auto vdata = f.emit(v, g.fetch_edata(ei), g);
      bool toprocess = true;

      // check 1: if idempotent, we can prune the redundant update (if has,
      // that's also OK)
      // TODO: this will not help to improve the performance, that's weird
      // toprocess = as.bitmap.mark_duplicate_lite(u);

      // check 2: if not push TO ALL, the target vertex must be Inactive
      // cond is provided by users to indicate whether u should accept the
      // update.
      // if(toprocess && !conf.conf_toall)
      if (toprocess)
        toprocess = f.cond(u, vdata, g);

      // if u pass all the checks, do the computation in the functor
      if (toprocess)
        toprocess = f.compAtomic(f.wa_of(u), vdata, g);

      // check 3:  enqueue the u only once. (if duplicate, wrong answer)
      // TODO: this will not help to improve the performance too, that's so
      // weird
      if (toprocess && !conf.pruning())
        toprocess = as.bitmap.mark_duplicate_atomic(u);

      // if u is updated successfully, write u to the queue directly.
      // cache mode.
      if (toprocess) {
        Qproxy<M>::push(as.queue, u);
      }
    } // for 32 edges
  } // for all the elements in the active set.
}

template <ASFmt fmt, QueueMode M, typename G, typename F>
__global__ void __expand_VC_WM_fused(active_set_t as, G g, F f, config_t conf) {
  const int *__restrict__ strict_adj_list = g.dg_adj_list;

  // used for local storage
  __shared__ int tmp[3 * THDNUM_EXPAND];

  const int assize = ASProxy<fmt, M>::get_size_hard(as);
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  if (assize == 0) {
    if (gtid == 0)
      as.halt_device();
    return;
  }
  const int cosize = 32;
  const int phase = gtid & (cosize - 1);
  const int warp_id = threadIdx.x >> 5;
  const int OFFSET_warp = 3 * cosize * warp_id;
  const int OFFSET_start_pos = OFFSET_warp + cosize;
  const int OFFSET_odegree = OFFSET_warp + 2 * cosize;
  const int assize_align =
      (assize & (cosize - 1)) ? (((assize >> 5) + 1) << 5) : assize;
  Status want = conf.want();

  for (int idx = gtid; idx < assize_align; idx += STRIDE) {
    // step 1: load vertexs into share memory;
    int v;
    if (idx < assize)
      v = ASProxy<fmt, M>::fetch(as, idx, want);
    else
      v = -1;
    if (v >= 0) {
      tmp[OFFSET_warp + phase] = v;
      tmp[OFFSET_start_pos + phase] = tex1Dfetch<int>(g.dt_start_pos, v);
      tmp[OFFSET_odegree + phase] = tex1Dfetch<int>(g.dt_odegree, v);
    } else {
      tmp[OFFSET_warp + phase] = -1;
      tmp[OFFSET_odegree + phase] = 0;
    }

    // step 2: get sum of edges for these 32 vertexs and scan odegree;
    int nedges_warp = 0;
    int offset = 1;
    for (int d = cosize >> 1; d > 0; d >>= 1) {
      if (phase < d) {
        int ai = offset * (2 * phase + 1) - 1;
        int bi = offset * (2 * phase + 2) - 1;
        tmp[OFFSET_odegree + bi] += tmp[OFFSET_odegree + ai];
      }
      offset <<= 1;
    }

    nedges_warp = tmp[OFFSET_odegree + cosize - 1];
    if (!phase)
      tmp[OFFSET_odegree + cosize - 1] = 0;

    for (int d = 1; d < cosize; d <<= 1) {
      offset >>= 1;
      if (phase < d) {
        int ai = offset * (2 * phase + 1) - 1;
        int bi = offset * (2 * phase + 2) - 1;

        int t = tmp[OFFSET_odegree + ai];
        tmp[OFFSET_odegree + ai] = tmp[OFFSET_odegree + bi];
        tmp[OFFSET_odegree + bi] += t;
      }
    }

    int full_tier = assize_align - cosize;
    int width = idx < (full_tier) ? cosize : (assize - full_tier);

    // step 3: process 32 edges in parallel
    for (int i = phase; i < nedges_warp; i += cosize) {
      int id = __upper_bound(&tmp[OFFSET_odegree], width, i) - 1;
      if (tmp[OFFSET_warp + id] < 0)
        continue;
      int ei = tmp[OFFSET_start_pos + id] + i - tmp[OFFSET_odegree + id];
      int u = __ldg(strict_adj_list + ei);
      int v = tmp[OFFSET_warp + id];
      auto vdata = f.emit(v, g.fetch_edata(ei), g);
      bool toprocess = true;

      // check 2: if not push TO ALL, the target vertex must be Inactive
      // cond is provided by users to indicate whether u should accept the
      // update.
      if (toprocess && !conf.conf_toall)
        toprocess = f.cond(u, vdata, g);

      // if u pass all the checks, do the computation in the functor
      if (toprocess) {
        // f.filter(u, g); // useless here
        toprocess = f.compAtomic(f.wa_of(u), vdata, g);
      }

      // check 3:  enqueue the u only once. (if duplicate, wrong answer)
      if (toprocess)
        toprocess = as.bitmap.mark_duplicate_atomic(u);

      // if u is updated successfully, write u to the queue directly.
      // cache mode.
      if (toprocess) {
        Qproxy<M>::push(as.queue, u);
      }
    } // for 32 edges
  } // for all the elements in the active set.
}

template <ASFmt fmt, QueueMode M, typename G, typename F>
__global__ void __expand_VC_WM(active_set_t as, G g, F f, config_t conf) {
  const int *__restrict__ strict_adj_list = g.dg_adj_list;

  // used for local storage
  __shared__ int tmp[3 * THDNUM_EXPAND];

  const int assize = ASProxy<fmt, M>::get_size(as);
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  const int cosize = 32;
  const int phase = gtid & (cosize - 1);
  const int warp_id = threadIdx.x >> 5;
  const int OFFSET_warp = 3 * cosize * warp_id;
  const int OFFSET_start_pos = OFFSET_warp + cosize;
  const int OFFSET_odegree = OFFSET_warp + 2 * cosize;
  // const int assize_align     =
  // (assize&(cosize-1))?(((assize>>5)+1)<<5):assize;
  const int assize_align = alignment(assize, cosize);
  Status want = conf.want();

  for (int idx = gtid; idx < assize_align; idx += STRIDE) {
    // step 1: load vertexs into share memory;
    int v;
    if (idx < assize)
      v = ASProxy<fmt, M>::fetch(as, idx, want);
    else
      v = -1;
    if (v >= 0) {
      tmp[OFFSET_warp + phase] = v;
      tmp[OFFSET_start_pos + phase] = tex1Dfetch<int>(g.dt_start_pos, v);
      tmp[OFFSET_odegree + phase] = tex1Dfetch<int>(g.dt_odegree, v);
    } else {
      tmp[OFFSET_warp + phase] = -1;
      tmp[OFFSET_odegree + phase] = 0;
    }

    // step 2: get sum of edges for these 32 vertexs and scan odegree;
    int nedges_warp = 0;
    int offset = 1;
    for (int d = cosize >> 1; d > 0; d >>= 1) {
      if (phase < d) {
        int ai = offset * (2 * phase + 1) - 1;
        int bi = offset * (2 * phase + 2) - 1;
        tmp[OFFSET_odegree + bi] += tmp[OFFSET_odegree + ai];
      }
      offset <<= 1;
    }

    nedges_warp = tmp[OFFSET_odegree + cosize - 1];
    if (!phase)
      tmp[OFFSET_odegree + cosize - 1] = 0;

    for (int d = 1; d < cosize; d <<= 1) {
      offset >>= 1;
      if (phase < d) {
        int ai = offset * (2 * phase + 1) - 1;
        int bi = offset * (2 * phase + 2) - 1;

        int t = tmp[OFFSET_odegree + ai];
        tmp[OFFSET_odegree + ai] = tmp[OFFSET_odegree + bi];
        tmp[OFFSET_odegree + bi] += t;
      }
    }

    int full_tier = assize_align - cosize;
    int width = idx < (full_tier) ? cosize : (assize - full_tier);

    // step 3: process 32 edges in parallel
    for (int i = phase; i < nedges_warp; i += cosize) {
      int id = __upper_bound(&tmp[OFFSET_odegree], width, i) - 1;
      if (tmp[OFFSET_warp + id] < 0)
        continue;
      int ei = tmp[OFFSET_start_pos + id] + i - tmp[OFFSET_odegree + id];
      int u = __ldg(strict_adj_list + ei);
      bool toprocess = true;

      // check 1: if idempotent, we can prune the redundant update
      if (toprocess && conf.pruning())
        toprocess = as.bitmap.mark_duplicate_lite(u);

      // check 2: if not push TO ALL, the target vertex must be Inactive
      if (toprocess && !conf.conf_toall)
        toprocess = as.bitmap.is_inactive(u);

      // if u pass all the checks, do the computation in the functor
      if (toprocess) {
        int v = tmp[OFFSET_warp + id];
        auto vdata = f.emit(v, g.fetch_edata(ei), g);
        f.compAtomic(f.wa_of(u), vdata, g);
      }
    } // for 32 edges
  } // for all the elements in the active set.
}

template <ASFmt fmt, QueueMode M, typename G, typename F>
__global__ void __rexpand_VC_WM(active_set_t as, G g, F f, config_t conf) {
  using edata_t = typename G::edge_t;
  using vdata_t = typename F::wa_t;
  const int *__restrict__ strict_adj_list =
      g.directed ? g.dgr_adj_list : g.dg_adj_list;
  edata_t *strict_edgedata = g.directed ? g.dgr_edgedata : g.dg_edgedata;

  __shared__ int tmp[3 * THDNUM_EXPAND];

  int assize = ASProxy<fmt, M>::get_size(as);
  int STRIDE = blockDim.x * gridDim.x;
  int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  int cosize = 32;
  int phase = gtid & (cosize - 1);
  int warp_id = threadIdx.x >> 5;
  int OFFSET_warp = 3 * cosize * warp_id;
  int OFFSET_start_pos = OFFSET_warp + cosize;
  int OFFSET_odegree = OFFSET_warp + 2 * cosize;
  int assize_align =
      (assize & (cosize - 1)) ? (((assize >> 5) + 1) << 5) : assize;
  Status want = conf.want();

  for (int idx = gtid; idx < assize_align; idx += STRIDE) {
    // step 1: load vertexs into share memory;
    int v;
    if (idx < assize)
      v = ASProxy<fmt, M>::fetch(as, idx, want);
    else
      v = -1;
    if (v >= 0) {
      tmp[OFFSET_warp + phase] = v;
      tmp[OFFSET_start_pos + phase] =
          g.get_in_start_pos(v); // tex1Dfetch<int>(g.dt_start_pos, v);
      tmp[OFFSET_odegree + phase] =
          g.get_in_degree(v); // tex1Dfetch<int>(g.dt_odegree, v);
    } else {
      tmp[OFFSET_warp + phase] = v;
      tmp[OFFSET_odegree + phase] = 0;
    }

    // step 2: get sum of edges for these 32 vertexs and scan odegree;
    int nedges_warp = 0;
    int offset = 1;
    for (int d = 32 >> 1; d > 0; d >>= 1) {
      if (phase < d) {
        int ai = offset * (2 * phase + 1) - 1;
        int bi = offset * (2 * phase + 2) - 1;
        tmp[OFFSET_odegree + bi] += tmp[OFFSET_odegree + ai];
      }
      offset <<= 1;
    }

    nedges_warp = tmp[OFFSET_odegree + 32 - 1];
    if (!phase)
      tmp[OFFSET_odegree + 32 - 1] = 0;

    for (int d = 1; d < 32; d <<= 1) {
      offset >>= 1;
      if (phase < d) {
        int ai = offset * (2 * phase + 1) - 1;
        int bi = offset * (2 * phase + 2) - 1;
        int t = tmp[OFFSET_odegree + ai];
        tmp[OFFSET_odegree + ai] = tmp[OFFSET_odegree + bi];
        tmp[OFFSET_odegree + bi] += t;
      }
    }

    // Binary search will not get the index which is out of range
    int full_tier = assize_align - cosize;
    int width = idx < (full_tier) ? cosize : (assize - full_tier);

    // step 3: process 32 edges in parallel
    int vote = 0;
    vdata_t vdata;
    for (int i = phase; i < nedges_warp; i += cosize) {
      vote = 0;
      int id = __upper_bound(&tmp[OFFSET_odegree], width, i) - 1;
      if (tmp[OFFSET_warp + id] < 0)
        continue; // v < 0
      int ei = tmp[OFFSET_start_pos + id] + i - tmp[OFFSET_odegree + id];
      int insegID = MIN((i - tmp[OFFSET_odegree + id]), phase);
      int rod =
          ((id == 31) ? (nedges_warp) : (tmp[OFFSET_odegree + id + 1])) - i - 1;
      int segsize = insegID + 1 + MIN(31 - phase, rod);

      int v = tmp[OFFSET_warp + id];
      int u = __ldg(strict_adj_list + ei);

      // rarely in the pull mode fusion
      // if(conf.conf_fromall || conf.conf_fuse_inspect || as.query(u) ==
      // Active){ Data source must be active all conf_fromall is enabled
      if (conf.conf_fromall || as.bitmap.is_active(u)) {
        vote = 1;
        vdata = f.emit(u, strict_edgedata + ei, g);
      }

      // reduce
      int offset = segsize;
      while (offset >> 1) {
        int th = offset >> 1;
        int delta = (offset + 1) >> 1;
        vdata_t _vdata = __exshfl_down(vdata, delta);
        int _vote = _shfl_down(vote, delta);
        if (insegID < th) {
          if (vote && _vote)
            f.comp(&vdata, _vdata, g);
          else if (_vote)
            vdata = _vdata;
          vote |= _vote;
        }
        offset = delta;
      }

      if (insegID == 0 && vote) {
        f.comp(f.wa_of(v), vdata, g);
      }
    }
  }
}

template <> struct ExpandProxy<VC, WM, Push> {
  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<CSR, E> g, F f,
                     config_t conf) {
    if (conf.conf_fuse_inspect) {
      if (conf.conf_pruning && conf.conf_asfmt == Queue &&
          as.queue.mode == Normal && conf.conf_toall == false) {
        __expand_VC_WM_fused_wtf<Queue, Normal>
            <<<conf.ctanum, conf.thdnum>>>(as, g, f, conf);
      } else {
        Launch_Expand_VC(WM_fused, as, g, f, conf);
      }
    } else {
      Launch_Expand_VC(WM, as, g, f, conf);
    }
    //__expand_VC_WM_fused<<<32,TH>>>(as, g, f, conf);
    //__expand_VC_WM<<<CN,TH>>>(as, g, f, conf);
  }

  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<COO, E> g, F f,
                     config_t conf) {}
};

template <> struct ExpandProxy<VC, WM, Pull> {
  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<CSR, E> g, F f,
                     config_t conf) {
    //__rexpand_VC_WM<<<CN,TH>>>(as, g, f, conf);
    Launch_RExpand_VC(WM, as, g, f, conf);
  }

  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<COO, E> g, F f,
                     config_t conf) {}
};

#endif
