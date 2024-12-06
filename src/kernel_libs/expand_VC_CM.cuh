#ifndef __expand_VC_CM_CUH
#define __expand_VC_CM_CUH

#include "abstraction/config.cuh"
#include "data_structures/active_set.cuh"
#include "data_structures/functor.cuh"
#include "data_structures/graph.cuh"
#include "kernel_libs/kernel_fusion.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

template <ASFmt fmt, QueueMode M, typename G, typename F>
__global__ void __expand_VC_CM_fused(active_set_t as, G g, F f, config_t conf) {
  const int *__restrict__ strict_adj_list = g.dg_adj_list;

  __shared__ int tmp[3 * THDNUM_EXPAND];

  const int assize = ASProxy<fmt, M>::get_size_hard(as);
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  if (assize == 0) {
    if (gtid == 0)
      as.halt_device();
    return;
  }
  const int cosize = blockDim.x;
  const int phase = threadIdx.x;
  const int blk_id = 0;
  const int OFFSET_blk = 3 * cosize * blk_id;
  const int OFFSET_start_pos = OFFSET_blk + cosize;
  const int OFFSET_odegree = OFFSET_blk + 2 * cosize;
  // const int assize_align     =
  // (assize&(cosize-1))?(((assize>>8)+1)<<8):assize;
  const int assize_align = alignment(assize, cosize);
  Status want = conf.want();

  for (int idx = gtid; idx < assize_align; idx += STRIDE) {
    __syncthreads();
    // step 1: load vertexs into share memory;
    int v;
    if (idx < assize)
      v = ASProxy<fmt, M>::fetch(as, idx, want);
    else
      v = -1;
    if (v >= 0) {
      tmp[OFFSET_blk + phase] = v;
      tmp[OFFSET_start_pos + phase] = tex1Dfetch<int>(g.dt_start_pos, v);
      tmp[OFFSET_odegree + phase] = tex1Dfetch<int>(g.dt_odegree, v);
    } else {
      tmp[OFFSET_blk + phase] = -1;
      tmp[OFFSET_odegree + phase] = 0;
    }
    __syncthreads();
    // step 2: get sum of edges for these cosize vertexs and scan odegree;
    int nedges_blk = 0;
    int offset = 1;
    for (int d = cosize >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (phase < d) {
        int ai = offset * (2 * phase + 1) - 1;
        int bi = offset * (2 * phase + 2) - 1;
        tmp[OFFSET_odegree + bi] += tmp[OFFSET_odegree + ai];
      }
      offset <<= 1;
    }

    __syncthreads();
    nedges_blk = tmp[OFFSET_odegree + cosize - 1];
    __syncthreads();
    if (!phase)
      tmp[OFFSET_odegree + cosize - 1] = 0;
    __syncthreads();

    for (int d = 1; d < cosize; d <<= 1) {
      offset >>= 1;
      __syncthreads();
      if (phase < d) {
        int ai = offset * (2 * phase + 1) - 1;
        int bi = offset * (2 * phase + 2) - 1;

        int t = tmp[OFFSET_odegree + ai];
        tmp[OFFSET_odegree + ai] = tmp[OFFSET_odegree + bi];
        tmp[OFFSET_odegree + bi] += t;
      }
    }
    __syncthreads();

    int full_tier = assize_align - cosize;
    int width = idx < (full_tier) ? cosize : (assize - full_tier);

    // step 3: process cosize edges in parallel
    for (int i = phase; i < nedges_blk; i += cosize) {
      int id = __upper_bound(&tmp[OFFSET_odegree], width, i) - 1;
      if (tmp[OFFSET_blk + id] < 0)
        continue;
      int ei = tmp[OFFSET_start_pos + id] + i - tmp[OFFSET_odegree + id];
      int u = __ldg(strict_adj_list + ei);
      int v = tmp[OFFSET_blk + id];
      auto vdata = f.emit(v, g.fetch_edata(ei), g);
      bool toprocess = true;

      // check 1: if idempotent, we can prune the redundant update (if has,
      // that's also OK)
      if (toprocess && conf.pruning())
        toprocess = as.bitmap.mark_duplicate_lite(u);

      // check 2: if not push TO ALL, the target vertex must be Inactive
      // cond is provided by users to indicate whether u should accept the
      // update.
      if (toprocess && !conf.conf_toall)
        toprocess = f.cond(u, vdata, g);

      // if u pass all the checks, do the computation in the functor
      if (toprocess) {
        // f.filter(u, g);// useless
        toprocess = f.compAtomic(f.wa_of(u), vdata, g);
      }

      // check 3:  enqueue the u only once. (if duplicate, wrong answer)
      if (toprocess && !conf.pruning())
        toprocess = as.bitmap.mark_duplicate_atomic(u);

      // if u is updated successfully, write u to the queue directly
      // atomic mode.
      if (toprocess) {
        Qproxy<M>::push(as.queue, u);
      }
    } // for
  } // for
}

template <ASFmt fmt, QueueMode M, typename G, typename F>
__global__ void __expand_VC_CM(active_set_t as, G g, F f, config_t conf) {
  const int *__restrict__ strict_adj_list = g.dg_adj_list;

  __shared__ int tmp[3 * THDNUM_EXPAND];

  const int assize = ASProxy<fmt, M>::get_size(as); // bitmap or queue?
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  // if(assize==0) { if(gtid==0) as.halt_device();return; }
  const int cosize = blockDim.x;
  const int phase = threadIdx.x;
  const int blk_id = 0;
  const int OFFSET_blk = 3 * cosize * blk_id;
  const int OFFSET_start_pos = OFFSET_blk + cosize;
  const int OFFSET_odegree = OFFSET_blk + 2 * cosize;
  // const int assize_align    =
  // (assize&(cosize-1))?(((assize>>10)+1)<<10):assize;
  const int assize_align = alignment(assize, cosize);
  Status want = conf.want();

  for (int idx = gtid; idx < assize_align; idx += STRIDE) {
    __syncthreads();
    // step 1: load vertexs into share memory;
    int v;
    if (idx < assize)
      v = ASProxy<fmt, M>::fetch(as, idx, want);
    else
      v = -1;
    if (v >= 0) {
      tmp[OFFSET_blk + phase] = v;
      tmp[OFFSET_start_pos + phase] = tex1Dfetch<int>(g.dt_start_pos, v);
      tmp[OFFSET_odegree + phase] = tex1Dfetch<int>(g.dt_odegree, v);
    } else {
      tmp[OFFSET_blk + phase] = -1;
      tmp[OFFSET_odegree + phase] = 0;
    }
    __syncthreads();
    // step 2: get sum of edges for these cosize vertexs and scan odegree;
    int nedges_blk = 0;
    int offset = 1;
    for (int d = cosize >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (phase < d) {
        int ai = offset * (2 * phase + 1) - 1;
        int bi = offset * (2 * phase + 2) - 1;
        tmp[OFFSET_odegree + bi] += tmp[OFFSET_odegree + ai];
      }
      offset <<= 1;
    }

    __syncthreads();
    nedges_blk = tmp[OFFSET_odegree + cosize - 1];
    __syncthreads();
    if (!phase)
      tmp[OFFSET_odegree + cosize - 1] = 0;
    __syncthreads();

    for (int d = 1; d < cosize; d <<= 1) {
      offset >>= 1;
      __syncthreads();
      if (phase < d) {
        int ai = offset * (2 * phase + 1) - 1;
        int bi = offset * (2 * phase + 2) - 1;

        int t = tmp[OFFSET_odegree + ai];
        tmp[OFFSET_odegree + ai] = tmp[OFFSET_odegree + bi];
        tmp[OFFSET_odegree + bi] += t;
      }
    }
    __syncthreads();

    int full_tier = assize_align - cosize;
    int width = idx < (full_tier) ? cosize : (assize - full_tier);

    // step 3: process cosize edges in parallel
    for (int i = phase; i < nedges_blk; i += cosize) {
      int id = __upper_bound(&tmp[OFFSET_odegree], width, i) - 1;
      if (tmp[OFFSET_blk + id] < 0)
        continue;
      int ei = tmp[OFFSET_start_pos + id] + i - tmp[OFFSET_odegree + id];
      int u = __ldg(strict_adj_list + ei);
      bool toprocess = true;

      // check 1: if idempotent, we can prune the redundant update (if has,
      // that's also OK)
      if (toprocess && conf.pruning())
        toprocess = as.bitmap.mark_duplicate_lite(u);

      // check 2: if not push TO ALL, the target vertex must be Inactive
      if (toprocess && !conf.conf_toall)
        toprocess = as.bitmap.is_inactive(u);

      // if u pass all the checks, do the computation in the functor
      if (toprocess) {
        int v = tmp[OFFSET_blk + id];
        auto vdata = f.emit(v, g.fetch_edata(ei), g);
        f.compAtomic(f.wa_of(u), vdata, g);
      }
    } // for
  } // for
}

template <ASFmt fmt, QueueMode M, typename G, typename F>
__global__ void __rexpand_VC_CM(active_set_t as, G g, F f, config_t conf) {
  const int *__restrict__ strict_adj_list = g.dg_adj_list;

  __shared__ int tmp[3 * THDNUM_EXPAND];

  const int assize = ASProxy<fmt, M>::get_size(as); // bitmap or queue?
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  if (assize == 0) {
    if (gtid == 0)
      as.halt_device();
    return;
  }
  const int cosize = blockDim.x;
  const int phase = threadIdx.x;
  const int blk_id = 0;
  const int OFFSET_blk = 3 * cosize * blk_id;
  const int OFFSET_start_pos = OFFSET_blk + cosize;
  const int OFFSET_odegree = OFFSET_blk + 2 * cosize;
  // const int assize_align    =
  // (assize&(cosize-1))?(((assize>>10)+1)<<10):assize;
  const int assize_align = alignment(assize, cosize);
  Status want = conf.want();

  for (int idx = gtid; idx < assize_align; idx += STRIDE) {
    __syncthreads();
    // step 1: load vertexs into share memory;
    int v;
    if (idx < assize)
      v = ASProxy<fmt, M>::fetch(as, idx, want);
    else
      v = -1;
    if (v >= 0) {
      tmp[OFFSET_blk + phase] = v;
      tmp[OFFSET_start_pos + phase] = tex1Dfetch<int>(g.dt_start_pos, v);
      tmp[OFFSET_odegree + phase] = tex1Dfetch<int>(g.dt_odegree, v);
    } else {
      tmp[OFFSET_blk + phase] = -1;
      tmp[OFFSET_odegree + phase] = 0;
    }
    __syncthreads();
    // step 2: get sum of edges for these cosize vertexs and scan odegree;
    int nedges_blk = 0;
    int offset = 1;
    for (int d = cosize >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (phase < d) {
        int ai = offset * (2 * phase + 1) - 1;
        int bi = offset * (2 * phase + 2) - 1;
        tmp[OFFSET_odegree + bi] += tmp[OFFSET_odegree + ai];
      }
      offset <<= 1;
    }

    __syncthreads();
    nedges_blk = tmp[OFFSET_odegree + cosize - 1];
    __syncthreads();
    if (!phase)
      tmp[OFFSET_odegree + cosize - 1] = 0;
    __syncthreads();

    for (int d = 1; d < cosize; d <<= 1) {
      offset >>= 1;
      __syncthreads();
      if (phase < d) {
        int ai = offset * (2 * phase + 1) - 1;
        int bi = offset * (2 * phase + 2) - 1;

        int t = tmp[OFFSET_odegree + ai];
        tmp[OFFSET_odegree + ai] = tmp[OFFSET_odegree + bi];
        tmp[OFFSET_odegree + bi] += t;
      }
    }
    __syncthreads();

    int full_tier = assize_align - cosize;
    int width = idx < (full_tier) ? cosize : (assize - full_tier);

    // step 3: process cosize edges in parallel
    for (int i = phase; i < nedges_blk; i += cosize) {
      int id = __upper_bound(&tmp[OFFSET_odegree], width, i) - 1;
      if (tmp[OFFSET_blk + id] < 0)
        continue;
      int ei = tmp[OFFSET_start_pos + id] + i - tmp[OFFSET_odegree + id];
      int u = __ldg(strict_adj_list + ei);
      bool toprocess = true;

      // data source must from Active vertex or the conf_fromall is enabled
      if (toprocess && !conf.conf_fromall)
        toprocess = as.bitmap.is_active(u);

      // if u pass all the checks, do the computation in the functor
      if (toprocess) {
        int v = tmp[OFFSET_blk + id];
        auto vdata = f.emit(u, g.fetch_edata(ei), g);
        f.compAtomic(f.wa_of(v), vdata, g);
      }
    } // for
  } // for
}

template <> struct ExpandProxy<VC, CM, Push> {
  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<CSR, E> g, F f,
                     config_t conf) {
    if (conf.conf_fuse_inspect)
      Launch_Expand_VC(CM_fused, as, g, f,
                       conf) else Launch_Expand_VC(CM, as, g, f, conf);
  }

  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<COO, E> g, F f,
                     config_t conf) {
    CudaCheckError();
  }
};

template <> struct ExpandProxy<VC, CM, Pull> {
  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<CSR, E> g, F f,
                     config_t conf) {
    Launch_RExpand_VC(CM, as, g, f, conf);
  }

  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<COO, E> g, F f,
                     config_t conf) {}
};

#endif
