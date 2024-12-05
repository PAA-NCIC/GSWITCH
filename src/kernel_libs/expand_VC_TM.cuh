#ifndef __expand_VC_TM_CUH
#define __expand_VC_TM_CUH

#include "abstraction/config.cuh"
#include "data_structures/active_set.cuh"
#include "data_structures/functor.cuh"
#include "data_structures/graph.cuh"
#include "kernel_libs/kernel_fusion.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

template <ASFmt fmt, QueueMode M, typename G, typename F>
__global__ void __expand_VC_TM_fused(active_set_t as, G g, F f, config_t conf) {
  const int *__restrict__ strict_adj_list = g.dg_adj_list;

  int assize = ASProxy<fmt, M>::get_size_hard(as);
  int STRIDE = blockDim.x * gridDim.x;
  int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  if (assize == 0) {
    if (gtid == 0)
      as.halt_device();
    return;
  }
  Status want = conf.want();

  for (int idx = gtid; idx < assize; idx += STRIDE) {
    int v = ASProxy<fmt, M>::fetch(as, idx, want);
    if (v < 0)
      continue;
    int start = tex1Dfetch<int>(g.dt_start_pos, v);
    int end = start + tex1Dfetch<int>(g.dt_odegree, v);
    for (int i = start; i < end; ++i) {
      int u = strict_adj_list[i];
      auto vdata =
          f.emit(v, g.fetch_edata(i), g); // f.emit(u, g.dg_edgedata+i, g);
      bool toprocess = true;

      // check 1: if idempotent, we can prune the redundant update (if has,
      // that's also OK)
      if (toprocess && conf.pruning())
        toprocess = as.bitmap.mark_duplicate_lite(u);

      // check 2: the push target must be inactive or the conf_toall is enabled
      // cond is provided by users to indicate whether u should accept the
      // update.
      if (toprocess && !conf.conf_toall) {
        toprocess = f.cond(u, vdata, g);
      }

      if (toprocess) {
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
__global__ void __expand_VC_TM(active_set_t as, G g, F f, config_t conf) {
  const int *__restrict__ strict_adj_list = g.dg_adj_list;

  int assize = ASProxy<fmt, M>::get_size(as);
  int STRIDE = blockDim.x * gridDim.x;
  int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  if (assize == 0) {
    if (gtid == 0)
      as.halt_device();
    return;
  }
  Status want = conf.want();

  for (int idx = gtid; idx < assize; idx += STRIDE) {
    int v = ASProxy<fmt, M>::fetch(as, idx, want);
    if (v < 0)
      continue;
    int start = tex1Dfetch<int>(g.dt_start_pos, v);
    int end = start + tex1Dfetch<int>(g.dt_odegree, v);
    for (int i = start; i < end; ++i) {
      int u = strict_adj_list[i];
      bool toprocess = true;

      // check 1: if idempotent, we can prune the redundant update (if has,
      // that's also OK)
      if (toprocess && conf.pruning())
        toprocess = as.bitmap.mark_duplicate_lite(u);

      // check 2: the push target must be inactive or the conf_toall is enabled
      if (toprocess && !conf.conf_toall) {
        toprocess = as.bitmap.is_inactive(u);
      }

      if (toprocess) {
        auto vdata =
            f.emit(v, g.fetch_edata(i), g); // f.emit(u, g.dg_edgedata+i, g);
        f.compAtomic(f.wa_of(u), vdata, g);
      }
    }
  }
}

template <typename G, typename F>
__global__ void __rexpand_VC_TM_BITMAP(active_set_t as, G g, F f,
                                       config_t conf) {
  using edata_t = typename G::edge_t;
  using vdata_t = typename F::wa_t;
  const int *__restrict__ strict_adj_list =
      g.directed ? g.dgr_adj_list : g.dg_adj_list;
  edata_t *strict_edgedata = g.directed ? g.dgr_edgedata : g.dg_edgedata;

  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  const int n_bytes = as.bitmap.inactive.n_words;
  const int *bits = (int *)(void *)as.bitmap.inactive.dg_bits;
  const int lane = threadIdx.x & 31;
  const int n_bytes_aligned =
      (n_bytes & 31) ? (((n_bytes >> 5) + 1) << 5) : n_bytes;

  for (int idx = gtid; idx < n_bytes_aligned; idx += STRIDE) {
    int inactive = 0;
    if (idx < n_bytes)
      inactive = ~bits[idx];

    for (int l = 0; l < 32; ++l) {
      int w = _shfl(inactive, l);
      int lid = idx - lane + l;
      if (w & (1 << lane)) {
        int v = (lid << 5) + lane;
        int start = g.get_in_start_pos(v);
        int end = start + g.get_in_degree(v);
        for (int i = start; i < end; ++i) {
          int u = strict_adj_list[i];
          bool toprocess = true;

          // if(!conf.conf_fromall && !conf.conf_fuse_inspect && as.query(u) !=
          // Active)
          //  data source must from Active vertex or the conf_fromall is enabled
          if (toprocess && !conf.conf_fromall)
            toprocess = as.bitmap.is_active(u);

          if (toprocess) {
            auto vdata = f.emit(u, strict_edgedata + i, g);
            f.comp(f.wa_of(v), vdata, g);
            if (conf.pruning())
              break; // idempotent optimization
          }
        }
      }
    }
  }
}

template <ASFmt fmt, QueueMode M, typename G, typename F>
__global__ void __rexpand_VC_TM(active_set_t as, G g, F f, config_t conf) {
  using edata_t = typename G::edge_t;
  using vdata_t = typename F::wa_t;
  const int *__restrict__ strict_adj_list =
      g.directed ? g.dgr_adj_list : g.dg_adj_list;
  edata_t *strict_edgedata = g.directed ? g.dgr_edgedata : g.dg_edgedata;

  int assize = ASProxy<fmt, M>::get_size(as);
  int STRIDE = blockDim.x * gridDim.x;
  int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  Status want = conf.want();

  for (int idx = gtid; idx < assize; idx += STRIDE) {
    int v = ASProxy<fmt, M>::fetch(as, idx, want);
    if (v < 0)
      continue;

    int start = g.get_in_start_pos(v);
    int end = start + g.get_in_degree(v);
    for (int i = start; i < end; ++i) {
      int u = strict_adj_list[i];
      bool toprocess = true;

      // if(!conf.conf_fromall && !conf.conf_fuse_inspect && as.query(u) !=
      // Active)
      //  data source must from Active vertex or the conf_fromall is enabled
      if (toprocess && !conf.conf_fromall)
        toprocess = as.bitmap.is_active(u);

      if (toprocess) {
        auto vdata = f.emit(u, strict_edgedata + i, g);
        f.comp(f.wa_of(v), vdata, g);
        if (conf.pruning())
          break; // idempotent optimization
      }
    }
  }
}

template <> struct ExpandProxy<VC, TM, Push> {
  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<CSR, E> g, F f,
                     config_t conf) {
    if (conf.conf_fuse_inspect) {
      Launch_Expand_VC(TM_fused, as, g, f, conf);
    } else
      Launch_Expand_VC(TM, as, g, f, conf);
  }

  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<COO, E> g, F f,
                     config_t conf) {}
};

template <> struct ExpandProxy<VC, TM, Pull> {
  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<CSR, E> g, F f,
                     config_t conf) {
    // if(as.fmt == Bitmap && conf.conf_target == Inactive) // don't know why
    // Bitmap is so slow, thus make it as a special case.
    //__rexpand_VC_TM_BITMAP<<<conf.ctanum, conf.thdnum>>>(as, g, f, conf);
    // else{
    Launch_RExpand_VC(TM, as, g, f, conf);
    // if(as.queue.mode==Normal) __rexpand_VC_TM<Normal><<<conf.ctanum,
    // conf.thdnum>>>(as, g, f, conf); else
    // __rexpand_VC_TM<Cached><<<conf.ctanum, conf.thdnum>>>(as, g, f, conf);
    //}
  }

  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<COO, E> g, F f,
                     config_t conf) {}
};

#endif
