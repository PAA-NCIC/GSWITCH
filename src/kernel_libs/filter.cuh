#ifndef __FILTER_H
#define __FILTER_H

#include "abstraction/config.cuh"
#include "data_structures/active_set.cuh"
#include "data_structures/functor.cuh"
#include "data_structures/graph.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

// popc for char
__device__ __forceinline__ int popc(char a) {
  int c;
  for (c = 0; a; ++c)
    a &= (a - 1);
  return c;
}

template <QueueMode M> __global__ void __copy_all(active_set_t as) {
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  int *base = Qproxy<M>::output_base(as.queue);
  for (int idx = gtid; idx < as.size; idx += STRIDE) {
    base[idx] = idx;
  }
  if (!gtid)
    *Qproxy<M>::output_size(as.queue) = as.size;
}

template <QueueMode M> __global__ void __filter_unfixed(active_set_t as) {
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  const int lane = threadIdx.x & 31;
  const int n_bytes = as.bitmap.active.bytes_size();
  const char *bits1 = (char *)(void *)as.bitmap.active.to_bytes();
  const char *bits2 = (char *)(void *)as.bitmap.inactive.to_bytes();
  int *base = Qproxy<M>::output_base(as.queue);
  for (int idx = gtid; idx < n_bytes; idx += STRIDE) {
    // if(idx>=as.size) break;
    char active = bits1[idx];
    char inactive = bits2[idx];
    char unfixed = active | (~inactive);
    int change = popc(unfixed);
    int rank, sum = 0;
    warpScan(change, rank, sum);
    int warp_base;
    if (!lane)
      warp_base = atomicAdd(Qproxy<M>::output_size(as.queue), sum);
    warp_base = _shfl(warp_base, 0);
    if (active) {
      for (int i = 0, c = 0; i < 8; ++i)
        if (unfixed & ((char)1 << i)) {
          base[warp_base + rank + (c++)] = (idx << 3) + i;
        }
    }
  }
}

// filter the active vertex in to a compact array atomic enqueue with
// aggregation.
template <QueueMode M> __global__ void __filter_active(active_set_t as) {
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  const int lane = threadIdx.x & 31;
  const int n_bytes = as.bitmap.active.bytes_size();
  const char *bits = (char *)(void *)as.bitmap.active.to_bytes();
  int *base = Qproxy<M>::output_base(as.queue);
  for (int idx = gtid; idx < n_bytes; idx += STRIDE) {
    // if(idx>=as.size) break;
    char active = bits[idx];
    int change = popc(active);
    int rank, sum = 0;
    warpScan(change, rank, sum);
    int warp_base;
    if (!lane)
      warp_base = atomicAdd(Qproxy<M>::output_size(as.queue), sum);
    warp_base = _shfl(warp_base, 0);
    if (active) {
      for (int i = 0, c = 0; i < 8; ++i)
        if (active & ((char)1 << i)) {
          base[warp_base + rank + (c++)] = (idx << 3) + i;
        }
    }
  }
}

// filter the inactive vertex in to a compact array atomic enqueue with
// aggregation.
template <QueueMode M> __global__ void __filter_inactive(active_set_t as) {
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  const int lane = threadIdx.x & 31;
  const int n_bytes = as.bitmap.inactive.bytes_size();
  const char *bits = (char *)(void *)as.bitmap.inactive.to_bytes();
  int *base = Qproxy<M>::output_base(as.queue);
  for (int idx = gtid; idx < n_bytes; idx += STRIDE) {
    char inactive = ~bits[idx];
    int change = popc(inactive);
    int rank, sum;
    warpScan(change, rank, sum);
    int warp_base;
    if (!lane)
      warp_base = atomicAdd(Qproxy<M>::output_size(as.queue), sum);
    warp_base = _shfl(warp_base, 0);
    if (inactive) {
      for (int i = 0, c = 0; i < 8; ++i)
        if (inactive & ((char)1 << i)) {
          base[warp_base + rank + (c++)] = (idx << 3) + i;
        }
    }
  }
}

///////////////////////////////////////////////////////////

// bitmap to local_bin, stride mode
__global__ void __filter_local_stride(active_set_t as, config_t conf) {
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  const int OFFSET = gtid * BIN_SZ;
  const int rest = as.size & (gridDim.x * blockDim.x - 1);
  const int vpt = (as.size / (gridDim.x * blockDim.x)) + (gtid < rest ? 1 : 0);
  const int start =
      gtid * (as.size / (gridDim.x * blockDim.x)) + (gtid < rest ? gtid : rest);
  const int end = start + vpt;
  Status want = conf.want();
  int qsize = 0;

  if (want == Active) {
    for (int idx = start; idx < end; ++idx) {
      if (as.bitmap.is_active(idx))
        as.bins.dg_bin[OFFSET + (qsize++)] = idx;
    }
  } else if (want == Inactive) {
    for (int idx = start; idx < end; ++idx) {
      if (as.bitmap.is_inactive(idx))
        as.bins.dg_bin[OFFSET + (qsize++)] = idx;
    }
  }

  as.bins.dg_size[gtid] = qsize;
}

// bitmap to local_bin, interleave mode
__global__ void __filter_local_interleave(active_set_t as, config_t conf) {

  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  const int OFFSET = gtid * BIN_SZ;
  Status want = conf.want();
  int qsize = 0;

  if (want == Active) {
    for (int idx = gtid; idx < as.size; idx += STRIDE) {
      if (as.bitmap.is_active(idx))
        as.bins.dg_bin[OFFSET + (qsize++)] = idx;
    }
  } else if (want == Inactive) {
    for (int idx = gtid; idx < as.size; idx += STRIDE) {
      if (as.bitmap.is_inactive(idx))
        as.bins.dg_bin[OFFSET + (qsize++)] = idx;
    }
  }

  as.bins.dg_size[gtid] = qsize;
}

template <QueueMode M>
__host__ void __filter_atomic(active_set_t &as, config_t conf) {
  const int ctanum = CTANUM >> 3; // TODO: based on metaphysics
  const int thdnum = THDNUM;
  if (conf.conf_target == Active) {
    __filter_active<M><<<ctanum, thdnum>>>(as);
  } else if (conf.conf_target == Inactive) {
    __filter_inactive<M><<<ctanum, thdnum>>>(as);
  } else {
    //__copy_all<M><<<ctanum,thdnum>>>(as);
    __filter_unfixed<M><<<ctanum, thdnum>>>(as);
  }
}

template <QueueMode M>
__host__ void __filter_scan(active_set_t &as, config_t conf) {
  if (conf.conf_cm == Interleave)
    __filter_local_stride<<<CTANUM, THDNUM>>>(as, conf);
  else
    __filter_local_interleave<<<CTANUM, THDNUM>>>(as, conf);
  const int cta_num = CTANUM >> 1;
  const int thd_num = THDNUM >> 1;
  scan<cta_num, thd_num, int>(as.bins.dg_size, as.bins.dg_offset,
                              CTANUM * THDNUM);
  __compress<<<CTANUM, THDNUM>>>(
      as.bins.dg_bin, as.bins.dg_size, as.bins.dg_offset,
      Qproxy<M>::output_base(as.queue), Qproxy<M>::output_size(as.queue));
}

// put all the filter variants togather the first round does not need filtering
template <typename G, typename F>
__host__ void __filter(active_set_t &as, G g, F f, config_t &conf) {
  TRACE();

#define Launch_Filter(acc, as, conf)                                           \
  if (conf.conf_qmode == Normal)                                               \
    __filter_##acc<Normal>(as, conf);                                          \
  else                                                                         \
    __filter_##acc<Cached>(as, conf);

  as.set_fmt(conf.conf_asfmt);

  // if use Queue format, copy the data into a compact array
  if (conf.conf_asfmt == Queue) {
    bool tofilter = true;
    if (conf.conf_fuse_inspect)
      tofilter = false;
    if (conf.conf_switch_to_fusion) {
      tofilter = true;
      conf.conf_switch_to_fusion = false;
    }
    if (conf.conf_first_round) {
      tofilter = true;
    }

    if (tofilter) {
      if (as.queue.mode == Cached && as.queue.traceback) {
        // do_noting
      } else {
        // if(fets.cap < 3.5)
        // Launch_Filter(scan, as, conf); // for oldder architecture (need
        // two-phase copy) else
        Launch_Filter(atomic, as,
                      conf); // for newer architecture  (only copy once)
      }
    } else {
      // mark duplicate in the expand phase
      // if not fused the inspection. there is no need to clean the visited
      // bitmap (only BFS will use it)
      if (!conf.conf_pruning)
        as.bitmap.visited.reset();
    }
    as.queue.swap();
  }

#undef Launch_Filter
  // for (int i = 0; i < as.queue.get_qsize_host(); i++)
  //   LOG("%d ", as.queue.subqueue[0].query(as.queue.offset_in + i));
  // LOG("[%d,%d,%d]\n", as.queue.cursor, as.queue.offset_in, as.queue.offset_out);
  // for (int i = 0; i < 10; i++) {
  //   printf("%d ", as.queue.subqueue[as.queue.cursor].query(i));
  // }
  // printf("\n");
  // LOG("$ %d ", as.queue.get_qsize_host());
  CUBARRIER();
}

#endif
