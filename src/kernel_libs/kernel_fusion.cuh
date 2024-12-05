#ifndef __KERNEL_FUSION_CUH
#define __KERNEL_FUSION_CUH

#include "data_structures/active_set.cuh"
#include "data_structures/functor.cuh"
#include "data_structures/graph.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

const int PER_OUT = 8;
const int LOG_PER_OUT = 3;

template <QueueMode M> __device__ void __direct_write(int *x, queue_t queue) {
  if (queue.mode == Cached && queue.traceback)
    return;
  int loc = atomicAdd(Qproxy<M>::output_size(queue), PER_OUT);
  int *base = Qproxy<M>::output_base(queue);
  for (int i = 0; i < PER_OUT; ++i) {
    base[loc + i] = x[i];
  }
}

// this function is modified from gunrock
template <QueueMode M>
__device__ void
__write_global_queue_warp(Block_Scan<int, 10>::Temp_Space &sh_scan_space,
                          int *sh_output_cache, int &thread_output,
                          queue_t queue) {
  if (queue.mode == Cached && queue.traceback) {
    thread_output = 0;
    return;
  }
  const int OFFSET = threadIdx.x << LOG_PER_OUT;
  int output_loc = 0;
  Block_Scan<int, 10>::Warp_Scan(thread_output, output_loc);
  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int *base = Qproxy<M>::output_base(queue);
  if (lane == 31) {
    if (output_loc + thread_output != 0)
      sh_scan_space.warp_counter_offset[warp_id] =
          atomicAdd(Qproxy<M>::output_size(queue), output_loc + thread_output);
  }

  if (thread_output != 0) {
    output_loc += sh_scan_space.warp_counter_offset[warp_id];
    for (int i = 0; i < thread_output; ++i) {
      base[output_loc + i] = sh_output_cache[OFFSET + i];
    }
  }
  thread_output = 0;
}

template <QueueMode M, typename G, typename F>
__global__ void __compensation_for_queue(queue_t queue, G g, F f,
                                         config_t conf) {
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  int size = Qproxy<M>::get_qsize(queue); // must be queue !
  for (int idx = gtid; idx < size; idx += STRIDE) {
    int victim = Qproxy<M>::fetch(queue, idx);
    // printf("%d\n", victim);
    f.filter(victim, g);
  }
}

// compensation for kernel fusion data update !!!.
// Q: why not fuse this into kernels?
// A: the updating phase must be finished before any vertex can touch others'
// data.
template <QueueMode M, typename G, typename F>
__host__ void __compensation(active_set_t &as, G g, F f, config_t conf) {
  if (!conf.conf_compensation)
    return;
  if (as.fmt == Queue) {
    __compensation_for_queue<M>
        <<<conf.ctanum, conf.thdnum>>>(as.queue, g, f, conf);
  } else {
    // TODO: fusion will not enabled as bitmap in current version.
    //  do nothing;
  }
}

template <Centric C> struct CompensationProxy {};

template <> struct CompensationProxy<VC> {
  template <typename E, typename F>
  static void compensation(active_set_t &as, device_graph_t<CSR, E> g, F f,
                           config_t conf) {
    if (conf.conf_fuse_inspect && !conf.conf_first_round) {
      if (conf.conf_qmode == Normal)
        __compensation<Normal>(as, g, f, conf);
      else
        __compensation<Cached>(as, g, f, conf);
    }
  }
  template <typename E, typename F>
  static void compensation(active_set_t &as, device_graph_t<COO, E> g, F f,
                           config_t conf) {}
};

template <> struct CompensationProxy<EC> {
  template <typename E, typename F>
  static void compensation(active_set_t &as, device_graph_t<COO, E> g, F f,
                           config_t conf) {}
  template <typename E, typename F>
  static void compensation(active_set_t &as, device_graph_t<CSR, E> g, F f,
                           config_t conf) {}
};

template <ASFmt fmt, QueueMode M, typename G, typename F>
__global__ void __super_fusion(active_set_t as, G g, F f, config_t conf) {
  const int *__restrict__ strict_adj_list = g.dg_adj_list;

  __shared__ int aset[2048];
  __shared__ int asize[2];
  int tid = threadIdx.x;
  int cosize = 32;
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  int STRIDE = 32;
  int OFFSET = 1024;
  int c = 0;
  if (tid == 0) {
    asize[0] = asize[1] = 0;
    aset[c * OFFSET + asize[c]] = ASProxy<fmt, M>::fetch(as, 0, Active);
    asize[c]++;
  }
  __syncthreads();

  for (;;) {
    // if(threadIdx.x==0) printf("%d\n", asize[c]);
    for (int idx = wid; idx < asize[c]; idx += STRIDE) {
      int v = aset[idx + c * OFFSET];
      int start = tex1Dfetch<int>(g.dt_start_pos, v);
      int end = start + tex1Dfetch<int>(g.dt_odegree, v);
      for (int i = start + lane; i < end; i += cosize) {
        int u = strict_adj_list[i];
        auto vdata = f.emit(v, g.fetch_edata(i), g);
        bool toprocess = true;

        // check 1: if idempotent, we can prune the redundant update (if has,
        // that's also OK)
        // if(toprocess && conf.pruning())
        // toprocess = as.bitmap.mark_duplicate_lite(u);

        if (toprocess && !conf.conf_toall) {
          toprocess = f.cond(u, vdata, g);
        }

        if (toprocess) {
          toprocess = f.compAtomic(f.wa_of(u), vdata, g);
        }

        // check 3:  enqueue the u only once. (if duplicate, wrong answer)
        if (toprocess)
          toprocess = as.bitmap.mark_duplicate_atomic(u);

        if (toprocess) {
          int loc = atomicAdd(asize + (c ^ 1), 1);
          aset[loc + (c ^ 1) * OFFSET] = u;
        }
      }
    }
    __syncthreads();
    asize[c] = 0;
    c ^= 1;
    if (asize[c] == 0)
      break;
    g.update_level();
    __syncthreads();
  }
}

template <typename E, typename F>
void super_fusion(active_set_t as, device_graph_t<CSR, E> g, F f,
                  config_t conf) {
  __super_fusion<Queue, Normal><<<1, 1024>>>(as, g, f, conf);
}

template <typename E, typename F>
void super_fusion(active_set_t as, device_graph_t<COO, E> g, F f,
                  config_t conf) {}

#endif
