#ifndef __ACTIVE_SET_CUH
#define __ACTIVE_SET_CUH

#include "data_structures/bitmap.cuh"
#include "data_structures/notification.cuh"
#include "data_structures/queue.cuh"
#include "data_structures/workset.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_sortedsearch.hxx>

struct active_set_t {
  int build(int _size, int CTA_NUM, int THD_NUM, QueueMode qmode) {
    context = new mgpu::standard_context_t(false);
    size = _size;
    int gpu_bytes = queue.build(size, qmode);
    gpu_bytes += bins.build(size, CTA_NUM, THD_NUM);
    gpu_bytes += bitmap.build(size);
    gpu_bytes += nt.build();
    gpu_bytes += workset.build(size);
    return gpu_bytes;
  }

  void reset(bool traceback = false) {
    queue.reset(traceback);
    bins.reset();
    bitmap.reset();
    nt.reset();
    workset.reset();
  }

  void init(int root = -1) { bitmap.init(root); }

  bool is_converged() { return nt.is_converged(); }
  template <typename G, typename F> bool finish(G &g, F &f, config_t &conf) {
    bool converged = is_converged();
    if (conf.conf_window && conf.conf_fusion && converged) {
      window_delay_update(size, g, f);
    }
    return converged && nt.exit(g, f, conf);
  }

  size_t get_size_host() {
    if (fmt == Queue)
      size_cached = queue.get_qsize_host();
    else
      size_cached = size;
    return get_size_host_cached();
  }

  size_t get_size_host_cached() { return size_cached; }

  void set_fmt(ASFmt _fmt) { fmt = _fmt; }
  void halt_host() { nt.notify_host(); }

  __device__ __tbdinline__ size_t get_size() {
    return (fmt == Queue ? get_queue_size_hard<Normal>() : get_bitmap_size());
  }

  __device__ __tbdinline__ int fetch(int idx, Status wanted) {
    return (fmt == Queue ? fetch_queue<Normal>(idx, wanted)
                         : fetch_bitmap(idx, wanted));
  }

  __device__ __tbdinline__ void halt_device() { nt.notify_device(); }

  __device__ __tbdinline__ int fetch_bitmap(int idx, Status wanted) {
    if (wanted == All && (bitmap.is_active(idx) || bitmap.is_inactive(idx)))
      return idx;
    else if (wanted == Active && bitmap.is_active(idx))
      return idx;
    else if (wanted == Inactive && bitmap.is_inactive(idx))
      return idx;
    return -1;
  }

  __device__ __tbdinline__ size_t get_bitmap_size() { return size; }

  template <QueueMode Mode>
  __device__ __tbdinline__ int fetch_queue(int idx, Status wanted) {
    return Qproxy<Mode>::fetch(queue, idx);
  }

  __device__ __tbdinline__ size_t get_queue_size() { return size_cached; }

  template <QueueMode Mode>
  __device__ __tbdinline__ size_t get_queue_size_hard() {
    return Qproxy<Mode>::get_qsize(queue);
  }

  mgpu::standard_context_t *context;
  ASFmt fmt = Queue;
  size_t size;
  size_t size_cached;
  bin_t bins;
  queue_t queue;
  bitmap_t bitmap;
  workset_t workset;
  notification_t nt;
};

template <ASFmt fmt, QueueMode M> struct ASProxy {};

template <QueueMode M> struct ASProxy<Queue, M> {
  static __device__ __tbdinline__ int fetch(active_set_t &as, const int &idx,
                                            const Status &wanted) {
    return as.fetch_queue<M>(idx, wanted);
  }
  static __device__ __tbdinline__ int get_size(active_set_t &as) {
    return as.get_queue_size();
  }
  static __device__ __tbdinline__ int get_size_hard(active_set_t &as) {
    return as.get_queue_size_hard<M>();
  }
};

template <QueueMode M> struct ASProxy<Bitmap, M> {
  static __device__ __tbdinline__ int fetch(active_set_t &as, const int &idx,
                                            const Status &wanted) {
    return as.fetch_bitmap(idx, wanted);
  }
  static __device__ __tbdinline__ int get_size(active_set_t &as) {
    return as.get_bitmap_size();
  }
  static __device__ __tbdinline__ int get_size_hard(active_set_t &as) {
    return as.get_bitmap_size();
  }
};

#define Launch_Expand_VC(lb, as, g, f, conf)                                   \
  if (as.fmt == Queue) {                                                       \
    if (as.queue.mode == Normal)                                               \
      __expand_VC_##lb<Queue, Normal>                                          \
          <<<conf.ctanum, conf.thdnum>>>(as, g, f, conf);                      \
    else                                                                       \
      __expand_VC_##lb<Queue, Cached>                                          \
          <<<conf.ctanum, conf.thdnum>>>(as, g, f, conf);                      \
  } else {                                                                     \
    __expand_VC_##lb<Bitmap, Normal>                                           \
        <<<conf.ctanum, conf.thdnum>>>(as, g, f, conf);                        \
  }

#define Launch_RExpand_VC(lb, as, g, f, conf)                                  \
  if (as.fmt == Queue) {                                                       \
    if (as.queue.mode == Normal)                                               \
      __rexpand_VC_##lb<Queue, Normal>                                         \
          <<<conf.ctanum, conf.thdnum>>>(as, g, f, conf);                      \
    else                                                                       \
      __rexpand_VC_##lb<Queue, Cached>                                         \
          <<<conf.ctanum, conf.thdnum>>>(as, g, f, conf);                      \
  } else {                                                                     \
    __rexpand_VC_##lb<Bitmap, Normal>                                          \
        <<<conf.ctanum, conf.thdnum>>>(as, g, f, conf);                        \
  }

active_set_t build_active_set(int size, config_t conf) {
  active_set_t as;
  LOG(" -- Allocating memory for Active set storage...\n");
  int64_t gpu_bytes = as.build(size, CTANUM, THDNUM, conf.conf_qmode);
  LOG(" -- GPU Global Memory used for active set: %f GB.\n",
      (0.0 + gpu_bytes) / (1ll << 30));
  return as;
}

template <typename E, typename F>
__global__ void __window_delay_update(int size, device_graph_t<CSR, E> g, F f) {
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int idx = gtid; idx < size; idx += STRIDE) {
    f.filter(idx, g); // for update
  }
}

template <typename E, typename F>
__host__ void window_delay_update(int size, device_graph_t<CSR, E> g, F f) {
  __window_delay_update<<<CTANUM, THDNUM>>>(size, g, f);
}

template <typename E, typename F>
__host__ void window_delay_update(int size, device_graph_t<COO, E> g, F f) {}

#endif
