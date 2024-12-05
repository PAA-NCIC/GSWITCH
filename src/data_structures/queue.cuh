#ifndef __QUEUE_CUH
#define __QUEUE_CUH

#include "data_structures/graph.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

// used as temporary storage for local output of each thread.
// max capacity is 512 elements.
struct bin_t {
  int64_t build(int nvertexs, int CTA_NUM, int THD_NUM) {
    H_ERR(
        cudaMalloc((void **)&dg_bin, sizeof(int) * BIN_SZ * CTA_NUM * THD_NUM));
    H_ERR(cudaMalloc((void **)&dg_size, sizeof(int) * CTA_NUM * THD_NUM));
    H_ERR(cudaMalloc((void **)&dg_offset, sizeof(int) * CTA_NUM * THD_NUM));
    int gpu_bytes = sizeof(int) * CTA_NUM * THD_NUM * (BIN_SZ + 2);
    return gpu_bytes;
  }

  void reset() { /*do nothing*/ }

  int *dg_bin;    // BIN_SZ*CTA_NUM*THD_NUM
  int *dg_size;   // CTA_NUM*THD_NUM
  int *dg_offset; // CTA_NUM*THD_NUM
};

// used to store the active elements in a compact array.
struct subqueue_t {
  int build(size_t size) {
    int gpu_bytes = 0;
    max_size = size;
    H_ERR(cudaMalloc((void **)&dg_queue, sizeof(int) * size));
    H_ERR(cudaMalloc((void **)&dg_qsize, sizeof(int)));
    CLEAN(dg_queue, size);
    CLEAN(dg_qsize, 1);
    gpu_bytes += sizeof(int) * size + sizeof(int);
    return gpu_bytes;
  }

  void reset() {
    CLEAN(dg_queue, max_size);
    CLEAN(dg_qsize, 1);
  }

  void init(int root) {
    if (root >= 0) {
      excudaMemset(dg_qsize, 1, 1);
      excudaMemset(dg_queue, root, 1);
    }
  }

  int debug() { return get_qsize_host(); }
  int get_qsize_host() {
    int ret;
    TOHOST(dg_qsize, &ret, 1);
    return ret;
  }
  int query(int idx) {
    int ret;
    TOHOST(dg_queue + idx, &ret, 1);
    return ret;
  }
  void clean() { CLEAN(dg_qsize, 1); }

  __device__ void push(int ele) {
    int idx = atomicAdd(dg_qsize, 1);
    dg_queue[idx] = ele;
  }
  __device__ void set_qsize(size_t qs) { *dg_qsize = qs; }
  __device__ int get_qsize() { return *dg_qsize; }
  __device__ int fetch(int idx) { return __ldg(dg_queue + idx); }

  int *dg_queue; // nvertexs or nedges
  int *dg_qsize; // 1
  int max_size;
};

/* In Normal mode:
    - subqueue[cursor]: the input queue;
    - subqueue[cursor^1]: the output queue;
   In Cached mode:
    - subqueue[cursor]: the queue;
    - subqueue[cursor^1]: the quesize;
 */
// ping-pong queues
struct queue_t {
  int64_t build(int size, QueueMode qmode) {
    mode = qmode;
    offset_in = offset_out = 0;
    cursor = 0;
    traceback = false;
    return subqueue[0].build(size) + subqueue[1].build(size);
  }

  void reset(bool _traceback = false) {
    traceback = _traceback;
    if (mode == Normal) {
      subqueue[0].reset();
      subqueue[1].reset();
    }
    if (!traceback) {
      cursor = 0;
      offset_in = offset_out = 0;
    }
  }

  void abandon() {
    if (mode == Normal) {
      subqueue[cursor].clean();
    }
  }
  // cursor pointer to current queue, cursor+1 pointer next queue.
  void swap() {
    if (mode == Normal) {
      subqueue[cursor].clean();
      cursor ^= 1;
    } else {
      if (traceback) {
        cursor--;
        offset_in -= get_qsize_host();
      } else {
        cursor++;
        offset_in = offset_out;
        offset_out += get_qsize_host();
      }
    }
  }

  int get_qsize_host() {
    if (mode == Normal) {
      return subqueue[cursor].get_qsize_host();
    } else {
      return subqueue[1].query(cursor);
    }
  }

  // ############# Normal ###############
  __device__ __tbdinline__ int get_qsize_Normal() {
    return subqueue[cursor].dg_qsize[0];
  }

  __device__ __tbdinline__ int fetch_Normal(int idx) {
    return subqueue[cursor].fetch(idx);
  }

  __device__ __tbdinline__ void push_Normal(int element) {
    return subqueue[cursor ^ 1].push(element);
  }

  __host__ __device__ __tbdinline__ int *output_base_Normal() {
    return subqueue[cursor ^ 1].dg_queue;
  }

  __host__ __device__ __tbdinline__ int *output_size_Normal() {
    return subqueue[cursor ^ 1].dg_qsize;
  }

  // ############# Cached ###############
  __device__ __tbdinline__ int get_qsize_Cached() {
    return subqueue[1].dg_queue[cursor];
  }

  __device__ __tbdinline__ int fetch_Cached(int idx) {
    return subqueue[0].dg_queue[offset_in + idx];
  }

  __device__ __tbdinline__ void
  push_Cached(int element) { // disabled in traceback
    if (traceback)
      return;
    int idx = atomicAdd(output_size_Cached(), 1);
    subqueue[0].dg_queue[offset_out + idx] = element;
  }

  __host__ __device__ __tbdinline__ int *output_base_Cached() {
    return subqueue[0].dg_queue + offset_out;
  } // disabled in traceback

  __host__ __device__ __tbdinline__ int *output_size_Cached() {
    return subqueue[1].dg_queue + cursor + 1;
  } // disabled in traceback, must be zero at first.

  subqueue_t subqueue[2];
  bool traceback;
  int cursor;
  int offset_in, offset_out;
  QueueMode mode;
};

template <QueueMode M> struct Qproxy {};

template <> struct Qproxy<Normal> {
  static __device__ __tbdinline__ int get_qsize(queue_t &q) {
    return q.get_qsize_Normal();
  }
  static __device__ __tbdinline__ int fetch(queue_t &q, int idx) {
    return q.fetch_Normal(idx);
  }
  static __device__ __tbdinline__ void push(queue_t &q, int element) {
    q.push_Normal(element);
  }
  static __device__ __tbdinline__ int *output_base(queue_t &q) {
    return q.output_base_Normal();
  }
  static __device__ __tbdinline__ int *output_size(queue_t &q) {
    return q.output_size_Normal();
  }
};

template <> struct Qproxy<Cached> {
  static __device__ __tbdinline__ int get_qsize(queue_t &q) {
    return q.get_qsize_Cached();
  }
  static __device__ __tbdinline__ int fetch(queue_t &q, int idx) {
    return q.fetch_Cached(idx);
  }
  static __device__ __tbdinline__ void push(queue_t &q, int element) {
    return q.push_Cached(element);
  }
  static __device__ __tbdinline__ int *output_base(queue_t &q) {
    return q.output_base_Cached();
  }
  static __device__ __tbdinline__ int *output_size(queue_t &q) {
    return q.output_size_Cached();
  }
};

#endif
