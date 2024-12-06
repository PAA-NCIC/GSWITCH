#ifndef __BLOCK_SCAN_CUH
#define __BLOCK_SCAN_CUH

#include <cuda.h>

// this struct is modifiled from gunrock
template <typename T, int _LOG_THREADS> // 256 1<<8
struct Block_Scan {
  enum {
    LOG_THREADS = _LOG_THREADS,
    THREADS = 1 << LOG_THREADS,
    LOG_WARP_THREADS = 5, // GR_LOG_WARP_THREADS(CUDA_ARCH),
    WARP_THREADS = 1 << LOG_WARP_THREADS,
    WARP_THREADS_MASK = WARP_THREADS - 1,
    LOG_BLOCK_WARPS = _LOG_THREADS - LOG_WARP_THREADS,
    BLOCK_WARPS = 1 << LOG_BLOCK_WARPS, // = 8
  };

  struct Temp_Space {
    T warp_counter_offset[BLOCK_WARPS];
    T block_sum;
    int block_oft;
  };

  static __device__ __tbdinline__ void Warp_Scan(T thread_in, T &thread_out,
                                                 T &sum) {
    int lane_id = threadIdx.x & WARP_THREADS_MASK;
    T &lane_local = thread_out;
    T lane_recv;

    lane_local = thread_in;
    lane_recv = _shfl_xor(lane_local, 1);
    if ((lane_id & 1) == 0) {
      lane_local += lane_recv;
      lane_recv = _shfl_xor(lane_local, 2);
    }

    if ((lane_id & 3) == 0) {
      lane_local += lane_recv;
      lane_recv = _shfl_xor(lane_local, 4);
    }

    if ((lane_id & 7) == 0) {
      lane_local += lane_recv;
      lane_recv = _shfl_xor(lane_local, 8);
    }

    if ((lane_id & 0xF) == 0) {
      lane_local += lane_recv;
      lane_recv = _shfl_xor(lane_local, 0x10);
    }

    if (lane_id == 0) {
      lane_local += lane_recv;
    }
    sum = _shfl(lane_local, 0);
    if (lane_id == 0) {
      lane_recv = 0;
    }
    lane_local = lane_recv;
    lane_recv = _shfl_up(lane_local, 8);
    if ((lane_id & 15) == 8)
      lane_local += lane_recv;
    lane_recv = _shfl_up(lane_local, 4);
    if ((lane_id & 7) == 4)
      lane_local += lane_recv;

    lane_recv = _shfl_up(lane_local, 2);
    if ((lane_id & 3) == 2)
      lane_local += lane_recv;

    lane_recv = _shfl_up(lane_local, 1);
    if ((lane_id & 1) == 1)
      lane_local += lane_recv;
  }

  static __device__ __tbdinline__ void Warp_Scan(T thread_in, T &thread_out) {
    int lane_id = threadIdx.x & WARP_THREADS_MASK;
    T &lane_local = thread_out;
    T lane_recv;
    lane_local = thread_in;
    lane_recv = _shfl_xor(lane_local, 1);
    if ((lane_id & 1) == 0) {
      lane_local += lane_recv;
      lane_recv = _shfl_xor(lane_local, 2);
    }
    if ((lane_id & 3) == 0) {
      lane_local += lane_recv;
      lane_recv = _shfl_xor(lane_local, 4);
    }
    if ((lane_id & 7) == 0) {
      lane_local += lane_recv;
      lane_recv = _shfl_xor(lane_local, 8);
    }
    if ((lane_id & 0xF) == 0) {
      lane_local += lane_recv;
      lane_recv = _shfl_xor(lane_local, 0x10);
    }
    if (lane_id == 0) {
      lane_local += lane_recv;
      lane_recv = 0;
    }
    lane_local = lane_recv;
    lane_recv = _shfl_up(lane_local, 8);
    if ((lane_id & 15) == 8)
      lane_local += lane_recv;

    lane_recv = _shfl_up(lane_local, 4);
    if ((lane_id & 7) == 4)
      lane_local += lane_recv;

    lane_recv = _shfl_up(lane_local, 2);
    if ((lane_id & 3) == 2)
      lane_local += lane_recv;

    lane_recv = _shfl_up(lane_local, 1);
    if ((lane_id & 1) == 1)
      lane_local += lane_recv;
  }

  static __device__ __tbdinline__ void Warp_LogicScan(int thread_in,
                                                      T &thread_out) {
    unsigned int warp_flag = _ballot(thread_in);
    int lane_id = threadIdx.x & WARP_THREADS_MASK;
    unsigned int lane_mask = (1 << lane_id) - 1;
    thread_out = __popc(warp_flag & lane_mask);
  }

  static __device__ __tbdinline__ void Warp_LogicScan(int thread_in,
                                                      T &thread_out, T &sum) {
    unsigned int warp_flag = _ballot(thread_in);
    int lane_id = threadIdx.x & WARP_THREADS_MASK;
    unsigned int lane_mask = (1 << lane_id) - 1;
    thread_out = __popc(warp_flag & lane_mask);
    sum = __popc(warp_flag);
  }

  static __device__ __tbdinline__ void Scan(T thread_in, T &thread_out,
                                            Temp_Space &temp_space) {
    T warp_sum;
    int warp_id = threadIdx.x >> LOG_WARP_THREADS;
    Warp_Scan(thread_in, thread_out, warp_sum);
    if ((threadIdx.x & WARP_THREADS_MASK) == 0) {
      temp_space.warp_counter_offset[warp_id] = warp_sum;
    }
    __syncthreads();

    if ((warp_id) == 0) {
      warp_sum = threadIdx.x < BLOCK_WARPS
                     ? temp_space.warp_counter_offset[threadIdx.x]
                     : 0;
      Warp_Scan(warp_sum, warp_sum);
      if (threadIdx.x < BLOCK_WARPS)
        temp_space.warp_counter_offset[threadIdx.x] = warp_sum;
    }
    __syncthreads();

    thread_out += temp_space.warp_counter_offset[warp_id];
  }

  static __device__ __tbdinline__ void
  Scan(T thread_in, T &thread_out, Temp_Space &temp_space, T &block_sum) {
    T warp_sum;
    T warp_sum_tmp;
    int warp_id = threadIdx.x >> LOG_WARP_THREADS;
    Warp_Scan(thread_in, thread_out, warp_sum);
    if ((threadIdx.x & WARP_THREADS_MASK) == 0) {
      temp_space.warp_counter_offset[warp_id] = warp_sum;
    }
    __syncthreads();
    if ((warp_id) == 0) {
      warp_sum_tmp = threadIdx.x < BLOCK_WARPS
                         ? temp_space.warp_counter_offset[threadIdx.x]
                         : 0;
      Warp_Scan(warp_sum_tmp, warp_sum, block_sum);
      if (threadIdx.x < BLOCK_WARPS)
        temp_space.warp_counter_offset[threadIdx.x] = warp_sum;
      if (threadIdx.x == 0)
        temp_space.block_sum = block_sum;
    }
    __syncthreads();
    thread_out += temp_space.warp_counter_offset[warp_id];
    block_sum = temp_space.block_sum;
  }

  static __device__ __tbdinline__ void LogicScan(int thread_in, T &thread_out,
                                                 Temp_Space &temp_space) {
    T warp_sum;
    int warp_id = threadIdx.x >> LOG_WARP_THREADS;
    Warp_LogicScan(thread_in, thread_out, warp_sum);
    if ((threadIdx.x & WARP_THREADS_MASK) == 0)
      temp_space.warp_counter_offset[warp_id] = warp_sum;
    __syncthreads();
    if ((warp_id) == 0) {
      warp_sum = threadIdx.x < BLOCK_WARPS
                     ? temp_space.warp_counter_offset[threadIdx.x]
                     : 0;
      Warp_Scan(warp_sum, warp_sum);
      if (threadIdx.x < BLOCK_WARPS)
        temp_space.warp_counter_offset[threadIdx.x] = warp_sum;
    }
    __syncthreads();
    thread_out += temp_space.warp_counter_offset[warp_id];
  }

  static __device__ __tbdinline__ void LogicScan(int thread_in, T &thread_out,
                                                 Temp_Space &temp_space,
                                                 T &block_sum) {
    T warp_sum;
    int warp_id = threadIdx.x >> LOG_WARP_THREADS;
    Warp_LogicScan(thread_in, thread_out, warp_sum);
    if ((threadIdx.x & WARP_THREADS_MASK) == 0)
      temp_space.warp_counter_offset[warp_id] = warp_sum;
    __syncthreads();

    if ((warp_id) == 0) {
      warp_sum = threadIdx.x < BLOCK_WARPS
                     ? temp_space.warp_counter_offset[threadIdx.x]
                     : 0;
      Warp_Scan(warp_sum, warp_sum, block_sum);
      if (threadIdx.x < BLOCK_WARPS)
        temp_space.warp_counter_offset[threadIdx.x] = warp_sum;
      if (threadIdx.x == 0)
        temp_space.block_sum = block_sum;
    }
    __syncthreads();
    thread_out += temp_space.warp_counter_offset[warp_id];
    block_sum = temp_space.block_sum;
  }
};

__device__ int block_scan(int *tmp, int phase) {
  int total_block = 0;
  int offset = 1;
  for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (phase < d) {
      int ai = offset * (2 * phase + 1) - 1;
      int bi = offset * (2 * phase + 2) - 1;
      tmp[bi] += tmp[ai];
    }
    offset <<= 1;
  }

  __syncthreads();
  total_block = tmp[blockDim.x - 1];
  __syncthreads();
  if (!phase)
    tmp[blockDim.x - 1] = 0;
  __syncthreads();

  for (int d = 1; d < blockDim.x; d <<= 1) {
    __syncthreads();
    offset >>= 1;
    if (phase < d) {
      int ai = offset * (2 * phase + 1) - 1;
      int bi = offset * (2 * phase + 2) - 1;
      int t = tmp[ai];
      tmp[ai] = tmp[bi];
      tmp[bi] += t;
    }
  }
  __syncthreads();
  return total_block;
}

__device__ int warp_scan(int *tmp, int phase) {
  int total_warp = 0;
  int offset = 1;
  for (int d = 32 >> 1; d > 0; d >>= 1) {
    if (phase < d) {
      int ai = offset * (2 * phase + 1) - 1;
      int bi = offset * (2 * phase + 2) - 1;
      tmp[bi] += tmp[ai];
    }
    offset <<= 1;
  }

  total_warp = tmp[32 - 1];
  if (!phase)
    tmp[32 - 1] = 0;

  for (int d = 1; d < 32; d <<= 1) {
    offset >>= 1;
    if (phase < d) {
      int ai = offset * (2 * phase + 1) - 1;
      int bi = offset * (2 * phase + 2) - 1;

      int t = tmp[ai];
      tmp[ai] = tmp[bi];
      tmp[bi] += t;
    }
  }
  return total_warp;
}

#endif
