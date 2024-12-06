#ifndef __INTRINSICS_CUH
#define __INTRINSICS_CUH

#include <cuda.h>

template <typename T>
__device__ static __forceinline__ T _shfl_up(T var, unsigned int delta) {
#if (__CUDACC_VER_MAJOR__ >= 9)
  return __shfl_up_sync(__activemask(), var, delta);
#else
  return __shfl_up(var, delta);
#endif
}

template <typename T>
__device__ static __forceinline__ T _shfl_down(T var, unsigned int delta) {
#if (__CUDACC_VER_MAJOR__ >= 9)
  return __shfl_down_sync(__activemask(), var, delta);
#else
  return __shfl_down(var, delta);
#endif
}

template <typename T>
__device__ static __forceinline__ T _shfl(T var, int src_line) {
#if (__CUDACC_VER_MAJOR__ >= 9)
  return __shfl_sync(__activemask(), var, src_line);
#else
  return __shfl(__activemask(), src_line);
#endif
}

template <typename T>
__device__ static __forceinline__ T _shfl_xor(T var, int lane_mask) {
#if (__CUDACC_VER_MAJOR__ >= 9)
  return __shfl_xor_sync(__activemask(), var, lane_mask);
#else
  return __shfl_xor(var, lane_mask);
#endif
}

__device__ static __forceinline__ unsigned _ballot(int predicate) {
#if (__CUDACC_VER_MAJOR__ >= 9)
  return __ballot_sync(__activemask(), predicate);
#else
  return __ballot(predicate);
#endif
}

__device__ static __forceinline__ int _any(int predicate) {
#if (__CUDACC_VER_MAJOR__ >= 9)
  return __any_sync(__activemask(), predicate);
#else
  return __any(predicate);
#endif
}

#include "utils/block_scan.cuh"
#include "utils/scan.cuh"

__device__ __tbdinline__ void throw_fatal_error() { *(int *)0 = 0; }

__device__ __tbdinline__ int __lower_bound(int *array, int len, int key) {
  int s = 0;
  while (len > 0) {
    int half = len >> 1;
    int mid = s + half;
    if (array[mid] < key) {
      s = mid + 1;
      len = len - half - 1;
    } else {
      len = half;
    }
  }
  return s;
}

__device__ __tbdinline__ int __upper_bound(int *array, int len, int key) {
  int s = 0;
  while (len > 0) {
    int half = len >> 1;
    int mid = s + half;
    if (array[mid] > key) {
      len = half;
    } else {
      s = mid + 1;
      len = len - half - 1;
    }
  }
  return s;
}

//__device__ __tbdinline__ int
//__upper_bound(int* array, int len, int key){
//  for(int i = 0; i < len; ++i){
//    if(key < array[i]) return i;
//  }
//  return len;
//}

__device__ int alignment(int arbitrary, int base) {
  return ((arbitrary + base - 1) / base) * base;
}

__device__ int lanemask_lt() { return (1 << (threadIdx.x & 31)) - 1; }

__device__ int atomicAggInc(int *ctr) {
  unsigned int active = _ballot(1);
  int leader = __ffs(active) - 1;
  int change = __popc(active);
  unsigned int rank = __popc(active & lanemask_lt());
  int warp_res;
  if (rank == 0)
    warp_res = atomicAdd(ctr, change);
  // warp_res = __shfl_sync(active, warp_res, leader);
  warp_res = _shfl(warp_res, leader);
  return warp_res + rank;
}

template <typename T>
__device__ __tbdinline__ void warpScan(T thread_in, T &thread_out, T &sum) {
  int lane_id = threadIdx.x & 31;
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

__device__ __tbdinline__ int warpReduceMin(int val) {
  for (int offset = 32 >> 1; offset > 0; offset >>= 1) {
    int tmp = _shfl_down(val, offset);
    val = MIN(tmp, val);
  }
  return val;
}

__device__ __tbdinline__ int blockReduceMin(int val) {
  static __shared__ int shared[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;

  val = warpReduceMin(val);
  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : MAX_32S;
  if (wid == 0)
    val = warpReduceMin(val);
  return val;
}

__device__ __tbdinline__ int warpReduceMax(int val) {
  for (int offset = 32 >> 1; offset > 0; offset >>= 1) {
    int tmp = _shfl_down(val, offset);
    val = MAX(tmp, val);
  }
  return val;
}

__device__ __tbdinline__ int blockReduceMax(int val) {
  static __shared__ int shared[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax(val);
  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
  if (wid == 0)
    val = warpReduceMax(val);
  return val;
}

__device__ __tbdinline__ int warpReduceSum(int val) {
  for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    val += _shfl_down(val, offset);
  return val;
}

__device__ __tbdinline__ int blockReduceSum(int val) {
  static __shared__ int shared[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum(val);
  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
  if (wid == 0)
    val = warpReduceSum(val);
  return val;
}

// N = blocksize = 256
__global__ void reduce(int *dg_in, int *dg_ans, int N) {
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int STRIDE = blockDim.x * gridDim.x;
  int sum = 0;
  for (int idx = gtid; idx < N; idx += STRIDE)
    sum += dg_in[idx];
  sum = blockReduceSum(sum);
  if (threadIdx.x == 0)
    *dg_ans = sum;
}

// memset index
__global__ void __memsetIdx(int *array, int n, int spval, int sploc,
                            int total) {
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockDim.x * blockIdx.x;
  const int OFFSET = spval * sploc;
  for (int idx = gtid; idx < n; idx += STRIDE) {
    if (idx < sploc)
      array[idx] = spval * idx;
    else
      array[idx] = OFFSET + (spval - 1) * (idx - sploc);
  }
  if (gtid == 0)
    array[n] = total;
}

// local_bin to queue, compress
// BIN_SZ = 512
__global__ void __compress(int *dg_src, int *dg_size, int *dg_offset,
                           int *dg_dst, int *dg_qsize) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int n = dg_size[tid];
  const int dst_pos = dg_offset[tid];
  const int src_pos = BIN_SZ * tid;
  for (int i = 0; i < n; i++) {
    dg_dst[dst_pos + i] = dg_src[src_pos + i];
  }

  if (tid == blockDim.x * gridDim.x - 1)
    *dg_qsize = dst_pos + n;
}

template <typename data_t>
__global__ void __excudaMemset(data_t *dg_in, data_t dft, size_t N) {
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int STRIDE = blockDim.x * gridDim.x;
  for (int idx = gtid; idx < N; idx += STRIDE)
    dg_in[idx] = dft;
}

template <typename data_t>
__host__ void excudaMemset(data_t *dg_in, data_t dft, size_t N) {
  __excudaMemset<<<CTANUM, THDNUM>>>(dg_in, dft, N);
  // cudaThreadSynchronize();
}

// data_t is packed with 4bytes
template <typename data_t>
__device__ __tbdinline__ data_t __exshfl_down(data_t data, int delta) {
  int N = sizeof(data_t) / sizeof(int);
  int *x = (int *)&data;
  for (int i = 0; i < N; ++i) {
    x[i] = _shfl_down(x[i], delta);
  }
  return *((data_t *)x);
}

// data_t is packed with 4bytes
template <typename data_t>
__device__ __tbdinline__ bool __equals(data_t v, data_t u) {
  int N = sizeof(data_t) / sizeof(int);
  int *x = (int *)&v;
  int *y = (int *)&u;
  bool flag = true;
  for (int i = 0; i < N; ++i) {
    flag &= (x[i] == y[i]);
  }
  return flag;
}

__device__ float atomicMin(float *address, float val) {
  int *address_as_i = (int *)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
                      __float_as_int(::fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__global__ void pointer_jumping(int *dg_data, int nvertexs, int *dg_flag) {
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int s_tmp;

  if (!threadIdx.x)
    s_tmp = 0;
  __syncthreads();

  for (int idx = gtid; idx < nvertexs; idx += STRIDE) {
    int y = dg_data[idx];
    int x = dg_data[y];
    if (x != y) {
      dg_data[idx] = x;
      s_tmp = 1;
    }
  }
  __syncthreads();
  if (!threadIdx.x && s_tmp == 1)
    *dg_flag = 1;
}

template <typename F>
__host__ void p_jump(F f, int nvertexs, int *dg_flag, int *h_flag) {
  while (1) {
    *h_flag = 0;
    H_ERR(cudaMemcpy(dg_flag, h_flag, sizeof(int), H2D));
    pointer_jumping<<<CTANUM, THDNUM>>>(f.data.dg_wa, nvertexs, dg_flag);
    H_ERR(cudaMemcpy(h_flag, dg_flag, sizeof(int), D2H));
    if (*h_flag == 0)
      break;
  }
  // cudaThreadSynchronize();
}

#endif
