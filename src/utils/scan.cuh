#ifndef __SCAN_CUH
#define __SCAN_CUH

#include "utils/utils.cuh"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

// Blelloch scan
template <typename data_t>
__global__ void __pre_scan(data_t *dg_index, data_t *dg_input,
                           data_t *dg_output, data_t *dg_blk_sum, int n,
                           int blk_sz) {

  extern __shared__ data_t s_tmp[]; // contains blk_sz vaild element

  const int STRIDE = blockDim.x * gridDim.x;
  const int tid = threadIdx.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = 1;

  int ai = tid;
  int bi = tid + (blk_sz / 2);
  int bankoffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankoffsetB = CONFLICT_FREE_OFFSET(bi);
  int th = (1 + (n - 1) / blk_sz) * blk_sz;

  for (int idx = gtid;; idx += STRIDE) {

    // step 1: load to share memory
    int blk_id = (2 * idx) / blk_sz;
    int base = blk_id * blk_sz;
    if (base + ai >= th && base + bi >= th)
      break;
    if (dg_index) {
      s_tmp[ai + bankoffsetA] =
          (base + ai < n) ? dg_input[dg_index[base + ai]] : 0;
      s_tmp[bi + bankoffsetB] =
          (base + bi < n) ? dg_input[dg_index[base + bi]] : 0;
    } else {
      s_tmp[ai + bankoffsetA] = (base + ai < n) ? dg_input[base + ai] : 0;
      s_tmp[bi + bankoffsetB] = (base + bi < n) ? dg_input[base + bi] : 0;
    }

    // step 2: up-sweep
    for (int d = (blk_sz >> 1); d > 0; d >>= 1) {
      __syncthreads();
      if (tid < d) {
        int ai = offset * (2 * tid + 1) - 1;
        int bi = offset * (2 * tid + 2) - 1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);
        s_tmp[bi] += s_tmp[ai];
      }
      offset <<= 1;
    }

    // step 3: write the block sum and clear the last element
    if (tid == 0) {
      if (dg_blk_sum != NULL)
        dg_blk_sum[blk_id] =
            s_tmp[blk_sz - 1 + CONFLICT_FREE_OFFSET(blk_sz - 1)];
      s_tmp[blk_sz - 1 + CONFLICT_FREE_OFFSET(blk_sz - 1)] = 0;
    }

    // step 4: down-sweep
    for (int d = 1; d < blk_sz; d <<= 1) {
      offset >>= 1;
      __syncthreads();
      if (tid < d) {
        int ai = offset * (2 * tid + 1) - 1;
        int bi = offset * (2 * tid + 2) - 1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);

        data_t t = s_tmp[ai];
        s_tmp[ai] = s_tmp[bi];
        s_tmp[bi] += t;
      }
    }

    // step 5: write back to global memory
    __syncthreads();
    if (dg_output) {
      if (base + ai < n)
        dg_output[base + ai] = s_tmp[ai + bankoffsetA];
      if (base + bi < n)
        dg_output[base + bi] = s_tmp[bi + bankoffsetB];
    }
  }
}

// if blk_num < THD_NUM*2
// and the dg_blk_sum is not exact equal to THD_NUM*2
template <typename data_t>
__global__ void __post_scan(data_t *dg_output, data_t *dg_blk_sum, int n,
                            int blk_sz, int blk_num) {
  extern __shared__ data_t s_tmp[]; // contains blk_sz vaild element

  const int STRIDE = blockDim.x * gridDim.x;
  const int tid = threadIdx.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = 1;

  int ai = tid << 1;
  int bi = tid << 1 | 1;
  int bankoffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankoffsetB = CONFLICT_FREE_OFFSET(bi);

  // step 1: load to share memory (maybe bank conflict)
  s_tmp[ai + bankoffsetA] = (ai < blk_num) ? dg_blk_sum[ai] : 0;
  s_tmp[bi + bankoffsetB] = (bi < blk_num) ? dg_blk_sum[bi] : 0;

  // step 2: up-sweep
  for (int d = blk_sz >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      s_tmp[bi] += s_tmp[ai];
    }
    offset <<= 1;
  }

  // step 3: write the block sum and clear the last element
  if (tid == 0)
    s_tmp[blk_sz - 1 + CONFLICT_FREE_OFFSET(blk_sz - 1)] = 0;

  // step 4: down-sweep
  for (int d = 1; d < blk_sz; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      data_t t = s_tmp[ai];
      s_tmp[ai] = s_tmp[bi];
      s_tmp[bi] += t;
    }
  }

  // step 5: write back to global memory
  __syncthreads();

  for (int idx = gtid; (idx << 1) < n; idx += STRIDE) {
    int blk_id = 2 * idx / blk_sz;
    if (idx << 1 < n)
      dg_output[idx << 1] += s_tmp[blk_id + CONFLICT_FREE_OFFSET(blk_id)];
    if (idx << 1 | 1 < n)
      dg_output[idx << 1 | 1] += s_tmp[blk_id + CONFLICT_FREE_OFFSET(blk_id)];
  }
}

template <typename data_t>
__global__ void __final_scan(data_t *dg_output, data_t *dg_blk_sum, int n,
                             int blk_sz) {
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int idx = gtid; idx < n; idx += STRIDE) {
    int blk_id = idx / blk_sz;
    dg_output[idx] += dg_blk_sum[blk_id];
  }
}

template <size_t CTA_NUM = 256, size_t THD_NUM = 256, typename data_t>
__host__ void __recursive_scan(data_t *dg_input, int n, cudaStream_t &stream) {
  data_t *dg_blk_sum;
  const int blk_sz = THD_NUM << 1;
  const int blk_num = 1 + (n - 1) / blk_sz;
  const int padding = CONFLICT_FREE_OFFSET(blk_sz - 1);
  // TODO: Messag Pool
  H_ERR(cudaMalloc((void **)&dg_blk_sum, sizeof(data_t) * blk_num));

  __pre_scan<data_t>
      <<<CTA_NUM, THD_NUM, (padding + blk_sz) * sizeof(data_t), stream>>>(
          NULL, dg_input, dg_input, dg_blk_sum, n, blk_sz);

  // cudaThreadSynchronize();
  if (blk_num <= blk_sz) {
    __post_scan<data_t>
        <<<CTA_NUM, THD_NUM, (padding + blk_sz) * sizeof(data_t), stream>>>(
            dg_input, dg_blk_sum, n, blk_sz, blk_num);
  } else {
    __recursive_scan<CTA_NUM, THD_NUM, data_t>(dg_blk_sum, blk_num, stream);
    __final_scan<data_t>
        <<<CTA_NUM, THD_NUM, 0, stream>>>(dg_input, dg_blk_sum, n, blk_sz);
  }

  H_ERR(cudaFree(dg_blk_sum));
}

/****************************************************
 * Usage:
 * scan<CTA_NUM, THD_NUM>(dg_index, dg_input, dg_output, n);
 *  - dg_index: the index array
 *  - dg_input: the input array
 *  - dg_output: prefix sum array
 *  - n: the size of dg_input
 *    (n must be exact times of 2*THD_NUM)
 ****************************************************/
template <size_t CTA_NUM = 256, size_t THD_NUM = 256, typename data_t>
__host__ void scan(data_t *dg_index, data_t *dg_input, data_t *dg_output, int n,
                   cudaStream_t &stream) {
  data_t *dg_blk_sum;

  const int blk_sz = THD_NUM << 1;
  const int blk_num = 1 + (n - 1) / blk_sz;
  const int padding = CONFLICT_FREE_OFFSET(blk_sz - 1);

  if (n <= blk_sz) {
    __pre_scan<data_t>
        <<<1, THD_NUM, (padding + blk_sz) * sizeof(data_t), stream>>>(
            dg_index, dg_input, dg_output, NULL, n, blk_sz);
  } else if (n <= blk_sz * blk_sz) {
    // TODO: Messag Pool
    H_ERR(cudaMalloc((void **)&dg_blk_sum, sizeof(data_t) * blk_num));

    __pre_scan<data_t>
        <<<CTA_NUM, THD_NUM, (padding + blk_sz) * sizeof(data_t), stream>>>(
            dg_index, dg_input, dg_output, dg_blk_sum, n, blk_sz);

    // cudaThreadSynchronize();

    __post_scan<data_t>
        <<<CTA_NUM, THD_NUM, (padding + blk_sz) * sizeof(data_t), stream>>>(
            dg_output, dg_blk_sum, n, blk_sz, blk_num);

    // TODO: Message Pool
    H_ERR(cudaFree(dg_blk_sum));
  } else {
    // TODO: Messag Pool
    H_ERR(cudaMalloc((void **)&dg_blk_sum, sizeof(data_t) * blk_num));

    __pre_scan<data_t>
        <<<CTA_NUM, THD_NUM, (padding + blk_sz) * sizeof(data_t), stream>>>(
            dg_index, dg_input, dg_output, dg_blk_sum, n, blk_sz);

    // cudaThreadSynchronize();

    __recursive_scan<CTA_NUM, THD_NUM, data_t>(dg_blk_sum, blk_num, stream);

    // cudaThreadSynchronize();

    __final_scan<data_t>
        <<<CTA_NUM, THD_NUM, 0, stream>>>(dg_output, dg_blk_sum, n, blk_sz);

    H_ERR(cudaFree(dg_blk_sum));
  }
}

/****************************************************
 * Usage:
 * scan<CTA_NUM, THD_NUM>(dg_input, dg_output, n);
 *  - dg_input: the input array
 *  - dg_output: prefix sum array
 *  - n: the size of dg_input
 *    (n must be exact times of 2*THD_NUM)
 ****************************************************/
template <size_t CTA_NUM = 256, size_t THD_NUM = 256, typename data_t>
inline __host__ void scan(data_t *dg_input, data_t *dg_output, int n,
                          cudaStream_t &stream) {
  scan<CTA_NUM, THD_NUM, data_t>(NULL, dg_input, dg_output, n, stream);
}

////////////////////////////////////////////////////////////////////////////////
// TODO: UGLY
////////////////////////////////////////////////////////////////////////////////

template <size_t CTA_NUM = 256, size_t THD_NUM = 256, typename data_t>
__host__ void __recursive_scan(data_t *dg_input, int n) {
  data_t *dg_blk_sum;
  const int blk_sz = THD_NUM << 1;
  const int blk_num = 1 + (n - 1) / blk_sz;
  const int padding = CONFLICT_FREE_OFFSET(blk_sz - 1);
  // TODO: Messag Pool
  H_ERR(cudaMalloc((void **)&dg_blk_sum, sizeof(data_t) * blk_num));

  __pre_scan<data_t><<<CTA_NUM, THD_NUM, (padding + blk_sz) * sizeof(data_t)>>>(
      NULL, dg_input, dg_input, dg_blk_sum, n, blk_sz);

  // cudaThreadSynchronize();
  if (blk_num <= blk_sz) {
    __post_scan<data_t>
        <<<CTA_NUM, THD_NUM, (padding + blk_sz) * sizeof(data_t)>>>(
            dg_input, dg_blk_sum, n, blk_sz, blk_num);
  } else {
    __recursive_scan<CTA_NUM, THD_NUM, data_t>(dg_blk_sum, blk_num);
    __final_scan<data_t><<<CTA_NUM, THD_NUM>>>(dg_input, dg_blk_sum, n, blk_sz);
  }

  H_ERR(cudaFree(dg_blk_sum));
}

template <size_t CTA_NUM = 256, size_t THD_NUM = 256, typename data_t>
__host__ void scan(data_t *dg_index, data_t *dg_input, data_t *dg_output,
                   int n) {

  const int blk_sz = THD_NUM << 1;
  const int blk_num = 1 + (n - 1) / blk_sz;
  const int padding = CONFLICT_FREE_OFFSET(blk_sz - 1);

  if (global_dg_blk_sum == NULL)
    H_ERR(cudaMalloc((void **)&global_dg_blk_sum, sizeof(data_t) * blk_num));
  data_t *dg_blk_sum = global_dg_blk_sum;

  if (n <= blk_sz) {
    __pre_scan<data_t><<<1, THD_NUM, (padding + blk_sz) * sizeof(data_t)>>>(
        dg_index, dg_input, dg_output, NULL, n, blk_sz);
  } else if (n <= blk_sz * blk_sz) {
    // TODO: Preallocate
    // H_ERR(cudaMalloc((void **)&dg_blk_sum, sizeof(data_t)*blk_num));

    __pre_scan<data_t>
        <<<CTA_NUM, THD_NUM, (padding + blk_sz) * sizeof(data_t)>>>(
            dg_index, dg_input, dg_output, dg_blk_sum, n, blk_sz);

    // cudaThreadSynchronize();

    __post_scan<data_t>
        <<<CTA_NUM, THD_NUM, (padding + blk_sz) * sizeof(data_t)>>>(
            dg_output, dg_blk_sum, n, blk_sz, blk_num);

    // TODO: Message Pool
    // H_ERR(cudaFree(dg_blk_sum));
  } else {
    // TODO: Messag Pool
    // H_ERR(cudaMalloc((void **)&dg_blk_sum, sizeof(data_t)*blk_num));

    __pre_scan<data_t>
        <<<CTA_NUM, THD_NUM, (padding + blk_sz) * sizeof(data_t)>>>(
            dg_index, dg_input, dg_output, dg_blk_sum, n, blk_sz);

    // cudaThreadSynchronize();

    __recursive_scan<CTA_NUM, THD_NUM, data_t>(dg_blk_sum, blk_num);

    // cudaThreadSynchronize();

    __final_scan<data_t>
        <<<CTA_NUM, THD_NUM, 0>>>(dg_output, dg_blk_sum, n, blk_sz);

    // H_ERR(cudaFree(dg_blk_sum));
  }
}

template <size_t CTA_NUM = 256, size_t THD_NUM = 256, typename data_t>
inline __host__ void scan(data_t *dg_input, data_t *dg_output, int n) {
  scan<CTA_NUM, THD_NUM, data_t>(NULL, dg_input, dg_output, n);
}

#endif
