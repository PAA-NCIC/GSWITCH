#ifndef __expand_VC_TWOD_CUH
#define __expand_VC_TWOD_CUH

#include "abstraction/config.cuh"
#include "data_structures/active_set.cuh"
#include "data_structures/functor.cuh"
#include "data_structures/graph.cuh"
#include "kernel_libs/kernel_fusion.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

template <typename G, typename F>
__global__ void __phase1_light(active_set_t as, G g, F f) {
  using edata_t = typename G::edge_t;
  using vdata_t = typename F::wa_t;

  __shared__ vdata_t sv[BSZ];

  int base = blockIdx.x * BSZ;
  // step1: load into shared memory
  for (int i = threadIdx.x; i < BSZ; i += blockDim.x) {
    // WARNING: 2d-partition with edge weight not implemented.
    if (base + i < g.nvertexs)
      sv[i] = f.emit(base + i, NULL, g);
  }
  __syncthreads();

  // step2: store parial contribution int ot stream buffer
  for (int i = g.dg_chunks.ces[blockIdx.x] + threadIdx.x;
       i < g.dg_chunks.ces[blockIdx.x + 1]; i += blockDim.x) {
    int pos_to_write = g.dg_chunks.spf[i >> 3] + (threadIdx.x & 7);
    int pos_to_read = g.dg_chunks.sps[i];
    if (pos_to_read != -1)
      f.data.sb[pos_to_write] = sv[pos_to_read];
  }
}

template <typename G, typename F>
__global__ void __phase1_heavy(active_set_t as, G g, F f) {
  using edata_t = typename G::edge_t;
  using vdata_t = typename F::wa_t;

  __shared__ vdata_t sv[BSZ];
  // step1: load into shared memory
  int base = g.dg_chunks.hid[blockIdx.x] * BSZ;
  int write_base = g.dg_chunks.dpf[blockIdx.x];
  for (int i = threadIdx.x; i < BSZ; i += blockDim.x) {
    // WARNING: 2d-partition with edge weight not implemented.
    if (base + i < g.nvertexs)
      sv[i] = f.emit(base + i, NULL, g);
    f.data.sb[write_base + i] = f.data.zero;
  }
  __syncthreads();

  // step2: accumulate partial contribution and store them into stream buffer
  int cur = -1;
  vdata_t val = f.data.zero;
  for (int i = g.dg_chunks.segpos[blockIdx.x] + threadIdx.x;
       i < g.dg_chunks.segpos[blockIdx.x + 1]; i += blockDim.x) {
    int x = g.dg_chunks.segs[i];
    if (x == -1)
      continue;
    if (x < 0) {
      if (cur != -1)
        f.compAtomic(&f.data.sb[write_base + cur], val, g);
      cur = HASH(x);
      val = f.data.zero;
    } else {
      f.comp(&val, sv[x], g);
      // val += sv[x];
    }
  }
  if (cur != -1)
    f.compAtomic(&f.data.sb[write_base + cur], val, g);
}

template <typename G, typename F>
__global__ void __phase2(active_set_t as, G g, F f) {
  using edata_t = typename G::edge_t;
  using vdata_t = typename F::wa_t;

  __shared__ vdata_t sv[BSZ];

  // step1: load into shared memory
  int base = blockIdx.x * BSZ;
  for (int i = threadIdx.x; i < BSZ; i += blockDim.x) {
    sv[i] = f.data.zero;
  }

  __syncthreads();

  // step2: accumulate partial contribution
  for (int i = g.dg_chunks.cey[blockIdx.x] + threadIdx.x;
       i < g.dg_chunks.cey[blockIdx.x + 1]; i += blockDim.x) {
    if (!__equals(f.data.sb[i], f.data.zero) && g.dg_chunks.spt[i] >= 0)
      f.compAtomic(&sv[g.dg_chunks.spt[i]], f.data.sb[i], g);
    // if(g.dg_chunks.spt[i]>=0) f.compAtomic(&sv[g.dg_chunks.spt[i]],
    // f.data.sb[i], g); atomicAdd(&sv[g.dg_chunks.spt[i]], g.dg_chunks.sb[i]);
  }

  __syncthreads();

  // step3: update vertex data
  for (int i = threadIdx.x; i < BSZ; i += blockDim.x) {
    if (base + i < g.nvertexs)
      *f.wa_of(base + i) = sv[i];
  }
}

template <Direction D> struct ExpandProxy<VC, TWOD, D> {
  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<CSR, E> g, F f,
                     config_t conf) {
    __phase1_light<<<g.dg_chunks.scnt, THDNUM_EXPAND, 0,
                     global_helper.stream[0]>>>(as, g, f);
    __phase1_heavy<<<g.dg_chunks.dcnt, THDNUM_EXPAND, 0,
                     global_helper.stream[1]>>>(as, g, f);
    // cudaThreadSynchronize();
    __phase2<<<g.dg_chunks.csize, THDNUM_EXPAND>>>(as, g, f);
  }
  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<COO, E> g, F f,
                     config_t conf) {}
};

#endif
