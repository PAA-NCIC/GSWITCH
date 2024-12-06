#ifndef __FUNCTOR_CUH
#define __FUNCTOR_CUH

#include "data_structures/graph.cuh"
#include "data_structures/window.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"
#include <cuda.h>

// WA = write/read attributes
// RA = read-only attributes
template <typename WA, typename RA> struct vdata_t {

  void build(int size) {
    window.build();
    this->size = size;
    if (!std::is_same<Empty, WA>::value) {
      H_ERR(cudaMalloc((void **)&dg_wa, sizeof(WA) * size));
      h_wa = (WA *)malloc(sizeof(WA) * size);
    }
    if (!std::is_same<Empty, RA>::value) {
      H_ERR(cudaMalloc((void **)&dg_ra, sizeof(RA) * size));
      h_ra = (RA *)malloc(sizeof(RA) * size);
    }
    build_stream_buffer(); // for 2d-partition
  }

  // for 2d-partition
  void build_stream_buffer() {
    if (ENABLE_2D_PARTITION)
      H_ERR(cudaMalloc((void **)&sb, sizeof(WA) * SBSIZE));
  }

  template <typename X> void init_wa(X init) {
    for (int i = 0; i < size; ++i) {
      h_wa[i] = init(i);
    }
    H_ERR(cudaMemcpy(dg_wa, h_wa, sizeof(WA) * size, H2D));
  }

  void set_zero(WA z) { zero = z; }

  template <typename X> void init_ra(X init) {
    for (int i = 0; i < size; ++i) {
      h_ra[i] = init(i);
    }
    H_ERR(cudaMemcpy(dg_ra, h_ra, sizeof(RA) * size, H2D));
  }

  void sync_wa() { H_ERR(cudaMemcpy(h_wa, dg_wa, sizeof(WA) * size, D2H)); }

  void sync_ra() { H_ERR(cudaMemcpy(h_ra, dg_ra, sizeof(RA) * size, D2H)); }

  __device__ __tbdinline__ WA *fetch_wa(int vid) { return dg_wa + vid; }

  __device__ __tbdinline__ RA *fetch_ra(int vid) { return dg_ra + vid; }

  WA zero;
  RA *dg_ra;
  WA *dg_wa;
  RA *h_ra;
  WA *h_wa;
  int size;
  WA *sb; // sbsize
  window_t window;
};

template <Centric C, typename WA, typename RA, typename E> struct Functor {};

template <typename WA, typename RA, typename E> struct Functor<EC, WA, RA, E> {
  typedef WA wa_t;
  typedef RA ra_t;
  using G = device_graph_t<COO, E>;
  vdata_t<WA, RA> data;

  __device__ __tbdinline__ WA *wa_of(int v) { return data.fetch_wa(v); }
  __device__ __tbdinline__ RA *ra_of(int v) { return data.fetch_ra(v); }

  __device__ virtual Status filter(int v, int u, E *e) { return Inactive; }
  __device__ virtual void update(int v, int u, E *e) {}
};

template <typename WA, typename RA, typename E> struct Functor<VC, WA, RA, E> {
  typedef WA wa_t;
  typedef RA ra_t;
  using G = device_graph_t<CSR, E>;
  vdata_t<WA, RA> data;

  __device__ __tbdinline__ WA *wa_of(int v) { return data.fetch_wa(v); }
  __device__ __tbdinline__ RA *ra_of(int v) { return data.fetch_ra(v); }

  __device__ virtual Status filter(int v, G g) { return Inactive; }
  __device__ virtual WA emit(int v, E *e, G g) { return *data.fetch_wa(v); }
  __device__ virtual bool cond(int v, WA msg, G g) { return false; }
  __device__ virtual bool comp(WA *v_data, WA msg, G g) { return true; }
  __device__ virtual bool compAtomic(WA *v_data, WA msg, G g) { return true; }
  __device__ virtual bool exit(int v, G g) { return false; }
};

template <Centric C = VC, LB S = WM, Direction D = Push> struct ExpandProxy {};

#endif
