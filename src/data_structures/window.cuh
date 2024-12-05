#ifndef __WINDOW_CUH
#define __WINDOW_CUH

#include "abstraction/config.cuh"
#include "abstraction/features.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

#include "model/select_stepping.h"

template <typename E, typename F>
__global__ void __probe(device_graph_t<COO, E> g, F f) {}

// TODO: dangerous
template <typename E, typename F>
__global__ void // only for VC
__probe(device_graph_t<CSR, E> g, F f) {
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;

  int num = 0;
  for (int idx = gtid; idx < g.nvertexs; idx += STRIDE) {
    int x = *(int *)(void *)f.wa_of(idx);
    int y = *(int *)(void *)f.ra_of(idx);
    if (x != y && f.data.window.in(x))
      num++;
  }

  __syncthreads();
  num = blockReduceSum(num);
  if (!threadIdx.x)
    atomicAdd(&f.data.window.dg_cnt[0], num);
}

template <typename G, typename F> int probe(G g, F &f) {
  cudaMemset(f.data.window.dg_cnt, 0, sizeof(int));
  f.data.window.h_cnt[0] = 0;
  // cudaThreadSynchronize();
  __probe<<<CTANUM, THDNUM>>>(g, f);
  cudaMemcpy(f.data.window.h_cnt, f.data.window.dg_cnt, sizeof(int), D2H);
  return f.data.window.h_cnt[0];
}

struct window_t {
  double th, wsz_lim, wsz;
  // double shrink_rate = 0.3;
  // double inflat_rate = 0.2;
  // double reset_rate = 0.5;
  double shrink_rate = 0.3;
  double inflat_rate = 0.1;
  double reset_rate = 1;
  int win_bot = 3328;
  int win_top = 33679;
  bool enable = false;
  int *h_cnt, *dg_cnt;

  void build() {
    cudaMallocHost((void **)&h_cnt, sizeof(int));
    cudaMalloc((void **)&dg_cnt, sizeof(int));
    h_cnt[0] = 0;
  }

  bool too_large(double size) { return size > 5 * 33679; }
  bool ok(int size) { return size <= win_top && size > win_bot; }

  void set_lim(int bot, int top) {
    win_bot = bot;
    win_top = top;
  }

  void adjust(feature_t fets, config_t conf) {
    if (conf.conf_fusion) {
      wsz = wsz_lim;
      return;
    };
    // TODO: decide weather to adjust
    std::vector<double> v1;
    double last_processed = fets.active_vertex;
    double estimated = last_processed * fets.cur_avg_deg_active;
    v1.push_back(last_processed);
    v1.push_back(estimated);
    int l = select_stepping(v1);
    if (enable) {
      if (l == 1) {
        shrink();
        if (too_large(estimated))
          shrink();
      } else if (l == 0) {
        inflat();
      }
    }
    // std::cout << last_processed << " " << wsz << " " << wsz_lim << std::endl;
  }

  void set_init_winsize(double _wsz_lim) {
    th = 0;
    wsz_lim = _wsz_lim;
    wsz = _wsz_lim;
  }

  void shrink() { wsz = wsz * shrink_rate; }
  void inflat() {
    wsz = wsz / inflat_rate;
    if (wsz > wsz_lim)
      wsz = wsz_lim;
  }

  __device__ __tbdinline__ bool in(float x) {
    if (!enable)
      return true;
    else
      return x < th + wsz && x >= th;
  }

  __device__ __tbdinline__ bool less(float x) { return x < th; }

  __device__ __tbdinline__ bool greater(float x) { return x >= th + wsz; }

  void reset() {
    th = th + wsz;
    wsz = wsz_lim * reset_rate;
  }
};

#endif
