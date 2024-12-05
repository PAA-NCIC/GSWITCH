#ifndef __NOTIFICATION_CUH
#define __NOTIFICATION_CUH

#include "abstraction/config.cuh"
#include "data_structures/functor.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

template <typename G, typename F>
__global__ void __exit(G g, F f, config_t conf, int *dg_cnt) {
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ int s_tmp;

  if (!threadIdx.x)
    s_tmp = 0;
  __syncthreads();

  for (int idx = gtid; idx < g.nvertexs; idx += STRIDE) {
    bool tag = f.exit(idx, g);
    if (tag & s_tmp == 0)
      s_tmp = 1; // there exist work to do
    if (s_tmp)
      break;
  }

  __syncthreads();
  if (!threadIdx.x && s_tmp)
    dg_cnt[0] = s_tmp;
}

// the only indicator of conergence.
struct notification_t {
  int build() {
    cudaHostAlloc((void **)&h_tag, sizeof(bool), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **)&dg_tag, (void *)h_tag, 0);
    cudaMallocHost((void **)&h_cnt, sizeof(int));
    cudaMalloc((void **)&dg_cnt, sizeof(int));
    reset();
    return sizeof(bool);
  }

  void reset() {
    h_tag[0] = 0;
    already = false;
    h_cnt[0] = 0;
    CLEAN(dg_cnt, 1);
  }

  __device__ void notify_device() {
    if (!already)
      dg_tag[0] = 1;
  }

  void notify_host() {
    h_tag[0] = 1;
    already = true;
  }

  bool is_converged() { return h_tag[0]; }

  int get_cnt() {
    cudaMemcpy(h_cnt, dg_cnt, sizeof(int), D2H);
    return h_cnt[0];
  }

  template <typename E, typename F>
  bool exit(device_graph_t<COO, E> &g, F &f, config_t conf) {
    return true;
  }

  template <typename E, typename F>
  bool exit(device_graph_t<CSR, E> &g, F &f, config_t &conf) {
    if (!conf.conf_window)
      return true; // TODO: remove this for other exit condition
    __exit<<<CTANUM, THDNUM>>>(g, f, conf, dg_cnt);
    if (get_cnt()) {
      reset();
      f.data.window.reset();
      if (conf.conf_fusion) {
        conf.conf_fuse_inspect = false;
        conf.conf_switch_to_fusion = true; // @lzh, understand it?
      }
      return false;
    } else {
      return true;
    }
  }

  bool *h_tag, *dg_tag;
  int *h_cnt, *dg_cnt;
  bool already;
};

#endif
