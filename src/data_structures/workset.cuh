#ifndef __WORKSET_CUH
#define __WORKSET_CUH

#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

// used for LB:STRICT strategy
struct workset_t {
  int64_t build(int nvertexs) {
    H_ERR(cudaMalloc((void **)&dg_degree, sizeof(int) * (nvertexs + 1)));
    H_ERR(cudaMalloc((void **)&dg_udegree, sizeof(int) * (nvertexs + 1)));
    H_ERR(cudaMalloc((void **)&dg_size, sizeof(int)));
    H_ERR(cudaMalloc((void **)&dg_seid_per_blk,
                     sizeof(int) * (CTANUM_EXPAND + 1)));
    H_ERR(cudaMalloc((void **)&dg_idx, sizeof(int) * (CTANUM_EXPAND + 1)));
    int64_t gpu_bytes = sizeof(int) * (nvertexs + 1);
    return gpu_bytes;
  }

  void reset() { /*do nothing*/ }

  int get_usize() {
    int size;
    TOHOST(dg_size, &size, 1);
    return size;
  }

  int *get_udegre(int size) {
    int *udegree = (int *)malloc(sizeof(int) * size);
    TOHOST(dg_udegree, udegree, size);
    return udegree;
  }

  int *get_seid(int size) {
    int *seid = (int *)malloc(sizeof(int) * size);
    TOHOST(dg_seid_per_blk, seid, size);
    return seid;
  }

  int *dg_udegree;      // nvertexs
  int *dg_degree;       // nvertexs
  int *dg_size;         // 1
  int *dg_seid_per_blk; // CTANUM_EXPAND+1
  int *dg_idx;          // CTANUM_EXPAND+1
};

#endif
