#ifndef __STATISTICS_CUH
#define __STATISTICS_CUH

#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"
#include <fstream>
#include <iostream>

struct stat_t {

  void build(int _size = 6) {
    size = _size;
    H_ERR(cudaMalloc((void **)&dg_runtime_info, size * sizeof(int)));
    H_ERR(cudaMallocHost((void **)&h_runtime_info, size * sizeof(int)));
  }

  void reset() {}

  void collect(int num) {
    H_ERR(cudaMemcpy(h_runtime_info, dg_runtime_info, num * sizeof(int), D2H));
  }

  void clean() { H_ERR(cudaMemset(dg_runtime_info, 0, sizeof(int) * size)); }

  int active_num() { return h_runtime_info[0]; }
  int unactive_num() { return h_runtime_info[1]; }
  int push_workload() { return h_runtime_info[2]; }
  int pull_workload() { return h_runtime_info[3]; }
  int max_deg_active() { return h_runtime_info[4]; }
  int max_deg_unactive() { return h_runtime_info[5]; }

  void dump(std::ostream &out) {
    for (int i = 0; i < size; ++i) {
      out << h_runtime_info[i] << (i == size - 1 ? "\n" : " ");
    }
  }

  void show_hints() {
    LOG("Iteration %d: (Filter %.3f ms) (Expand %.3f ms) ", level,
        last_filter_time, last_expand_time);
  }

  int *dg_runtime_info;
  int *h_runtime_info;
  int size;

  // history
  double last_time = 0;
  double avg_time = 0;
  double last_filter_time;
  double last_expand_time;
  double avg_filter_time;
  double avg_expand_time;
  int level = 0;
};

#endif
