#ifndef __FEATURES_CUH
#define __FEATURES_CUH

#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"
#include <fstream>
#include <iostream>

#define DIV(a, b) ((b) == 0 ? 0 : ((a + .0) / (b)))

struct feature_t {
  void reset() { json.clean(); }
  void flatten() {
    // dataset
    fv[0] = nvertexs;
    fv[1] = nedges;
    fv[2] = avg_deg; // nedges / nvertexs
    fv[3] = std_deg;
    fv[4] = range;
    fv[5] = GI;
    fv[6] = Her;

    // workload
    fv[7] = active_vertex;
    fv[8] = unactive_vertex;
    fv[9] = push_workload;
    fv[10] = pull_workload;
    fv[11] = active_vertex_ratio;
    fv[12] = unactive_vertex_ratio;
    fv[13] = push_workload_ratio;
    fv[14] = pull_workload_ratio;

    // degree of workload
    fv[15] = cur_avg_deg_active;
    fv[16] = cur_avg_deg_unactive;
    fv[17] = cur_max_deg_active; // DIV(cur_max_deg_active, cur_avg_deg_active);
    fv[18] = cur_max_deg_unactive; // DIV(cur_max_deg_unactive,
                                   // cur_avg_deg_unactive);

    // history
    fv[19] = DIV(last_filter_time, avg_filter_time);
    fv[20] = DIV(last_expand_time, avg_expand_time);
    cross();
  }

  void cross() {
    if (fromall) {
      fv[7] += unactive_vertex;
      fv[9] += pull_workload;
      fv[11] += unactive_vertex_ratio;
      fv[13] += pull_workload_ratio;
      fv[15] += (active_vertex + unactive_vertex == 0)
                    ? 0
                    : ((cur_avg_deg_active * active_vertex +
                        cur_avg_deg_unactive * unactive_vertex) /
                       (active_vertex + unactive_vertex));
      fv[17] = MAX(cur_max_deg_unactive, cur_max_deg_active);
    }
    if (toall) {
      fv[8] += active_vertex;
      fv[10] += push_workload;
      fv[12] += active_vertex_ratio;
      fv[14] += push_workload_ratio;
      fv[16] += (active_vertex + unactive_vertex == 0)
                    ? 0
                    : ((cur_avg_deg_active * active_vertex +
                        cur_avg_deg_unactive * unactive_vertex) /
                       (active_vertex + unactive_vertex));
      fv[18] = MAX(cur_max_deg_active, cur_max_deg_unactive);
    }

    active_vertex = fv[7];
    unactive_vertex = fv[8];
    push_workload = fv[9];
    pull_workload = fv[10];
    active_vertex_ratio = fv[11];
    unactive_vertex_ratio = fv[12];
    push_workload_ratio = fv[13];
    pull_workload_ratio = fv[14];
    cur_avg_deg_active = fv[15];
    cur_avg_deg_unactive = fv[16];
    cur_max_deg_active = fv[17];
    cur_max_deg_unactive = fv[18];
  }

  void record() {
    flatten();
    for (int i = 0; i < 21; ++i) {
      json.add(std::to_string(i), fv[i]);
    }
    json.add("21", last_filter_time);
    json.add("22", last_expand_time);
    flush();
  }

  void flush() {
    if (cmd_opt.json != "") {
      std::ofstream out(cmd_opt.json, std::ios::app);
      // std::ofstream out(cmd_opt.json);
      json.dump(out);
      reset();
      out.close();
    }
  }

  void architecture_features() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cmd_opt.device);
    cap = prop.minor;
    while (cap > 1)
      cap /= 10;
    cap += prop.major;
    register_lim = prop.regsPerBlock / 512;
    thread_lim = prop.maxThreadsPerBlock;
    sm_num = prop.multiProcessorCount;
    // printf("  - capbility is %.2f\n", cap);
    // printf("  - max threads per Block: %d\n", thread_lim);
    // printf("  - max register per Block: %d\n", register_lim);
    // printf("  - processor Count: %d\n", sm_num);
    // printf("\n");
  }
  // Device (K40)
  // int max_threads_num = 1024;

  // Algorithm
  Centric centric;
  Computation pattern;
  int use_root = -1;
  bool fromall;
  bool toall;

  // Dataset
  int nvertexs;
  int nedges;
  double avg_deg;
  double std_deg;
  double max_deg;
  double range;
  double GI;
  double Her;
  int avg_edata;

  // Runtime
  int active_vertex;
  int unactive_vertex;
  double active_vertex_ratio;
  double unactive_vertex_ratio;

  int push_workload;
  int pull_workload;
  double push_workload_ratio;
  double pull_workload_ratio;

  double cur_max_deg_active;
  double cur_avg_deg_active;
  double cur_max_deg_unactive;
  double cur_avg_deg_unactive;

  // History
  int level;
  double last_time;
  double avg_time;
  double last_filter_time;
  double last_expand_time;
  double avg_filter_time;
  double avg_expand_time;

  double growing_rate;
  int first_workload = 0;

  // architecture
  double cap;
  int thread_lim;
  int register_lim;
  int sm_num;

  double fv[21];
  json_t json;
};

#endif
