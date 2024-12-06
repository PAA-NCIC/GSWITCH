#ifndef __CONFIG_CUH
#define __CONFIG_CUH

#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"
#include <fstream>
#include <iostream>

struct config_t {
  void dump(std::ostream &out = std::cout) {
    out << (conf_dir == Push ? "Push" : "Pull") << " ";
    out << (conf_asfmt == Queue ? "Queue" : "Bitmap") << " ";
    switch (conf_lb) {
    case TM:
      out << "TM" << " ";
      break;
    case WM:
      out << "WM" << " ";
      break;
    case CM:
      out << "CM" << " ";
      break;
    case STRICT:
      out << "STRICT" << " ";
      break;
    case TWOD:
      std::cout << "2D-partition";
      break;
    case ELB:
      out << "ELB";
      break;
    case TWC:
      out << "TWC" << " ";
      break;
    default:
      out << "Unknown";
      break;
    }
    out << "\n";
  }

  void show_hints() {
    std::cout << "[";
    std::cout << ((conf_dir == Push) ? "Push|" : "Pull|");
    std::cout << ((conf_asfmt == Queue) ? "Queue|" : "Bitmap|");
    switch (conf_lb) {
    case TM:
      std::cout << "TM";
      break;
    case WM:
      std::cout << "WM";
      break;
    case CM:
      std::cout << "CM";
      break;
    case STRICT:
      std::cout << "STRICT";
      break;
    case TWOD:
      std::cout << "2D-partition";
      break;
    case ELB:
      std::cout << "ELB";
      break;
    case TWC:
      std::cout << "TWC";
      break;
    default:
      std::cout << "Unknown";
      break;
    }
    std::cout << ((conf_fuse_inspect ? "|Fused" : "|Standalone"));
    std::cout << "] ";
  }

  void reset() {
    conf_first_round = true;
    conf_switch_to_fusion = false;
    conf_switch_to_standalone = false;
    conf_fuse_inspect = false;
    conf_idle = 0;
  }

  __device__ __forceinline__ bool ignore_u_state() {
    return conf_ignore_u_state;
  }

  __device__ __forceinline__ Status want() { return conf_target; }

  __device__ __forceinline__ bool pruning() { return conf_pruning; }

  __device__ __forceinline__ bool fuse_inspect() { return conf_fuse_inspect; }

  Direction conf_dir;
  Status conf_target;
  CullMode conf_cm;
  ASFmt conf_asfmt;
  LB conf_lb;
  Centric conf_centric;
  QueueMode conf_qmode;

  bool conf_fromall = false;
  bool conf_toall = false;
  bool conf_fixed = false;
  bool conf_inherit = false; // used for all to all scheme, need reset
  bool conf_2d_partition = false;

  bool conf_first_round = true;
  bool conf_switch_to_fusion = false;
  bool conf_switch_to_standalone = false;
  bool conf_pruning = false;
  bool conf_once = false;
  bool conf_ignore_u_state = false;
  bool conf_fuse_inspect = false;
  bool conf_lite = false;
  bool conf_window = false;
  bool conf_compensation = false;
  bool conf_super_fusion = false;
  bool conf_fusion = false;
  int conf_idle = 0;
  int ctanum = CTANUM_EXPAND;
  int thdnum = THDNUM_EXPAND;
};

#endif
