#ifndef __EXECUTOR_CUH
#define __EXECUTOR_CUH

#include "abstraction/config.cuh"
#include "abstraction/features.cuh"
#include "abstraction/statistics.cuh"
#include "kernel_libs/expand_EC.cuh"
#include "kernel_libs/expand_VC_CM.cuh"
#include "kernel_libs/expand_VC_ELB.cuh"
#include "kernel_libs/expand_VC_STRICT.cuh"
#include "kernel_libs/expand_VC_TM.cuh"
#include "kernel_libs/expand_VC_TWC.cuh"
#include "kernel_libs/expand_VC_TWOD.cuh"
#include "kernel_libs/expand_VC_WM.cuh"
#include "kernel_libs/filter.cuh"
#include "kernel_libs/kernel_fusion.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

struct executor_t {
  template <typename G, typename F>
  void filter(active_set_t &as, G &g, F &f, stat_t &stat, feature_t &fets,
              config_t &conf) {
    double s = stat.last_filter_time;
    __filter(as, g, f, conf);

    // specify grid size.
    conf.thdnum = THDNUM_EXPAND * (fets.register_lim / 90);
    if (conf.conf_fuse_inspect) {
      {
        conf.ctanum = 32;
        conf.thdnum = 512;
      }
      // conf.thdnum=512;
      // conf.ctanum=MIN(32, fets.avg_deg*as.get_size_host()/conf.thdnum/2+1);
    } else {
      as.get_size_host();
      int workloads =
          (conf.conf_dir == Push ? fets.push_workload : fets.pull_workload);

      conf.ctanum = MIN(CTANUM_EXPAND, workloads / conf.thdnum / 2 + 1);

      if (conf.ctanum < 4)
        conf.conf_idle += 1;
      else
        conf.conf_idle = 0;

      if (conf.ctanum < 5 && conf.conf_lb == STRICT)
        conf.conf_lb = CM;
      // if(conf.ctanum < fets.sm_num) conf.ctanum = fets.sm_num;
    }

    // vertex cannot update their data in the kernel fusion mode
    // do the update process before the processing.
    // this require the queue has no repetition
    // fortunately, we have used the bitmap to mark the duplicate of the input
    // queue.
    if (conf.conf_centric == VC)
      CompensationProxy<VC>::compensation(as, g, f, conf);

    double e = mwtime();
    // LOG("Level %d: (Cull %.3f ms)", g.level, e-s);

    stat.level = g.level;
    stat.last_filter_time = e - s;
    stat.avg_filter_time =
        (stat.avg_filter_time * g.level + stat.last_filter_time) /
        (g.level + 1);

    fets.last_filter_time = stat.last_filter_time;
    fets.avg_filter_time = stat.avg_filter_time;
  }

  template <typename G, typename F>
  void expand(active_set_t &as, G &g, F &f, stat_t &stat, feature_t &fets,
              config_t &conf) {
    TRACE();
    double s = mwtime();
    // TODO: all in one
    // if(conf.conf_super_fusion){
    //  super_fusion(as, g, f, conf);
    //  CUBARRIER();
    //  as.halt_host();
    //  LOG("\n-- Super Fusion ! --\n");
    //  return;
    //}

    if (conf.conf_centric == EC) {
      ExpandProxy<EC>::expand(as, g, f, conf);
    }
#define ENABLE(lb)                                                             \
  if (conf.conf_lb == lb) {                                                    \
    if (conf.conf_dir == Push)                                                 \
      ExpandProxy<VC, lb, Push>::expand(as, g, f, conf);                       \
    else if (conf.conf_dir == Pull)                                            \
      ExpandProxy<VC, lb, Pull>::expand(as, g, f, conf);                       \
  }

    if (conf.conf_centric == VC) {
      ENABLE(TM)
      ENABLE(WM)
      ENABLE(CM)
      ENABLE(STRICT)
      if (ENABLE_2D_PARTITION)
        ENABLE(TWOD)
      // ENABLE(ELB)
      // ENABLE(TWC)
    }
#undef ENABLE

    CUBARRIER();
    double e = mwtime();
    // LOG(" (Expand %.3f ms)\n",e-s);
    // LOG(" (Expand %.3f ms)",e-s);
    // std::cout << " " << stat.last_filter_time + e-s << std::endl;;

    stat.last_time = stat.last_filter_time + e - s;
    stat.avg_time = (stat.avg_time * (g.level - 1) + stat.last_time) / g.level;
    stat.last_expand_time = e - s;
    stat.avg_expand_time =
        (stat.avg_expand_time * (g.level - 1) + stat.last_expand_time) /
        g.level;

    fets.last_expand_time = stat.last_expand_time;

    // back to nromal mode
    if (fets.pattern == Idem && fets.avg_expand_time > 0 &&
        fets.last_expand_time > 5 * fets.avg_expand_time) {
      conf.conf_fuse_inspect = false;
      as.queue.swap();
    }

    fets.avg_expand_time = stat.avg_expand_time;
    fets.last_time = stat.last_time;
    fets.avg_time = stat.avg_time;
    // std::cout << fets.avg_filter_time << " " << fets.avg_expand_time <<
    // std::endl;
    if (cmd_opt.verbose) {
      stat.show_hints();
      conf.show_hints();
      if (fets.centric == VC)
        std::cout << "<" << conf.ctanum << "," << conf.thdnum << ">";
      else
        std::cout << "<" << CTANUM << "," << THDNUM << ">";
      LOG("\n");
    }
    conf.conf_first_round = false;
  }
};

#endif
