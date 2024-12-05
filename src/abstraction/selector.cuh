#ifndef __SELECTOR_CUH
#define __SELECTOR_CUH

#include "abstraction/config.cuh"
#include "abstraction/features.cuh"
#include "abstraction/statistics.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

#include "model/select_dir_asso.h"
#include "model/select_dir_asso_backup.h"
#include "model/select_dir_idem.h"
#include "model/select_fmt.h"
#include "model/select_lb_asso_backward.h"
#include "model/select_lb_asso_backward_backup.h"
#include "model/select_lb_asso_forward.h"
#include "model/select_lb_asso_forward_backup.h"
#include "model/select_lb_idem.h"

struct selector_t {

  void select(stat_t stats, feature_t &fets, config_t &conf) {
    // std::cout << fets.push_workload_ratio << " " << fets.pull_workload_ratio
    // << " " << fets.push_workload << " " << fets.pull_workload << " " <<
    // fets.unactive_vertex << " ";
    select_based_on_model(stats, fets, conf);
  }

  // estimate the potential of the fusion, to enable/disable kernel fusion.
  // TODO: BC performs bad since the conf_fuse_inspect is setted by this
  void trigger_fusion(stat_t stats, feature_t fets, config_t &conf) {

    if (!conf.conf_fuse_inspect) { // consider to enable fusion
      double potential = fets.push_workload_ratio + fets.pull_workload_ratio;

      // rule 1: for the long tails
      if (potential < 0.001) {
        conf.conf_fuse_inspect = true;
        conf.conf_switch_to_fusion = true;
      }

      // rule 2: for
      if (fets.growing_rate > 2)
        conf.conf_idle = 0;
      if (conf.conf_idle >= 3) {
        conf.conf_fuse_inspect = true;
        conf.conf_switch_to_fusion = true;
        conf.conf_idle = 0;
      }
    }
  }

  void fusion(stat_t stats, feature_t fets, config_t &conf) {
    conf.conf_dir = Push;
    conf.conf_asfmt = Queue;
    conf.conf_cm = Interleave;
    conf.conf_lb = WM;
    conf.conf_fuse_inspect = true;
    conf.conf_centric = fets.centric;
    conf.conf_fromall = fets.fromall;
    conf.conf_toall = fets.toall;
    if (conf.conf_dir == Push) {
      if (conf.conf_fromall)
        conf.conf_target = All;
      else
        conf.conf_target = Active;
    } else {
      if (conf.conf_toall && conf.conf_dir == Pull)
        conf.conf_target = All;
      else
        conf.conf_target = Inactive;
    }
  }

  void select_idem(stat_t &stats, feature_t &fets, config_t &conf) {
    std::vector<double> v1;
    int l;

    // step 1: selection for direction
    // for(int i=0; i<=18; ++i) v1.push_back(fets.fv[i]);
    v1.push_back(fets.fv[8] / (fets.fv[7] + 0.1));
    v1.push_back(fets.fv[10] / (fets.fv[9] + 0.1));
    v1.push_back(fets.fv[16] / (fets.fv[15] + 0.1));
    v1.push_back(fets.fv[18] / (fets.fv[17] + 0.1));
    v1.push_back(fets.fv[9] / (fets.fv[15] + 0.1));
    v1.push_back(fets.fv[10] / (fets.fv[16] + 0.1));
    v1.push_back(fets.fv[9] / (fets.fv[17] + 0.1));
    v1.push_back(fets.fv[10] / (fets.fv[18] + 0.1));
    l = select_dir_idem(v1);
    conf.conf_dir = (l == 1 ? Pull : Push);
    if (fets.push_workload_ratio * 5 < fets.pull_workload_ratio)
      conf.conf_dir = Push;
    // if(fets.push_workload < (1<<18) && fets.push_workload_ratio*2 <
    // fets.pull_workload_ratio) conf.conf_dir = Push;

    if (cmd_opt.ins.has_direction)
      conf.conf_dir = cmd_opt.ins.d;

    // step 2: selection for asfmt
    v1.clear();
    if (conf.conf_dir == Pull) {
      conf.conf_asfmt = Bitmap;
      if (fets.fv[12] < 0.01)
        conf.conf_asfmt = Queue;
    } else {
      v1.push_back(fets.fv[11]);
      if (select_fmt(v1))
        conf.conf_asfmt = Bitmap;
      else
        conf.conf_asfmt = Queue;
    }
    if (cmd_opt.ins.has_asfmt)
      conf.conf_asfmt = cmd_opt.ins.asf;

    // step 3: selection for load balance
    if (conf.conf_dir == Pull)
      conf.conf_lb = TM;
    else {
      v1.clear();
      for (int i = 0; i < 7; ++i)
        v1.push_back(fets.fv[i]);
      for (int i = 7; i < 18; i += 2)
        v1.push_back(fets.fv[i]);
      l = select_lb_idem(v1);

      if (l == 0)
        conf.conf_lb = WM;
      else if (l == 1)
        conf.conf_lb = CM;
      else
        conf.conf_lb = STRICT;
    }
    if (cmd_opt.ins.has_lb)
      conf.conf_lb = cmd_opt.ins.lb;
  }

  void select_ASSO(stat_t &stats, feature_t &fets, config_t &conf) {
    std::vector<double> v1;
    int l;

    // step 1: selection for direction
    v1.push_back(fets.fv[8] / (fets.fv[7] + 0.1));
    v1.push_back(fets.fv[10] / (fets.fv[9] + 0.1));
    v1.push_back(fets.fv[16] / (fets.fv[15] + 0.1));
    v1.push_back(fets.fv[18] / (fets.fv[17] + 0.1));
    v1.push_back(fets.fv[9] / (fets.fv[15] + 0.1));
    v1.push_back(fets.fv[10] / (fets.fv[16] + 0.1));
    v1.push_back(fets.fv[9] / (fets.fv[17] + 0.1));
    v1.push_back(fets.fv[10] / (fets.fv[18] + 0.1));
    l = select_dir_asso(v1);
    conf.conf_dir = (l == 1 ? Pull : Push);
    if (cmd_opt.ins.has_direction)
      conf.conf_dir = cmd_opt.ins.d;
    // if(fets.push_workload_ratio*5 < fets.pull_workload_ratio) conf.conf_dir =
    // Push;

    if (conf.conf_fromall && conf.conf_toall) {
      v1.clear();
      for (int i = 0; i < 7; ++i)
        v1.push_back(fets.fv[i]);
      // fets.fv[1]/=2;
      int l;
      l = select_dir_asso_backup(v1);
      if (l)
        conf.conf_dir = Pull;
      else
        conf.conf_dir = Push;
      if (cmd_opt.ins.has_direction)
        conf.conf_dir = cmd_opt.ins.d;
    }

    // step 2: selection for asfmt
    v1.clear();
    if (conf.conf_dir == Push)
      v1.push_back(fets.fv[11]);
    else
      v1.push_back(fets.fv[12]);
    if (select_fmt(v1))
      conf.conf_asfmt = Bitmap;
    else
      conf.conf_asfmt = Queue;
    if (fets.fromall && fets.toall && !conf.conf_fixed)
      conf.conf_asfmt = Bitmap;
    if (cmd_opt.ins.has_asfmt)
      conf.conf_asfmt = cmd_opt.ins.asf;

    // step 3: selection for load-balance
    v1.clear();
    for (int i = 0; i < 7; ++i)
      v1.push_back(fets.fv[i]);
    if (conf.conf_dir == Pull) {
      if (conf.conf_fromall && conf.conf_toall && !conf.conf_fixed) {
        l = select_lb_asso_backward_backup(v1);
      } else {
        for (int i = 8; i <= 18; i += 2)
          v1.push_back(fets.fv[i]);
        l = select_lb_asso_backward(v1);
      }
    } else {
      if (conf.conf_fromall && conf.conf_toall && !conf.conf_fixed) {
        l = select_lb_asso_backward_backup(v1);
      } else {
        for (int i = 7; i < 18; i += 2)
          v1.push_back(fets.fv[i]);
        l = select_lb_asso_forward(v1);
      }
    }

    if (l == 0)
      conf.conf_lb = WM;
    else if (l == 1)
      conf.conf_lb = CM;
    else
      conf.conf_lb = STRICT;
    // if(conf.conf_dir==Push) if(fets.fromall && fets.toall) conf.conf_lb = WM;
    if (cmd_opt.ins.has_lb)
      conf.conf_lb = cmd_opt.ins.lb;
  }

  void select_Mono(stat_t &stats, feature_t &fets, config_t &conf) {

    // step 1: selection for direction
    conf.conf_dir = Push;
    if (cmd_opt.ins.has_direction)
      conf.conf_dir = cmd_opt.ins.d;

    // step 1: selection for asfmt
    conf.conf_asfmt = Queue;
    if (cmd_opt.ins.has_asfmt)
      conf.conf_asfmt = cmd_opt.ins.asf;

    // step 1: selection for loadbalance
    // conf.conf_lb = WM;
    std::vector<double> v1;
    int l;
    for (int i = 0; i < 7; ++i)
      v1.push_back(fets.fv[i]);
    for (int i = 7; i < 18; i += 2)
      v1.push_back(fets.fv[i]);
    l = select_lb_asso_forward(v1);
    if (l == 0)
      conf.conf_lb = WM;
    else if (l == 1)
      conf.conf_lb = CM;
    else
      conf.conf_lb = STRICT;
    if (cmd_opt.ins.has_lb)
      conf.conf_lb = cmd_opt.ins.lb;
  }

  void select_based_on_model(stat_t stats, feature_t fets, config_t &conf) {
    TRACE();
    if (fets.centric == EC) {
      conf.conf_target = Active;
      if (fets.active_vertex_ratio > 0.8)
        conf.conf_asfmt = Bitmap;
      else
        conf.conf_asfmt = Queue;
      conf.conf_fuse_inspect = false;
      return;
    }

    // the first time herer, conf.conf_toall and conf.conf_fromall are not set
    if (conf.conf_toall && conf.conf_fromall && !conf.conf_fixed) {
      conf.conf_inherit = true;
      return;
    }

    conf.conf_centric = fets.centric;
    conf.conf_fromall = fets.fromall;
    conf.conf_toall = fets.toall;

    if (fets.pattern == Idem)
      trigger_fusion(stats, fets, conf);
    else
      conf.conf_fuse_inspect = conf.conf_fusion;
    // TODO: make it more general
    // if(conf.conf_fuse_inspect) if(fets.max_deg<33 && fets.avg_deg<33 &&
    // fets.use_root != -1) conf.conf_super_fusion = true;
    if (conf.conf_fuse_inspect) {
      fusion(stats, fets, conf);
      return;
    }

    if (fets.pattern == Idem)
      select_idem(stats, fets, conf);
    else if (fets.pattern == ASSO)
      select_ASSO(stats, fets, conf);
    else if (fets.pattern == Mono)
      select_Mono(stats, fets, conf);

    if (conf.conf_dir == Push) {
      if (conf.conf_fromall)
        conf.conf_target = All;
      else
        conf.conf_target = Active;
    } else {
      if (conf.conf_toall && conf.conf_dir == Pull)
        conf.conf_target = All;
      else
        conf.conf_target = Inactive;
    }
  }
};

#endif
