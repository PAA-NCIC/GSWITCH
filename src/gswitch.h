#ifndef __GSWITCH_H
#define __GSWITCH_H

#include <cuda.h>

#include "utils/fileIO.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

#include "data_structures/active_set.cuh"
#include "data_structures/functor.cuh"
#include "data_structures/graph.cuh"

#include "abstraction/executor.cuh"
#include "abstraction/inspector.cuh"
#include "abstraction/selector.cuh"
#include "model/select_fusion.h"

template <typename G, typename F>
void init_conf(stat_t &stats, feature_t &fets, config_t &conf, G &g, F &f) {
  stats.build();
  fets.architecture_features();
  global_helper.build();
  conf.conf_inherit = false;
  if (fets.use_root >= 0)
    fets.first_workload = g.dg.get_degree(fets.use_root);

  if (fets.centric == VC) {
    fets.flatten();
    std::vector<double> fv;
    for (int i = 2; i < 7; ++i)
      fv.push_back(fets.fv[i]);

    if (select_fusion(fv))
      conf.conf_fuse_inspect = true;
    else
      conf.conf_fuse_inspect = false;

    if (fets.toall && fets.fromall)
      conf.conf_fuse_inspect = false;

    if (cmd_opt.ins.has_fusion) {
      if (cmd_opt.ins.fusion == Fused)
        conf.conf_fuse_inspect = true;
      else
        conf.conf_fuse_inspect = false;
    }
    if (conf.conf_fuse_inspect)
      conf.conf_fusion = true;
    if (!conf.conf_fuse_inspect) {
      conf.conf_qmode = Normal;
    }
  }

  if (conf.conf_window) {
    f.data.window.enable =
        true; // used in device function (which can only touch the as)
    f.data.window.set_init_winsize(
        DIV(8 * g.el.mean_weight * 32, g.hg.attr.avg_deg));
    LOG(" -- window size: %f\n",
        DIV(8 * g.el.mean_weight * 32, g.hg.attr.avg_deg));
  }
}

template <typename G, typename F>
void reset_conf(stat_t &stats, feature_t &fets, config_t &conf, G &g, F &f) {
  conf.reset();
  fets.architecture_features();
  conf.conf_inherit = false;
  if (fets.use_root >= 0)
    fets.first_workload = g.dg.get_degree(fets.use_root);

  if (fets.centric == VC) {
    fets.flatten();
    std::vector<double> fv;
    for (int i = 2; i < 7; ++i)
      fv.push_back(fets.fv[i]);

    if (select_fusion(fv))
      conf.conf_fuse_inspect = true;
    else
      conf.conf_fuse_inspect = false;

    if (fets.toall && fets.fromall)
      conf.conf_fuse_inspect = false;

    if (cmd_opt.ins.has_fusion) {
      if (cmd_opt.ins.fusion == Fused)
        conf.conf_fuse_inspect = true;
      else
        conf.conf_fuse_inspect = false;
    }
    if (conf.conf_fuse_inspect)
      conf.conf_fusion = true;
    if (!conf.conf_fuse_inspect) {
      conf.conf_qmode = Normal;
    }
  }

  if (conf.conf_window) {
    f.data.window.enable =
        true; // used in device function (which can only touch the as)
    f.data.window.set_init_winsize(
        DIV(8 * g.el.mean_weight * 32, g.hg.attr.avg_deg));
    LOG(" -- window size: %f\n",
        DIV(8 * g.el.mean_weight * 32, g.hg.attr.avg_deg));
  }
}

#endif
