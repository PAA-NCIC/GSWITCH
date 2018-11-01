

set(SROUCE 
  utils/2d_partition.cuh
  utils/cmdline.cuh
  utils/common.cuh
  utils/intrinsics.cuh
  utils/utils.cuh
  utils/block_scan.cuh
  utils/scan.cuh
  utils/fileIO.cuh
  utils/json.cuh
  data_structures/active_set.cuh
  data_structures/bitmap.cuh
  data_structures/queue.cuh
  data_structures/window.cuh
  data_structures/workset.cuh
  data_structures/functor.cuh
  data_structures/graph.cuh
  data_structures/notification.cuh
  kernel_libs/filter.cuh
  kernel_libs/expand_EC.cuh
  kernel_libs/expand_VC_CM.cuh
  kernel_libs/expand_VC_STRICT.cuh
  kernel_libs/expand_VC_TM.cuh
  kernel_libs/expand_VC_TWC.cuh
  kernel_libs/expand_VC_TWOD.cuh
  kernel_libs/expand_VC_WM.cuh
  kernel_libs/kernel_fusion.cuh
  abstraction/config.cuh
  abstraction/statistics.cuh
  abstraction/features.cuh
  abstraction/inspector.cuh
  abstraction/selector.cuh
  abstraction/executor.cuh
  gswitch.h)


#cuda_add_executable(gswitch ${SROUCE})
cuda_add_library(gswitch ${SROUCE} STATIC)
