#ifndef __expand_VC_STRICT_CUH
#define __expand_VC_STRICT_CUH

#include "abstraction/config.cuh"
#include "data_structures/active_set.cuh"
#include "data_structures/functor.cuh"
#include "data_structures/graph.cuh"
#include "kernel_libs/kernel_fusion.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_sortedsearch.hxx>

const int SCRATH = 256; // process SCRATH vertexs in each epoch at most
struct smem_t {
  int eid_start; // process eid idx in active_edges
  int eid_size;
  int vidx_start; // process vertex idx in active_edges
  int vidx_size;
  int processed;
  int vidx_cur_start;
  int vidx_cur_size;
  int chunk_end;

  int v[SCRATH];
  int v_start_pos[SCRATH];   // from vidx_start to min(256,vidx_end)
  int v_degree_scan[SCRATH]; // from vidx_start to min(256,vidx_end)
};

template <ASFmt fmt, QueueMode M, typename G>
__global__ void __prepare(active_set_t as, G g, config_t conf) {
  const int STRIDE = blockDim.x * gridDim.x;
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  // const int assize = ASProxy<fmt,M>::get_size_hard(as);
  const int assize = ASProxy<fmt, M>::get_size(as);

  Status want = conf.want();

  int v, num;

  for (int idx = gtid; idx < assize; idx += STRIDE) {
    v = ASProxy<fmt, M>::fetch(as, idx, want);
    if (v >= 0) {
      if (conf.conf_dir == Push)
        num = tex1Dfetch<int>(g.dt_odegree, v);
      else
        num = g.get_in_degree(v);
    } else
      num = 0;
    as.workset.dg_degree[idx] = num + 1;
  }

  // block reduce
  // tmp = blockReduceSum(tmp);
  // if(!threadIdx.x) atomicAdd(as.workset.dg_size, tmp);
}

template <ASFmt fmt, QueueMode M, typename G, typename F>
__global__ void __expand_VC_STRICT_fused(active_set_t as, G g, F f,
                                         config_t conf) {
  const int *__restrict__ strict_adj_list = g.dg_adj_list;

  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  const int assize = ASProxy<fmt, M>::get_size(as);
  if (assize == 0) {
    if (gtid == 0)
      as.halt_device();
    return;
  }
  const int tid = threadIdx.x;
  Status want = conf.want();

  __shared__ smem_t smem;
  if (threadIdx.x == 0) {
    smem.vidx_start = __ldg(as.workset.dg_idx + blockIdx.x);
    smem.eid_start =
        __ldg(as.workset.dg_seid_per_blk + blockIdx.x) - smem.vidx_start;
    int vidx_end = __ldg(as.workset.dg_idx + blockIdx.x + 1);
    int eid_end = __ldg(as.workset.dg_seid_per_blk + blockIdx.x + 1) - vidx_end;
    smem.vidx_start -= smem.vidx_start > 0 ? 1 : 0;
    smem.vidx_size = vidx_end - smem.vidx_start;
    smem.eid_size = eid_end - smem.eid_start;
    smem.processed = 0;
  }
  __syncthreads();

  if (smem.eid_size <= 0)
    return;

  while (smem.processed < smem.vidx_size) {
    // compute workload for this round
    __syncthreads();
    if (threadIdx.x == 0) {
      smem.vidx_cur_start = smem.vidx_start + smem.processed;
      int rest = smem.vidx_size - smem.processed;
      smem.vidx_cur_size = rest > SCRATH ? SCRATH : rest; // limits
      smem.processed += smem.vidx_cur_size;
      int end_idx = smem.vidx_cur_start + smem.vidx_cur_size;
      if (end_idx < assize)
        smem.chunk_end = __ldg(as.workset.dg_udegree + end_idx) - end_idx;
      else
        smem.chunk_end = smem.eid_start + smem.eid_size;
    }
    __syncthreads();

    // load the values for this round, smem should have enough space
    for (int i = tid; i < smem.vidx_cur_size; i += blockDim.x) {
      int idx = smem.vidx_cur_start + i;
      int v = ASProxy<fmt, M>::fetch(as, idx, want);
      smem.v[i] = v;
      smem.v_degree_scan[i] = __ldg(as.workset.dg_udegree + idx) - idx;
      if (v >= 0) {
        smem.v_start_pos[i] = tex1Dfetch<int>(g.dt_start_pos, v);
      }
    }
    __syncthreads();

    // compute the interval of this round [block_start, block_end)
    int block_start = smem.v_degree_scan[0];
    block_start = block_start < smem.eid_start ? smem.eid_start : block_start;
    int block_end = smem.chunk_end;
    block_end = (block_end > (smem.eid_start + smem.eid_size))
                    ? (smem.eid_start + smem.eid_size)
                    : block_end;
    int block_size = block_end - block_start;

    // process the vertices in interleave mode
    int vidx, v, v_start_pos, v_degree_scan, ei;
    for (int idx = tid; idx < block_size; idx += blockDim.x) {
      int eid = block_start + idx;
      vidx = __upper_bound(smem.v_degree_scan, smem.vidx_cur_size, eid) - 1;
      v = smem.v[vidx];
      if (v < 0)
        continue;
      v_start_pos = smem.v_start_pos[vidx];
      v_degree_scan = smem.v_degree_scan[vidx];
      int uidx = eid - v_degree_scan;
      int u = __ldg(strict_adj_list + uidx + v_start_pos);
      ei = uidx + v_start_pos;
      bool toprocess = true;
      auto vdata = f.emit(v, g.fetch_edata(ei), g);

      // check 1: if idempotent, we can prune the redundant update
      if (toprocess && conf.pruning())
        toprocess = as.bitmap.mark_duplicate_lite(u);

      // check 2: if not push TO ALL, the target vertex must be Inactive
      if (toprocess && !conf.conf_toall)
        toprocess = f.cond(u, vdata, g);

      // if u pass all the checks, do the computation in the functor
      if (toprocess) {
        toprocess = f.compAtomic(f.wa_of(u), vdata, g);
      }

      // check 3:  enqueue the u only once. (if duplicate, wrong answer)
      if (toprocess && !conf.pruning())
        toprocess = as.bitmap.mark_duplicate_atomic(u);

      // if u is updated successfully, write u to the queue directly
      // atomic mode.
      if (toprocess) {
        Qproxy<M>::push(as.queue, u);
      }
    } // for
  } // while
}

template <ASFmt fmt, QueueMode M, typename G, typename F>
__global__ void __expand_VC_STRICT_wtf(active_set_t as, G g, F f,
                                       config_t conf) {
  const int *__restrict__ strict_adj_list = g.dg_adj_list;

  const int assize = ASProxy<fmt, M>::get_size(as);
  const int tid = threadIdx.x;
  Status want = conf.want();

  __shared__ smem_t smem;
  if (threadIdx.x == 0) {
    smem.vidx_start = __ldg(as.workset.dg_idx + blockIdx.x);
    smem.eid_start =
        __ldg(as.workset.dg_seid_per_blk + blockIdx.x) - smem.vidx_start;
    int vidx_end = __ldg(as.workset.dg_idx + blockIdx.x + 1);
    int eid_end = __ldg(as.workset.dg_seid_per_blk + blockIdx.x + 1) - vidx_end;
    smem.vidx_start -= smem.vidx_start > 0 ? 1 : 0;
    smem.vidx_size = vidx_end - smem.vidx_start;
    smem.eid_size = eid_end - smem.eid_start;
    smem.processed = 0;
  }
  __syncthreads();

  if (smem.eid_size <= 0)
    return;

  while (smem.processed < smem.vidx_size) {
    //  // compute workload for this round
    __syncthreads();
    if (threadIdx.x == 0) {
      smem.vidx_cur_start = smem.vidx_start + smem.processed;
      int rest = smem.vidx_size - smem.processed;
      smem.vidx_cur_size = rest > SCRATH ? SCRATH : rest; // limits
      smem.processed += smem.vidx_cur_size;
      int end_idx = smem.vidx_cur_start + smem.vidx_cur_size;
      if (end_idx < assize)
        smem.chunk_end = __ldg(as.workset.dg_udegree + end_idx) - end_idx;
      else
        smem.chunk_end = smem.eid_start + smem.eid_size;
    }
    __syncthreads();

    // load the values for this round, smem should have enough space
    for (int i = tid; i < smem.vidx_cur_size; i += blockDim.x) {
      int idx = smem.vidx_cur_start + i;
      int v = ASProxy<fmt, M>::fetch(as, idx, want);
      smem.v[i] = v;
      smem.v_degree_scan[i] = __ldg(as.workset.dg_udegree + idx) - idx;
      smem.v_start_pos[i] = tex1Dfetch<int>(g.dt_start_pos, v);
    }
    __syncthreads();

    // compute the interval of this round [block_start, block_end)
    int block_start = smem.v_degree_scan[0];
    block_start = block_start < smem.eid_start ? smem.eid_start : block_start;
    int block_end = smem.chunk_end;
    block_end = (block_end > (smem.eid_start + smem.eid_size))
                    ? (smem.eid_start + smem.eid_size)
                    : block_end;
    int block_size = block_end - block_start;

    // process the vertices in interleave mode
    // int vidx,v,v_start_pos,v_degree_scan,ei;
    for (int idx = tid; idx < block_size; idx += blockDim.x) {
      int eid = block_start + idx;
      int vidx = __upper_bound(smem.v_degree_scan, smem.vidx_cur_size, eid) - 1;
      int v = smem.v[vidx];
      int v_start_pos = smem.v_start_pos[vidx];
      int v_degree_scan = smem.v_degree_scan[vidx];
      int uidx = eid - v_degree_scan;
      int u = __ldg(strict_adj_list + uidx + v_start_pos);
      int ei = uidx + v_start_pos;
      bool toprocess = true;

      // check 1: if idempotent, we can prune the redundant update
      if (toprocess)
        toprocess = as.bitmap.mark_duplicate_lite(u);

      // check 2: if not push TO ALL, the target vertex must be Inactive
      if (toprocess && !conf.conf_toall)
        toprocess = as.bitmap.is_inactive(u);

      // if u pass all the checks, do the computation in the functor
      if (toprocess) {
        auto vdata = f.emit(v, g.fetch_edata(ei), g);
        f.compAtomic(f.wa_of(u), vdata, g);
      }
    } // for
  } // while
}

template <ASFmt fmt, QueueMode M, typename G, typename F>
__global__ void __expand_VC_STRICT(active_set_t as, G g, F f, config_t conf) {
  const int *__restrict__ strict_adj_list = g.dg_adj_list;

  const int assize = ASProxy<fmt, M>::get_size(as);
  const int tid = threadIdx.x;
  Status want = conf.want();

  __shared__ smem_t smem;
  if (threadIdx.x == 0) {
    smem.vidx_start = __ldg(as.workset.dg_idx + blockIdx.x);
    smem.eid_start =
        __ldg(as.workset.dg_seid_per_blk + blockIdx.x) - smem.vidx_start;
    int vidx_end = __ldg(as.workset.dg_idx + blockIdx.x + 1);
    int eid_end = __ldg(as.workset.dg_seid_per_blk + blockIdx.x + 1) - vidx_end;
    smem.vidx_start -= smem.vidx_start > 0 ? 1 : 0;
    smem.vidx_size = vidx_end - smem.vidx_start;
    smem.eid_size = eid_end - smem.eid_start;
    smem.processed = 0;
  }
  __syncthreads();

  if (smem.eid_size <= 0)
    return;

  while (smem.processed < smem.vidx_size) {
    //  // compute workload for this round
    __syncthreads();
    if (threadIdx.x == 0) {
      smem.vidx_cur_start = smem.vidx_start + smem.processed;
      int rest = smem.vidx_size - smem.processed;
      smem.vidx_cur_size = rest > SCRATH ? SCRATH : rest; // limits
      smem.processed += smem.vidx_cur_size;
      int end_idx = smem.vidx_cur_start + smem.vidx_cur_size;
      if (end_idx < assize)
        smem.chunk_end = __ldg(as.workset.dg_udegree + end_idx) - end_idx;
      else
        smem.chunk_end = smem.eid_start + smem.eid_size;
    }
    __syncthreads();

    // load the values for this round, smem should have enough space
    for (int i = tid; i < smem.vidx_cur_size; i += blockDim.x) {
      int idx = smem.vidx_cur_start + i;
      int v = ASProxy<fmt, M>::fetch(as, idx, want);
      smem.v[i] = v;
      smem.v_degree_scan[i] = __ldg(as.workset.dg_udegree + idx) - idx;
      if (v >= 0)
        smem.v_start_pos[i] = tex1Dfetch<int>(g.dt_start_pos, v);
      //}
    }
    __syncthreads();

    // compute the interval of this round [block_start, block_end)
    int block_start = smem.v_degree_scan[0];
    block_start = block_start < smem.eid_start ? smem.eid_start : block_start;
    int block_end = smem.chunk_end;
    block_end = (block_end > (smem.eid_start + smem.eid_size))
                    ? (smem.eid_start + smem.eid_size)
                    : block_end;
    int block_size = block_end - block_start;

    // process the vertices in interleave mode
    // int vidx,v,v_start_pos,v_degree_scan,ei;
    for (int idx = tid; idx < block_size; idx += blockDim.x) {
      int eid = block_start + idx;
      int vidx = __upper_bound(smem.v_degree_scan, smem.vidx_cur_size, eid) - 1;
      int v = smem.v[vidx];
      if (v < 0)
        continue;
      int v_start_pos = smem.v_start_pos[vidx];
      int v_degree_scan = smem.v_degree_scan[vidx];
      int uidx = eid - v_degree_scan;
      int u = __ldg(strict_adj_list + uidx + v_start_pos);
      int ei = uidx + v_start_pos;
      bool toprocess = true;

      // check 1: if idempotent, we can prune the redundant update
      if (toprocess && conf.pruning())
        toprocess = as.bitmap.mark_duplicate_lite(u);

      // check 2: if not push TO ALL, the target vertex must be Inactive
      if (toprocess && !conf.conf_toall)
        toprocess = as.bitmap.is_inactive(u);

      // if u pass all the checks, do the computation in the functor
      if (toprocess) {
        auto vdata = f.emit(v, g.fetch_edata(ei), g);
        f.compAtomic(f.wa_of(u), vdata, g);
      }
    } // for
  } // while
}

template <ASFmt fmt, QueueMode M, typename G, typename F>
__global__ void __rexpand_VC_STRICT(active_set_t as, G g, F f, config_t conf) {
  using edata_t = typename G::edge_t;
  using vdata_t = typename F::wa_t;
  const int *__restrict__ strict_adj_list =
      g.directed ? g.dgr_adj_list : g.dg_adj_list;
  edata_t *strict_edgedata = g.directed ? g.dgr_edgedata : g.dg_edgedata;

  const int assize = ASProxy<fmt, M>::get_size(as);
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  if (assize == 0) {
    if (gtid == 0)
      as.halt_device();
    return;
  }
  const int tid = threadIdx.x;
  Status want = conf.want();

  __shared__ smem_t smem;
  if (threadIdx.x == 0) {
    smem.vidx_start = __ldg(as.workset.dg_idx + blockIdx.x);
    smem.eid_start =
        __ldg(as.workset.dg_seid_per_blk + blockIdx.x) - smem.vidx_start;
    int vidx_end = __ldg(as.workset.dg_idx + blockIdx.x + 1);
    int eid_end = __ldg(as.workset.dg_seid_per_blk + blockIdx.x + 1) - vidx_end;
    smem.vidx_start -= smem.vidx_start > 0 ? 1 : 0;
    smem.vidx_size = vidx_end - smem.vidx_start;
    smem.eid_size = eid_end - smem.eid_start;
    smem.processed = 0;
  }
  __syncthreads();

  if (smem.eid_size <= 0)
    return;

  while (smem.processed < smem.vidx_size) {
    // compute workload for this round
    __syncthreads();
    if (threadIdx.x == 0) {
      smem.vidx_cur_start = smem.vidx_start + smem.processed;
      int rest = smem.vidx_size - smem.processed;
      smem.vidx_cur_size = rest > SCRATH ? SCRATH : rest; // limits
      smem.processed += smem.vidx_cur_size;
      int end_idx = smem.vidx_cur_start + smem.vidx_cur_size;
      if (end_idx < assize)
        smem.chunk_end = __ldg(as.workset.dg_udegree + end_idx) - end_idx;
      else
        smem.chunk_end = smem.eid_start + smem.eid_size;
    }
    __syncthreads();

    // load the values for this round, smem should have enough space
    for (int i = tid; i < smem.vidx_cur_size; i += blockDim.x) {
      int idx = smem.vidx_cur_start + i;
      int v = ASProxy<fmt, M>::fetch(as, idx, want);
      smem.v[i] = v;
      smem.v_degree_scan[i] = __ldg(as.workset.dg_udegree + idx) - idx;
      if (v >= 0) {
        smem.v_start_pos[i] = tex1Dfetch<int>(g.dt_start_pos, v);
      }
    }
    __syncthreads();

    // compute the interval of this round [block_start, block_end)
    int block_start = smem.v_degree_scan[0];
    block_start = block_start < smem.eid_start ? smem.eid_start : block_start;
    int block_end = smem.chunk_end;
    block_end = (block_end > (smem.eid_start + smem.eid_size))
                    ? (smem.eid_start + smem.eid_size)
                    : block_end;
    int block_size = block_end - block_start;

    //{// process the vertices in interleave mode
    //  int vidx,v,v_start_pos,v_degree_scan,ei;
    //  for(int idx=tid; idx < block_size; idx+= blockDim.x){
    //    int eid = block_start + idx;
    //    vidx = __upper_bound(smem.v_degree_scan, smem.vidx_cur_size, eid)-1;
    //    v = smem.v[vidx];
    //    if(v < 0) continue;
    //    v_start_pos = smem.v_start_pos[vidx];
    //    v_degree_scan = smem.v_degree_scan[vidx];
    //    int uidx = eid-v_degree_scan;
    //    int u   = __ldg(strict_adj_list+uidx+v_start_pos);
    //    ei = uidx + v_start_pos;
    //    bool toprocess = true;

    //    // Data source must be active all conf_fromall is enabled
    //    if(toprocess && !conf.conf_fromall)
    //      toprocess = as.bitmap.is_active(u);

    //    if(toprocess){
    //      auto vdata = f.emit(u, strict_edgedata+ei, g);
    //      // this vertex may be processed in other CTAs, thus atomic must
    //      remain. f.compAtomic(f.wa_of(v), vdata, g);
    //   }
    //  } //for
    //} // interleave

    { // process the vertices in stride mode
      bool remain = false;
      bool change = true;
      vdata_t reduction;
      int vidx, v, v_start_pos, v_degree_scan, limit;
      int base = block_size / blockDim.x;
      int rest = block_size % blockDim.x;
      int thread_workloads = base + ((threadIdx.x < rest) ? 1 : 0);
      int thread_start =
          (base * threadIdx.x) + ((threadIdx.x < rest) ? threadIdx.x : rest);
      for (int idx = thread_start; idx < thread_start + thread_workloads;
           idx++) {
        int eid = block_start + idx;
        if (change) {
          vidx = __upper_bound(smem.v_degree_scan, smem.vidx_cur_size, eid) - 1;
          v = smem.v[vidx];
          if (v < 0)
            continue;
          v_start_pos = smem.v_start_pos[vidx];
          v_degree_scan = smem.v_degree_scan[vidx];
          limit =
              (vidx == SCRATH - 1 ? block_end : smem.v_degree_scan[vidx + 1]);
        }

        if (eid + 1 == limit)
          change = true;
        else
          change = false;

        int uidx = eid - v_degree_scan;
        int u = __ldg(strict_adj_list + uidx + v_start_pos);
        int ei = uidx + v_start_pos;
        bool toprocess = true;
        // Data source must be active all conf_fromall is enabled
        if (toprocess && !conf.conf_fromall)
          toprocess = as.bitmap.is_active(u);

        if (toprocess) {
          auto vdata = f.emit(u, strict_edgedata + ei, g);
          if (!remain)
            reduction = vdata;
          else
            f.comp(&reduction, vdata, g);
          remain = true;
        }

        if (change && remain) {
          // this vertex may be processed in other CTAs, thus atomic must
          // remain.
          f.compAtomic(f.wa_of(v), reduction, g);
          remain = false;
        }
      } // for
      if (remain) {
        f.compAtomic(f.wa_of(v), reduction, g);
        remain = false;
      }
    } // stride
  } // while
}

template <> struct ExpandProxy<VC, STRICT, Push> {
  template <typename E, typename F>
  static void expand(active_set_t &as, device_graph_t<CSR, E> g, F f,
                     config_t conf) {
    if (!conf.conf_inherit) {
      // step 1: init
      int nactives = (conf.conf_fuse_inspect ? as.get_size_host()
                                             : as.get_size_host_cached());
      if (nactives == 0) {
        as.halt_host();
        return;
      }
      cudaMemset(as.workset.dg_size, 0, sizeof(int));

      // step 2: prepare the degrees and the scaned degrees
      if (as.fmt == Queue) {
        if (as.queue.mode == Normal)
          __prepare<Queue, Normal>
              <<<1 + conf.ctanum / 10, conf.thdnum>>>(as, g, conf);
        else
          __prepare<Queue, Cached>
              <<<1 + conf.ctanum / 10, conf.thdnum>>>(as, g, conf);
      } else
        __prepare<Bitmap, Normal>
            <<<1 + conf.ctanum / 10, conf.thdnum>>>(as, g, conf);

      mgpu::scan<mgpu::scan_type_exc>(
          as.workset.dg_degree, nactives, as.workset.dg_udegree,
          mgpu::plus_t<int>(), as.workset.dg_size, *as.context);

      // step 3: compute the sorted block index.
      int active_edges = as.workset.get_usize();
      int blksz = conf.ctanum;
      __memsetIdx<<<1, conf.ctanum>>>(as.workset.dg_seid_per_blk, blksz,
                                      1 + active_edges / blksz,
                                      active_edges % blksz, active_edges);
      mgpu::sorted_search<mgpu::bounds_lower>(
          as.workset.dg_seid_per_blk, blksz + 1, as.workset.dg_udegree,
          nactives, as.workset.dg_idx, mgpu::less_t<int>(), *as.context);
    }
    if (conf.conf_fuse_inspect) {
      Launch_Expand_VC(STRICT_fused, as, g, f, conf);
    } else {
      if (conf.conf_pruning && conf.conf_asfmt == Queue &&
          as.queue.mode == Normal) {
        // this is just to avoid warp degradation, f**k the nvcc
        __expand_VC_STRICT_wtf<Queue, Normal>
            <<<conf.ctanum, conf.thdnum>>>(as, g, f, conf);
      } else {
        Launch_Expand_VC(STRICT, as, g, f, conf);
      }
    }
    //__expand_VC_STRICT<<<conf.ctanum, conf.thdnum>>>(as, g, f, conf);
    // cudaThreadSynchronize();
  }

  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<COO, E> g, F f,
                     config_t conf) {}
};

template <> struct ExpandProxy<VC, STRICT, Pull> {
  template <typename E, typename F>
  static void expand(active_set_t &as, device_graph_t<CSR, E> g, F f,
                     config_t conf) {
    if (!conf.conf_inherit) {
      // step 1: init
      int nactives = as.get_size_host();
      cudaMemset(as.workset.dg_size, 0, sizeof(int));

      // step 2: prepare the degree and the scaned degree
      if (as.fmt == Queue) {
        if (as.queue.mode == Normal)
          __prepare<Queue, Normal>
              <<<1 + conf.ctanum / 10, conf.thdnum>>>(as, g, conf);
        else
          __prepare<Queue, Cached>
              <<<1 + conf.ctanum / 10, conf.thdnum>>>(as, g, conf);
      } else
        __prepare<Bitmap, Normal>
            <<<1 + conf.ctanum / 10, conf.thdnum>>>(as, g, conf);

      // mgpu::scan<mgpu::scan_type_exc>(as.workset.dg_degree, nactives,
      // as.workset.dg_udegree, *as.context);
      mgpu::scan<mgpu::scan_type_exc>(
          as.workset.dg_degree, nactives, as.workset.dg_udegree,
          mgpu::plus_t<int>(), as.workset.dg_size, *as.context);

      // step 3: compute the sorted block index.
      int active_edges = as.workset.get_usize();
      int blksz = conf.ctanum;
      __memsetIdx<<<1, conf.ctanum>>>(as.workset.dg_seid_per_blk, blksz,
                                      1 + active_edges / blksz,
                                      active_edges % blksz, active_edges);
      mgpu::sorted_search<mgpu::bounds_lower>(
          as.workset.dg_seid_per_blk, blksz + 1, as.workset.dg_udegree,
          nactives, as.workset.dg_idx, mgpu::less_t<int>(), *as.context);
      // as.context->synchronize();
    }
    Launch_RExpand_VC(STRICT, as, g, f, conf);
    //__rexpand_VC_STRICT<<<conf.ctanum, conf.thdnum>>>(as, g, f, conf);
    // cudaThreadSynchronize();
  }

  template <typename E, typename F>
  static void expand(active_set_t as, device_graph_t<COO, E> g, F f,
                     config_t conf) {}
};

#endif
