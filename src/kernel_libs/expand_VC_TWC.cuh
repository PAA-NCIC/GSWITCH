#ifndef __expand_VC_TWC_CUH
#define __expand_VC_TWC_CUH

#include "abstraction/config.cuh"
#include "data_structures/active_set.cuh"
#include "data_structures/functor.cuh"
#include "data_structures/graph.cuh"
#include "utils/intrinsics.cuh"
#include "utils/utils.cuh"

// This optimization variant has been move to expand_VC_TM.cuh

//__global__ void
//__expand_TWC(active_set_t as, graph_t g, int lvl, int mode){
//  const index_t* __restrict__ strict_adj_list = g.dg_adj_list;
//
//  int STRIDE,gtid,phase,cosize,qsize;
//  if(mode==0){
//    STRIDE = blockDim.x*gridDim.x;
//    gtid   = threadIdx.x + blockIdx.x*blockDim.x;
//    cosize = 1;
//    phase  = 0;
//    qsize  = as.small.get_qsize();
//  }else if(mode==1){
//    STRIDE = (blockDim.x*gridDim.x)>>5;
//    gtid   = (threadIdx.x + blockIdx.x*blockDim.x)>>5;
//    cosize = 32;
//    phase  = (threadIdx.x + blockIdx.x*blockDim.x) & (cosize-1);
//    qsize  = as.medium.get_qsize();
//  }else{
//    STRIDE = gridDim.x;
//    gtid   = blockIdx.x;
//    cosize = blockDim.x;
//    phase  = threadIdx.x;
//    qsize  = as.large.get_qsize();
//  }
//
//
//  for(int idx=gtid; idx<qsize; idx+=STRIDE){
//    int v;
//    if(mode==0) v = tex1Dfetch<int>(as.small.dt_queue, idx);
//    else if(mode==1)  v = tex1Dfetch<int>(as.medium.dt_queue, idx);
//    else v = tex1Dfetch<int>(as.large.dt_queue, idx);
//    int end = tex1Dfetch<int>(g.dt_odegree, v);
//    int start = tex1Dfetch<int>(g.dt_start_pos, v);
//    end += start;
//
//    for(int i=start+phase; i<end; i+=cosize){
//      int u = __ldg(strict_adj_list+i);
//      int u_s = as.bitmap.get_state(u);
//      if(u_s==-1) as.bitmap.mark(u, lvl);
//    }
//    if(mode==1) while(!__all(1));
//    else if(mode==2) __syncthreads();
//  }
//}
//
//__host__ void expand(active_set_t as, graph_t g, int lvl){
//  for(int i=0;i<3;++i){
//    __vexpand<<<CTANUM,THDNUM,0,as.streams[i]>>>(as,g,lvl,i);
//  }
//  for(int i=0;i<3;i++) cudaStreamSynchronize(as.streams[i]);
//}

#endif
