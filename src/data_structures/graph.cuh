#ifndef __GRAPH_H
#define __GRAPH_H

#include "abstraction/features.cuh"
#include "utils/fileIO.cuh"
#include "utils/utils.cuh"

// partial specialization
template <GraphFmt fmt, typename E> struct device_graph_t {};

// edge-centric
template <typename E> struct device_graph_t<COO, E> {
  typedef E edge_t;
  int64_t build(host_graph_t<COO, E> hgraph) {
    this->nvertexs = hgraph.nvertexs;
    this->nedges = hgraph.nedges;
    this->level = 0;

    int64_t gpu_bytes = 0;
    H_ERR(cudaMalloc((void **)&dg_coo, sizeof(packed_t) * hgraph.nedges));
    H_ERR(cudaMemcpy(dg_coo, hgraph.edge_list, sizeof(packed_t) * hgraph.nedges,
                     H2D));
    gpu_bytes += sizeof(packed_t) * hgraph.nedges;
    return gpu_bytes;
  }

  __device__ __tbdinline__ int get_level() { return level; }

  inline int get_degree(int idx) { return 0; }

  // WARNING,TODO: This update is only vaild in the top level
  //  since the primitives we have accept parameter by assignment
  inline void update_level(int inc = 1) { level += inc; }
  inline void reset_level(int inc = 1) { level = 0; }

  int64_t nvertexs;
  int64_t nedges;
  int level;
  packed_t *dg_coo;
  cudaTextureObject_t dt_edgelist;
};

// vertex-centric
template <typename E> struct device_graph_t<CSR, E> {
  typedef E edge_t;

  int64_t build(host_graph_t<CSR, E> hgraph) {
    this->nvertexs = hgraph.nvertexs;
    this->nedges = hgraph.nedges;
    this->level = 0;
    this->directed = hgraph.directed;

    if (ENABLE_2D_PARTITION)
      dg_chunks.todevice(hgraph.chunks);

    int64_t gpu_bytes = 0;
    H_ERR(cudaMalloc((void **)&dg_adj_list, sizeof(int) * hgraph.nedges));
    H_ERR(cudaMalloc((void **)&dg_odegree, sizeof(int) * hgraph.nvertexs));
    H_ERR(cudaMalloc((void **)&dg_start_pos, sizeof(int) * hgraph.nvertexs));

    H_ERR(cudaMemcpy(dg_adj_list, hgraph.adj_list, sizeof(int) * hgraph.nedges,
                     H2D));
    H_ERR(cudaMemcpy(dg_odegree, hgraph.odegrees, sizeof(int) * hgraph.nvertexs,
                     H2D));
    H_ERR(cudaMemcpy(dg_start_pos, hgraph.start_pos,
                     sizeof(int) * hgraph.nvertexs, H2D));

    if (directed) {
      H_ERR(cudaMalloc((void **)&dgr_adj_list, sizeof(int) * hgraph.nedges));
      H_ERR(cudaMalloc((void **)&dgr_odegree, sizeof(int) * hgraph.nvertexs));
      H_ERR(cudaMalloc((void **)&dgr_start_pos, sizeof(int) * hgraph.nvertexs));

      H_ERR(cudaMemcpy(dgr_adj_list, hgraph.r_adj_list,
                       sizeof(int) * hgraph.nedges, H2D));
      H_ERR(cudaMemcpy(dgr_odegree, hgraph.r_odegrees,
                       sizeof(int) * hgraph.nvertexs, H2D));
      H_ERR(cudaMemcpy(dgr_start_pos, hgraph.r_start_pos,
                       sizeof(int) * hgraph.nvertexs, H2D));
    }

    if (!std::is_same<Empty, E>::value) {
      LOG(" -- edgedata is not Empty\n");
      H_ERR(cudaMalloc((void **)&dg_edgedata, sizeof(E) * hgraph.nedges));
      H_ERR(cudaMemcpy(dg_edgedata, hgraph.edge_weights,
                       sizeof(int) * hgraph.nedges, H2D));

      if (directed) {
        H_ERR(cudaMalloc((void **)&dgr_edgedata, sizeof(E) * hgraph.nedges));
        H_ERR(cudaMemcpy(dgr_edgedata, hgraph.r_edge_weights,
                         sizeof(int) * hgraph.nedges, H2D));
      }
      // if(std::is_same<float,E>::value) build_tex<float>(dt_edgedata,
      // (float*)dg_edgedata, nedges); else if(std::is_same<int,E>::value)
      // build_tex<int>(dt_edgedata, (int*)dg_edgedata, nedges);
    }

    build_tex<int>(dt_odegree, dg_odegree, nvertexs);
    build_tex<int>(dt_start_pos, dg_start_pos, nvertexs);

    if (directed) {
      build_tex<int>(dtr_odegree, dgr_odegree, nvertexs);
      build_tex<int>(dtr_start_pos, dgr_start_pos, nvertexs);
    }

    gpu_bytes += sizeof(int) * nvertexs * 3;
    gpu_bytes += sizeof(int) * nedges;

    if (directed) {
      gpu_bytes <<= 1;
    }

    return gpu_bytes;
  }

  __device__ __tbdinline__ E *fetch_edata(const int eid) {
    return dg_edgedata + eid;
  }

  __device__ __tbdinline__ E *fetch_edata_r(const int eid) {
    return dgr_edgedata + eid;
  }

  __device__ __tbdinline__ int get_out_degree(const int vid) {
    return tex1Dfetch<int>(dt_odegree, vid);
  }

  __device__ __tbdinline__ int get_in_degree(const int vid) {
    if (directed)
      return tex1Dfetch<int>(dtr_odegree, vid);
    return tex1Dfetch<int>(dt_odegree, vid);
  }

  __device__ __tbdinline__ int get_in_start_pos(const int vid) {
    if (directed)
      return tex1Dfetch<int>(dtr_start_pos, vid);
    return tex1Dfetch<int>(dt_start_pos, vid);
  }

  __device__ __tbdinline__ int get_level() { return level; }

  inline int get_degree(int idx) {
    int ret;
    TOHOST(dg_odegree + idx, &ret, 1);
    return ret;
  }

  // WARNING,TODO: This update is only vaild in the top level
  //  since the primitives we have accept parameter by assignment
  __host__ __device__ inline void update_level(int inc = 1) { level += inc; }
  __host__ __device__ inline void reset_level(int inc = 1) { level = 0; }

  chunk_t dg_chunks;

  E *dg_edgedata;
  int *dg_adj_list;
  int *dg_start_pos;
  int *dg_odegree;

  // for reversed graph
  E *dgr_edgedata;
  int *dgr_adj_list;
  int *dgr_start_pos;
  int *dgr_odegree;

  int nvertexs;
  int nedges;
  int level;
  bool directed;
  // cudaTextureObject_t dt_edgedata;
  cudaTextureObject_t dt_odegree;
  cudaTextureObject_t dt_start_pos;

  cudaTextureObject_t dtr_odegree;
  cudaTextureObject_t dtr_start_pos;
};

template <GraphFmt Fmt, typename E> struct graph_t {
  edgelist_t<E> el;
  host_graph_t<Fmt, E> hg;
  device_graph_t<Fmt, E> dg;
};

template <Centric C, typename E> struct Fake {};
template <typename E> struct Fake<VC, E> {
  static graph_t<CSR, E> func(std::string path, feature_t &fets,
                              bool with_header = false,
                              bool with_weight = false, bool directed = false) {
    int64_t gpu_bytes = 0;
    graph_t<CSR, E> g;
    g.el.read_mtx(path, directed, with_weight, with_header);
    g.hg.build(g.el);
    gpu_bytes = g.dg.build(g.hg);
    fets.avg_deg = g.hg.attr.avg_deg;
    fets.std_deg = g.hg.attr.std_deg;
    fets.range = g.hg.attr.range;
    fets.GI = g.hg.attr.GI;
    fets.Her = g.hg.attr.Her;
    fets.nvertexs = g.el.nvertexs;
    fets.nedges = g.el.nedges;
    fets.avg_edata = g.el.mean_weight;
    LOG(" -- GPU Global Memory used for Graph: %f\n",
        (0.0 + gpu_bytes) / (1ll << 30));
    return g;
  }

  static graph_t<CSR, E> func2(edgelist_t<E> el, feature_t &fets) {
    graph_t<CSR, E> g;
    g.el = el;
    g.hg.build(g.el);
    int64_t gpu_bytes = g.dg.build(g.hg);
    fets.avg_deg = g.hg.attr.avg_deg;
    fets.std_deg = g.hg.attr.std_deg;
    fets.range = g.hg.attr.range;
    fets.GI = g.hg.attr.GI;
    fets.Her = g.hg.attr.Her;
    fets.nvertexs = g.el.nvertexs;
    fets.nedges = g.el.nedges;
    fets.avg_edata = g.el.mean_weight;
    LOG(" -- GPU Global Memory used for Graph: %f\n",
        (0.0 + gpu_bytes) / (1ll << 30));
    return g;
  }
};

template <typename E> struct Fake<EC, E> {
  static graph_t<COO, E> func(std::string path, feature_t &fets,
                              bool with_header = false,
                              bool with_weight = false, bool directed = false) {
    int64_t gpu_bytes = 0;
    graph_t<COO, E> g;
    g.el.read_mtx(path, directed, with_weight, with_header);
    g.hg.build(g.el);
    gpu_bytes = g.dg.build(g.hg);
    fets.avg_deg = g.hg.attr.avg_deg;
    fets.std_deg = g.hg.attr.std_deg;
    fets.range = g.hg.attr.range;
    fets.max_deg = g.hg.attr.max_deg;
    fets.GI = g.hg.attr.GI;
    fets.Her = g.hg.attr.Her;
    fets.nvertexs = g.el.nvertexs;
    fets.nedges = g.el.nedges;
    fets.avg_edata = g.el.mean_weight;
    LOG(" -- GPU Global Memory used for Graph: %f\n",
        (0.0 + gpu_bytes) / (1ll << 30));
    return g;
  }

  static graph_t<COO, E> func2(edgelist_t<E> el, feature_t &fets) {
    graph_t<COO, E> g;
    g.el = el;
    g.hg.build(g.el);
    int64_t gpu_bytes = g.dg.build(g.hg);
    fets.avg_deg = g.hg.attr.avg_deg;
    fets.std_deg = g.hg.attr.std_deg;
    fets.range = g.hg.attr.range;
    fets.max_deg = g.hg.attr.max_deg;
    fets.GI = g.hg.attr.GI;
    fets.Her = g.hg.attr.Her;
    fets.nvertexs = g.el.nvertexs;
    fets.nedges = g.el.nedges;
    fets.avg_edata = g.el.mean_weight;
    LOG(" -- GPU Global Memory used for Graph: %f\n",
        (0.0 + gpu_bytes) / (1ll << 30));
    return g;
  }
};

template <Centric C, typename E = Empty>
auto build_graph(std::string path, feature_t &fets, bool with_header = false,
                 bool with_weight = false, bool directed = false)
    -> decltype(Fake<C, E>::func(path, fets, with_header, with_weight,
                                 directed)) {
  return Fake<C, E>::func(path, fets, with_header, with_weight, directed);
}

template <Centric C, typename E = Empty>
auto build_graph(edgelist_t<E> el,
                 feature_t &fets) -> decltype(Fake<C, E>::func2(el, fets)) {
  return Fake<C, E>::func2(el, fets);
}

// base template
#endif
