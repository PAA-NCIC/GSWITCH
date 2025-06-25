#ifndef __FILEIO_H
#define __FILEIO_H

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include "utils/2d_partition.cuh"
#include "utils/utils.cuh"

struct graph_attr_t {
  double avg_deg = 0;
  double std_deg = 0;
  double max_deg = 0;
  double range = 0;
  double GI = 0;
  double Her = 0;

  void fill(int *deg, int nv) {
    std::vector<double> tmp;
    for (int i = 0; i < nv; ++i) {
      tmp.push_back(deg[i]);
    }

    std::sort(tmp.begin(), tmp.end());
    for (auto x : tmp) {
      avg_deg += x;
      std_deg += x * x;
    }
    std_deg = sqrt((std_deg - avg_deg * avg_deg / nv) / (nv - 1));
    avg_deg /= nv;
    range = (tmp[nv - 1] - tmp[0]); /// avg_deg;
    max_deg = tmp[nv - 1];

    double sum = 0;
    for (int i = 1; i <= nv; ++i) {
      sum += tmp[i - 1];
    }

    for (int i = 1; i <= nv; ++i) {
      double du = tmp[i - 1];
      GI += (nv - i + 0.5) * du / nv / sum;
      if (du != 0)
        Her += du * (log(sum) - log(du)) / (sum);
    }
    GI = 1 - 2 * GI;
    Her = Her / log(nv);

    // std::cout << avg_deg << " " << std_deg << " " << range << " " << GI << "
    // " << Her << std::endl;
  }
};

// normailze index start from 0
template <typename E> class edgelist_t {
public:
  void read_mtx(std::string path, bool directed = false,
                bool with_weight = false, bool with_header = false) {
    double start = wtime();
    LOG("Loading %s\n", path.c_str());

    this->directed = directed;
    this->with_weight = with_weight;
    std::ifstream fin(path);
    if (!fin.is_open())
      ASSERT(false, "can not open file");

    // skip comments
    while (1) {
      char c = fin.peek();
      if (c >= '0' && c <= '9')
        break;
      fin.ignore(std::numeric_limits<std::streamsize>::max(), fin.widen('\n'));
    }

    // if with header (meaningless, just ignore)
    if (with_header) {
      fin >> nvertexs >> nvertexs >> nedges;
      fin.ignore(std::numeric_limits<std::streamsize>::max(), fin.widen('\n'));
    }

    else
      nvertexs = nedges = -1;

    vvector.clear();
    evector.clear();

    vmin = std::numeric_limits<int>::max();
    vmax = -1;
    while (fin.good()) {
      int v0, v1;
      E w;
      fin >> v0 >> v1;
      if (fin.eof())
        break;

      vmin = std::min(v0, vmin);
      vmin = std::min(v1, vmin);
      vmax = std::max(v0, vmax);
      vmax = std::max(v1, vmax);

      if (with_weight)
        fin >> w;
      else {
        fin.ignore(std::numeric_limits<std::streamsize>::max(),
                   fin.widen('\n'));
      }

      if (fin.eof())
        break;

      if (v0 == v1)
        continue;
      vvector.push_back(v0);
      vvector.push_back(v1);

      if (with_weight)
        evector.push_back(w);
    }

    nvertexs = (vmax - vmin + 1);
    nedges = vvector.size() >> 1;
    if (nvertexs == 0 || nedges == 0)
      exit(1);

    fin.close();
    double end = wtime();
    LOG("IO time: %f s.\n", end - start);
    // reorder();
  }

  void gen_weight(int lim) {
    if (with_weight && evector.size() > 0) {
      int64_t tot = 0;
      for (size_t i = 0; i < evector.size(); ++i) {
        tot += evector[i];
      }
      mean_weight = (tot + .0) / evector.size();
    } else {
      with_weight = true;
      int64_t tot = 0;
      LOG(" -- generate edge weight 0~%d\n", lim);
      for (int i = 0; i < nedges; i++) {
        E w = (E)rand_device<int>::rand_weight(lim);
        tot += w;
        evector.push_back(w);
      }
      mean_weight = (tot + .0) / nedges;
    }
  }

  void reorder() {
    struct pair_t {
      int deg, id;
    };
    std::vector<pair_t> od(nvertexs, {0, 0});
    for (size_t i = 0; i < od.size(); ++i)
      od[i].id = i;
    for (size_t i = 0; i < vvector.size(); i += 2) {
      int v = vvector[i] - vmin;
      od[v].deg++;
    }
    std::sort(
        od.begin(), od.end(),
        [](const pair_t &a, const pair_t &b) -> bool { return a.deg > b.deg; });
    std::vector<int> mapper(nvertexs, 0);
    for (size_t i = 0; i < od.size(); ++i)
      mapper[od[i].id] = i;

    for (size_t i = 0; i < vvector.size(); i += 2) {
      int v = vvector[i] - vmin;
      int u = vvector[i + 1] - vmin;
      vvector[i] = mapper[v] + vmin;
      vvector[i + 1] = mapper[u] + vmin;
    }
  }

public:
  std::vector<int> vvector;
  std::vector<E> evector;
  int vmin, vmax;
  int nvertexs;
  int nedges;
  bool directed;
  bool with_weight;
  double mean_weight = 0;
};

template <GraphFmt format, typename E> class host_graph_t {};

template <typename E> class host_graph_t<CSR, E> {

public:
  void build(edgelist_t<E> el, bool quiet = false) {

    double start = wtime();
    this->directed = el.directed;
    if (std::is_same<Empty, E>::value) {
      this->with_weight = false;
    } else {
      this->with_weight = true;
    }

    int64_t vmax = el.vmax;
    int64_t vmin = el.vmin;

    int64_t mem_used = 0;
    nvertexs = (vmax - vmin + 1);
    nedges = el.vvector.size() >> 1;

    if (!directed)
      nedges <<= 1;

    odegrees = (int *)calloc(nvertexs, sizeof(int));
    start_pos = (int *)malloc(sizeof(int) * nvertexs);
    mem_used += sizeof(int) * nvertexs * 2;

    if (directed) {
      r_odegrees = (int *)calloc(nvertexs, sizeof(int));
      r_start_pos = (int *)malloc(sizeof(int) * nvertexs);
      mem_used += sizeof(int) * nvertexs * 2;
    }

    if (with_weight) {
      edge_weights = (E *)malloc(sizeof(E) * nedges);
      mem_used += sizeof(E) * nedges;
      if (directed) {
        r_edge_weights = (E *)malloc(sizeof(E) * nedges);
        mem_used += sizeof(E) * nedges;
      }
    } else {
      edge_weights = NULL;
      r_edge_weights = NULL;
    }

    for (size_t i = 0; i < el.vvector.size(); i += 2) {
      odegrees[el.vvector[i] - vmin]++;
      if (!directed)
        odegrees[el.vvector[i + 1] - vmin]++;
      else
        r_odegrees[el.vvector[i + 1] - vmin]++;
    }

    start_pos[0] = 0;
    for (int i = 1; i < nvertexs; ++i) {
      start_pos[i] = odegrees[i - 1] + start_pos[i - 1];
      odegrees[i - 1] = 0;
    }
    odegrees[nvertexs - 1] = 0;

    adj_list = (int *)malloc(sizeof(int) * nedges);
    mem_used += sizeof(int) * nedges;

    if (directed) {
      r_start_pos[0] = 0;
      for (int i = 1; i < nvertexs; ++i) {
        r_start_pos[i] = r_odegrees[i - 1] + r_start_pos[i - 1];
        r_odegrees[i - 1] = 0;
      }
      r_odegrees[nvertexs - 1] = 0;

      r_adj_list = (int *)malloc(sizeof(int) * nedges);
      mem_used += sizeof(int) * nedges;
    }

    for (size_t i = 0; i < (el.vvector.size() >> 1); i++) {
      int v0 = el.vvector[i << 1] - vmin;
      int v1 = el.vvector[i << 1 | 1] - vmin;

      adj_list[start_pos[v0] + odegrees[v0]] = v1;
      if (with_weight)
        edge_weights[start_pos[v0] + odegrees[v0]] = el.evector[i];
      odegrees[v0]++;

      // double it
      if (!directed) {
        adj_list[start_pos[v1] + odegrees[v1]] = v0;
        if (with_weight)
          edge_weights[start_pos[v1] + odegrees[v1]] = el.evector[i];
        odegrees[v1]++;
      } else {
        r_adj_list[r_start_pos[v1] + r_odegrees[v1]] = v0;
        if (with_weight)
          r_edge_weights[r_start_pos[v1] + r_odegrees[v1]] = el.evector[i];
        r_odegrees[v1]++;
      }
    }

    if (!quiet && !with_weight) {
      rinse();
    }

    // CSR_pretty_print();

    attr.fill(odegrees, nvertexs);

    double end = wtime();

    if (quiet)
      return;
    if (cmd_opt.verbose) {
      std::cout << "CSR transform time used: " << end - start << " s."
                << std::endl;
      std::cout << " -- nvertexs: " << nvertexs << " nedges: " << nedges
                << std::endl;
      std::cout << " -- degree range: " << attr.range << "; avg. degree "
                << attr.avg_deg << "; max degree " << attr.max_deg << std::endl;
      std::cout << " -- isdirected: " << (directed ? "Yes" : "No") << std::endl;
      std::cout << " -- isweighted: " << (with_weight ? "Yes" : "No")
                << std::endl;
      std::cout << "Host Graph memory used: "
                << (0.0 + mem_used + 2 * sizeof(int) + 2 * sizeof(bool)) /
                       (1l << 30)
                << " GB." << std::endl;
    }
    if (ENABLE_2D_PARTITION) {
      chunks.build(nvertexs, adj_list, start_pos, odegrees);
    }
  }

  void store_to_binfile(std::string fname) {
    std::string newf = ".csr_" + std::string(get_fname(fname.c_str()));
    std::string dir = std::string(get_dirname(fname.c_str()));
    std::string fpath = dir + "/" + newf;
    if (fexist(fpath.c_str()))
      return;
    std::ofstream fout(fpath, std::ios::binary);
    fout.write((char *)&nvertexs, sizeof(nvertexs));
    fout.write((char *)&nedges, sizeof(nvertexs));
    fout.write((char *)&directed, sizeof(directed));
    fout.write((char *)&with_weight, sizeof(with_weight));

    fout.write((char *)odegrees, sizeof(int) * nvertexs);
    fout.write((char *)start_pos, sizeof(int) * nvertexs);
    fout.write((char *)adj_list, sizeof(int) * nedges);
    if (with_weight)
      fout.write((char *)edge_weights, sizeof(E) * nedges);

    if (directed) {
      fout.write((char *)r_odegrees, sizeof(int) * nvertexs);
      fout.write((char *)r_start_pos, sizeof(int) * nvertexs);
      fout.write((char *)r_adj_list, sizeof(int) * nedges);
      if (with_weight)
        fout.write((char *)r_edge_weights, sizeof(E) * nedges);
    }
  }

  void load_from_binfile(std::string fname) {
    std::string newf = ".csr_" + std::string(get_fname(fname.c_str()));
    std::string dir = std::string(get_dirname(fname.c_str()));
    std::string fpath = dir + "/" + newf;
    if (!file_exist(fpath.c_str()))
      exit(1);
    std::ifstream fin(fpath, std::ios::binary);

    fin.read((char *)&nvertexs, sizeof(nvertexs));
    fin.read((char *)&nedges, sizeof(nvertexs));
    fin.read((char *)&directed, sizeof(directed));
    fin.read((char *)&with_weight, sizeof(with_weight));

    fin.read((char *)odegrees, sizeof(int) * nvertexs);
    fin.read((char *)start_pos, sizeof(int) * nvertexs);
    fin.read((char *)adj_list, sizeof(int) * nedges);
    if (with_weight)
      fin.read((char *)edge_weights, sizeof(E) * nedges);

    if (directed) {
      fin.read((char *)r_odegrees, sizeof(int) * nvertexs);
      fin.read((char *)r_start_pos, sizeof(int) * nvertexs);
      fin.read((char *)r_adj_list, sizeof(int) * nedges);
      if (with_weight)
        fin.read((char *)r_edge_weights, sizeof(E) * nedges);
    }
  }

  int random_root() {
    int root;
    while (1) {
      root = random() % nvertexs;
      if (odegrees[root])
        break;
    }
    return root;
  }

  void dedup_edges() {
    auto dedup_fn = [this](int &idx, int v, int *_adj_list, int *_start_pos,
                           int *_odegrees) {
      int s = _start_pos[v];
      _start_pos[v] = idx;
      int e = (v == (this->nvertexs - 1) ? this->nedges : _start_pos[v + 1]);
      std::sort(_adj_list + s, _adj_list + e);
      for (int j = s; j < e; ++j) {
        int u = _adj_list[j];
        // skip duplicated edges
        if (j > s && u == _adj_list[j - 1]) {
          continue;
        }
        _adj_list[idx++] = u;
      }
      _odegrees[v] = idx - _start_pos[v];
    };
    int t = 0, tr = 0;
    for (int v = 0; v < nvertexs; ++v) {
      dedup_fn(t, v, adj_list, start_pos, odegrees);
      if (this->directed) {
        dedup_fn(tr, v, r_adj_list, r_start_pos, r_odegrees);
      }
    }
    LOG(" -- remove duplicate edges: %d -> %d\n", nedges, t);
    if (this->directed && t != tr) {
      printf("Inconsistent nedges in directed graphs.");
    }
    this->nedges = t;
  }

  void remove_solo_vertices() {
    std::vector<int> rename;
    int top = 0;
    for (int i = 0; i < nvertexs; ++i) {
      bool solo = (odegrees[i] == 0);
      bool r_solo = !this->directed || (r_odegrees[i] == 0);
      if (solo && r_solo) {
        rename.push_back(-1);
      } else {
        rename.push_back(top++);
      }
    }

    auto remove_solo_fn = [this, rename](int &idx, int v, int *_adj_list,
                                         int *_start_pos, int *_odegrees) {
      if (rename[v] == -1) {
        return;
      }
      _start_pos[idx] = _start_pos[v];
      _odegrees[idx] = _odegrees[v];
      int s = _start_pos[v];
      int e = (v == (this->nvertexs - 1) ? this->nedges : _start_pos[v + 1]);
      for (int j = s; j < e; ++j) {
        _adj_list[j] = rename[_adj_list[j]];
      }
      idx++;
    };

    int t = 0, tr = 0;
    for (int v = 0; v < nvertexs; ++v) {
      remove_solo_fn(t, v, adj_list, start_pos, odegrees);
      if (this->directed) {
        remove_solo_fn(tr, v, r_adj_list, r_start_pos, r_odegrees);
      }
    }

    if (top != t || (this->directed && t != tr)) {
      printf("Error in file CSR transform top=%d, t=%d, tr=%d.\n", top, t, tr);
    }
    LOG(" -- remove solo vertices: %d -> %d\n", nvertexs, top);
    nvertexs = top;
  }

  void rinse() {
    // with_weight
    if (with_weight) {
      printf("skip edgelist rinsing for weighted graph.");
      return;
    }
    dedup_edges();
    remove_solo_vertices();
  }

  void CSR_pretty_print() {
    printf("--------- DEBUG ----------\n");
    printf("nvertex = %d, nedges = %d\n", this->nvertexs, this->nedges);
    auto pretty_print_fn = [](int v, int *_adj_list, int *_start_pos,
                              int *_odegrees) {
      printf("d(%d)=%d, N(%d) = [ ", v, _odegrees[v], v);
      for (int j = _start_pos[v]; j < _start_pos[v] + _odegrees[v]; ++j) {
        int u = _adj_list[j];
        char sepchar = (j == _start_pos[v] + _odegrees[v] - 1) ? ' ' : ',';
        printf("%d%c", u, sepchar);
      }
      printf("]\n");
    };

    printf("--------- adj_list ----------\n");
    for (int v = 0; v < this->nvertexs; v++) {
      pretty_print_fn(v, adj_list, start_pos, odegrees);
    }
    if (this->directed) {
      printf("-------- r_adj_list ---------\n");
      for (int v = 0; v < this->nvertexs; v++) {
        pretty_print_fn(v, r_adj_list, r_start_pos, r_odegrees);
      }
    }
  }

public:
  E *edge_weights = NULL;
  int *adj_list = NULL;
  int *start_pos = NULL;
  int *odegrees = NULL;

  // for reversed graph
  E *r_edge_weights = NULL;
  int *r_adj_list = NULL;
  int *r_start_pos = NULL;
  int *r_odegrees = NULL;

  int nvertexs;
  int nedges;
  bool directed;
  bool with_weight;

  chunk_t chunks;

  graph_attr_t attr;

  // double mean_odegree;
  // int max_odegree;
  // int min_odegree;
};

template <typename E> class host_graph_t<COO, E> {
public:
  void build(edgelist_t<E> el, bool quiet = false) {
    double start = wtime();
    this->directed = el.directed;
    this->with_weight = el.with_weight;

    int64_t vmax = el.vmax;
    int64_t vmin = el.vmin;

    int64_t mem_used = 0;
    nvertexs = (vmax - vmin + 1);
    nedges = el.vvector.size() >> 1;

    edge_list = (packed_t *)malloc(sizeof(packed_t) * nedges);
    if (with_weight) {
      edge_weights = (E *)malloc(sizeof(E) * nedges);
      mem_used += sizeof(E) * nedges;
    }
    mem_used += sizeof(packed_t) * nedges;

    for (int i = 0; i < nedges; i++) {
      int v0 = el.vvector[i << 1] - vmin;
      int v1 = el.vvector[i << 1 | 1] - vmin;
      packed_t v = ((int64_t)v0) << 32 | v1;
      edge_list[i] = v;
      if (with_weight)
        edge_weights[i] = el.evector[i];
    }
    double end = wtime();

    if (quiet)
      return;
    if (cmd_opt.verbose) {
      std::cout << "COO transform time used: " << end - start << " s."
                << std::endl;
      std::cout << " -- nvertexs: " << nvertexs << " nedges: " << nedges
                << std::endl;
      std::cout << " -- isdirected: " << (directed ? "Yes" : "No") << std::endl;
      std::cout << " -- isweighted: " << (with_weight ? "Yes" : "No")
                << std::endl;
      std::cout << "Host Graph memory used: "
                << (0.0 + mem_used + 2 * sizeof(int) + 2 * sizeof(bool)) /
                       (1l << 30)
                << " GB." << std::endl;
    }
  }

public:
  E *edge_weights;
  packed_t *edge_list;
  int nvertexs;
  int nedges;
  bool directed;
  bool with_weight;
  // int mean_odegree = 1;

  graph_attr_t attr;
};

#endif
