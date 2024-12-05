#ifndef __2D_PARTITION_CUH
#define __2D_PARTITION_CUH

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "utils/utils.cuh"

struct chunk_t {
  void build(int nvertexs, int *adj_list, int *start_pos, int *odegrees) {
    LOG("step1: meta data\n");
    csize = CEIL(nvertexs, BSZ);
    nchunks = csize * csize;
    LOG(" -- Construct Rows=%d, Cols=%d Maxtrix\n", csize, csize);

    LOG("step2: group edges by chunks id\n");
    struct tup_t {
      int id, v, u;
    };
    std::vector<tup_t> vvector;
    for (int i = 0; i < nvertexs; ++i) {
      for (int j = 0; j < odegrees[i]; ++j) {
        int v = i;
        int u = adj_list[start_pos[v] + j];
        int id_row = v / BSZ;
        int id_col = u / BSZ;
        vvector.push_back(tup_t{id_row * csize + id_col, v, u});
      }
    }
    std::sort(
        vvector.begin(), vvector.end(),
        [](const tup_t &a, const tup_t &b) -> bool { return a.id < b.id; });

    LOG("step3: compute chunk meta info\n");
    std::vector<int> cnt(nchunks + 1, 0);
    std::vector<int> ant(nchunks, 0);
    std::vector<int> heavys;
    std::vector<int> lights;
    for (const tup_t &it : vvector) {
      cnt[it.id + 1]++;
    }
    for (size_t i = 1; i < cnt.size(); ++i) {
      cnt[i] += cnt[i - 1];
    }
    for (int i = 0; i < nchunks; ++i) {
      if (cnt[i + 1] - cnt[i] > THRESHOLD)
        heavys.push_back(i);
      else
        lights.push_back(i);
    }
    for (int i = 0; i < nchunks; ++i)
      ant[i] = ALIGN((cnt[i + 1] - cnt[i]), 8);

    dcnt = heavys.size();
    scnt = csize; // row panel;
    LOG(" -- we have %d heavy blocks and %d light block panel\n", dcnt, scnt);
    hid = (int *)malloc(dcnt * sizeof(int));
    for (int i = 0; i < dcnt; ++i)
      hid[i] = heavys[i] / csize;

    LOG("step4: build heavy blocks\n");
    std::vector<int> segsize(dcnt, 0);
    segpos = (int *)malloc((dcnt + 1) * sizeof(int));
    dpf = (int *)malloc(dcnt * sizeof(int));
    int top = 0;
    dsize = 0;
    for (int i = 0; i < nchunks; ++i) {
      if (cnt[i + 1] - cnt[i] > THRESHOLD) {
        std::sort(&vvector[cnt[i]], &vvector[cnt[i + 1]],
                  [](const tup_t &a, const tup_t &b) -> bool {
                    if (a.u == b.u)
                      return a.v < b.v;
                    else
                      return a.u < b.u;
                  });
        segsize[top] = 1 + CEIL(cnt[i + 1] - cnt[i] + BSZ, THDNUM_EXPAND);
        segpos[top] = dsize;
        dsize += segsize[top] * THDNUM_EXPAND;
        top++;
      }
    }
    segpos[top] = dsize;
    LOG(" -- we have dsize = %d padded heavy edges (%d==%d?)\n", dsize, top,
        dcnt);

#define NID(t, sz) ((t % sz) * THDNUM_EXPAND + (t / sz))
    segs = (int *)malloc(dsize * sizeof(int));
    memset(segs, -1, sizeof(int) * dsize);
    for (int i = 0; i < dcnt; ++i) {
      int chunk_id = heavys[i];
      int top = 0;
      int cur = -1;
      int base = segpos[i];
      int sz = segsize[i];
      // std::cout << base << " " << sz << std::endl;
      // std::cout << chunk_id << " " << cnt[chunk_id] << " " << cnt[chunk_id+1]
      // << std::endl; std::sort(&vvector[cnt[chunk_id]],
      // &vvector[cnt[chunk_id+1]], [](const tup_t& a, const tup_t&
      // b)->bool{return a.u<b.u;});
      for (int j = cnt[chunk_id]; j < cnt[chunk_id + 1]; ++j) {
        int v = vvector[j].v; // src
        int u = vvector[j].u; // dst
        if (cur != u) {
          segs[base + NID(top, sz)] = HASH(u % BSZ);
          top++;
          cur = u;
        }
        if (top % sz == 0) {
          segs[base + NID(top, sz)] = HASH(cur % BSZ);
          top++;
        }
        segs[base + NID(top, sz)] = v % BSZ;
        top++;
      }
      if (top > sz * THDNUM_EXPAND) {
        LOG(" -- fill error, exit %d %d %d\n", top, sz * THDNUM_EXPAND,
            cnt[i + 1] - cnt[i]);
        exit(1);
      }
    }
#undef NID
    int validcnt = 0;
    for (int i = 0; i < dsize; ++i) {
      if (segs[i] >= 0)
        validcnt++;
    }
    LOG(" -- valid vertices: %d\n", validcnt);
    double usedbytes_heavy = (dsize + dcnt + 1.0) * sizeof(int);
    LOG(" -- Host Graph Heavy blocks used: %.3f Gb.\n",
        (0.0 + usedbytes_heavy) / (1l << 30));

    LOG("step5: build light block panels\n");
    // compute ces
    ces = (int *)malloc((scnt + 1) * sizeof(int));
    memset(ces, 0, sizeof(int) * (scnt + 1));
    for (size_t i = 0; i < lights.size(); ++i) {
      int chunk_id = lights[i];
      // std::sort(&vvector[cnt[chunk_id]], &vvector[cnt[chunk_id+1]], [](const
      // tup_t& a, const tup_t& b)->bool{if(a.v==b.v) return a.u<b.u; else
      // return a.v<b.v;});
      int panel_id = chunk_id / csize; // row panel
      // ces[panel_id+1] += cnt[chunk_id+1]-cnt[chunk_id];
      ces[panel_id + 1] += ant[chunk_id];
    }
    for (int i = 1; i < scnt + 1; ++i) {
      ces[i] += ces[i - 1];
    }
    ssize = ces[scnt];
    LOG(" -- we have ssize = %d light edges\n", ssize);

    spf = (int *)malloc((CEIL(ssize, 8)) * sizeof(int));
    // spf = (int*)malloc(ssize*sizeof(int));
    sps = (short *)malloc(ssize * sizeof(short));
    memset(sps, -1, sizeof(short) * ssize);
    top = 0;
    for (size_t i = 0; i < lights.size(); ++i) {
      int chunk_id = lights[i];
      for (int j = cnt[chunk_id]; j < cnt[chunk_id + 1]; ++j) {
        int v = vvector[j].v;
        int u = vvector[j].u;
        sps[top++] = (short)(v % BSZ);
      }
      top += (int)ant[chunk_id] - cnt[chunk_id + 1] + cnt[chunk_id];
    }
    if (top != ssize) {
      LOG(" -- fill sps error\n");
      exit(1);
    }

    LOG("step6: build stream buffer\n");
    std::vector<int> pc(nchunks, 0);
    // for(int i=0; i<nchunks; ++i) pc[i] = (cnt[i+1]-cnt[i] >THRESHOLD)?
    // (BSZ):(cnt[i+1]-cnt[i]);
    for (int i = 0; i < nchunks; ++i)
      pc[i] = (cnt[i + 1] - cnt[i] > THRESHOLD) ? (BSZ) : (ant[i]);
    sbsize = 0;
    for (int i = 0; i < nchunks; ++i)
      sbsize += pc[i];
    LOG(" -- we have %d partial contribution to stream\n", sbsize);
    SBSIZE = sbsize;

    cey = (int *)malloc((csize + 1) * sizeof(int));
    memset(cey, 0, sizeof(int) * (csize + 1));
    for (int i = 0; i < nchunks; ++i) {
      int panel_id = i % csize; // col panel
      cey[panel_id + 1] += pc[i];
    }
    for (int i = 1; i < csize + 1; ++i)
      cey[i] += cey[i - 1];
    if (cey[csize] != sbsize)
      LOG(" -- cey error!\n");

    // sb  = (float*)malloc(sbsize*sizeof(float));
    spt = (short *)malloc(sbsize * sizeof(short));
    memset(spt, -1, sizeof(short) * sbsize);
#define CO(i, sz) ((i % sz) * sz + (i / sz))
    std::vector<int> pc_col_acc(nchunks + 1, 0);
    for (int i = 0; i < nchunks; ++i)
      pc_col_acc[i + 1] = pc[CO(i, csize)];
    for (int i = 1; i < nchunks + 1; ++i)
      pc_col_acc[i] += pc_col_acc[i - 1];

    // spt
    for (int i = 0; i < nchunks; ++i) {
      int base = pc_col_acc[CO(i, csize)];
      if (cnt[i + 1] - cnt[i] > THRESHOLD) { // heavy block
        for (int j = 0; j < BSZ; ++j)
          spt[base + j] = j;
      } else { // light block pc[i] = cnt[i+1] - cnt[i]
        int _base = cnt[i];
        for (int j = 0; j < cnt[i + 1] - cnt[i]; ++j) {
          int u = vvector[_base + j].u;
          spt[base + j] = (short)(u % BSZ);
          // spt[base+j] = u;
        }
      }
    }

    // spf
    int sbase = 0;
    for (size_t i = 0; i < lights.size(); ++i) {
      int chunk_id = lights[i];
      // int edge_size = cnt[chunk_id+1] - cnt[chunk_id];
      int edge_size = ant[chunk_id];
      int _base = pc_col_acc[CO(chunk_id, csize)];
      for (int j = 0; j < edge_size; ++j) {
        if ((j & 7) == 0)
          spf[(sbase + j) >> 3] = _base + j;
        // spf[sbase+j] = _base+j;
      }
      sbase += edge_size;
    }
    if (sbase != ssize)
      LOG(" -- spf error !\n");

    // dpf
    for (size_t i = 0; i < heavys.size(); ++i) {
      int chunk_id = heavys[i];
      int _base = pc_col_acc[CO(chunk_id, csize)];
      dpf[i] = _base;
    }
#undef CO
  }

  void todevice(chunk_t &c) {
    nchunks = c.nchunks;
    csize = c.csize;

    dcnt = c.dcnt;
    dsize = c.dsize;
    TODEVICE(segs, dsize, c.segs);
    TODEVICE(segpos, dcnt + 1, c.segpos);
    TODEVICE(dpf, dcnt, c.dpf);
    TODEVICE(hid, dcnt, c.hid);

    scnt = c.scnt;
    ssize = c.ssize;
    TODEVICE(ces, scnt + 1, c.ces);
    TODEVICE(spf, (CEIL(ssize, 8)), c.spf);
    // TODEVICE(spf, ssize, c.spf);
    TODEVICE_SHORT(sps, ssize, c.sps);

    sbsize = c.sbsize;
    // TODEVICE_FLOAT(sb, sbsize, c.sb);
    TODEVICE_SHORT(spt, sbsize, c.spt);
    TODEVICE(cey, csize + 1, c.cey);

    double gpu_bytes =
        (dsize + 2 * dcnt + scnt + 2 * ssize + 2 * sbsize + csize + 3) *
        sizeof(int);
    LOG(" -- Mutilgraph Dense Format consumes %.3f Gb GPU global memory.\n",
        (gpu_bytes / (1ll << 30)));
  }
  // gernal:
  int nchunks; // number of chunks
  int csize;   // nchunks = csize*csize;
               // heavy blocks:
  int dcnt;
  int dsize;   // heavy edges size
  int *segs;   // dsize
  int *hid;    // dcnt
  int *dpf;    // dcnt
  int *segpos; // dcnt+1
               // light block panel (row panel):
  int scnt;    // = csize;
  int ssize;   // light edges size
  int *ces;    // scnt+1;
  int *spf;    // ssize
  short *sps;  // ssize [0,BSZ]
               // stream buffer:
  int sbsize;  // get by pc matrix
  // float* sb;     // sbsize; // auto-reset
  short *spt; // sbsize;  [0,BSZ]
  int *cey;   // csize+1; (col panel)
};

#endif
