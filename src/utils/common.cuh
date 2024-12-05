#ifndef __COMMON_H
#define __COMMON_H

#include <cstdio>

#define ALL_ACTIVE -1
#define ALL_INACTIVE -2

#define ENABLE_2D_PARTITION false

// const data
const int BIN_SZ = 512;
const int Q_NUM = 3;
const int CTANUM = 256;
const int THDNUM = 512;
const int CTANUM_EXPAND = 256;
const int THDNUM_EXPAND = 512;
// const int CTANUM_EXPAND = 512;
// const int THDNUM_EXPAND = 1024;
const int MAXBIN = CTANUM * THDNUM * BIN_SZ;

int SBSIZE = 0; // quiet
const int BSZ = (THDNUM_EXPAND * 6);
const int THRESHOLD = (6144 * 4);
// const int THRESHOLD = (126144*4);

enum Status { Inactive = 0, Active = 1, All = 2, Fixed = 3 }; // 00, 01, 10, 11
enum GraphFmt { CSR, COO };
enum Centric { EC, VC };
enum Computation { Idem, Mono, ASSO };
enum QueueMode { Normal, Cached };

enum Direction { Push, Pull };
enum ASFmt { Queue, Bitmap };
enum LB { TM, WM, CM, STRICT, TWOD, ELB, TWC };
enum Fusion { Fused, Standalone };
enum CullMode { Interleave, Stride };

#endif
