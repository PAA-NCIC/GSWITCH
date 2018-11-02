<p align="center">
  <a href="https://github.com/jokopi/GSWITCH"><img src="./docs/assets/imgs/gswitch-icon.png"></a>
  <br>
  <a href="https://github.com/jokopi/GSWITCH/releases/tag/v0.1"><img src="https://img.shields.io/badge/gswitch-0.1-blue.svg"></a>
  <br>
</p>

# GSWITCH
GSWITCH is a pattern-based algorithmic autotuning system that dynamically switched to the suitable optimization variants with negligible overhead.
Specifically, It is a CUDA library targeting the GPU-based graph processing application, it supports both vertex-centric or edge-centric abstractions. 
By far, GSWITCH can automatically determine the suitable optimization variants in Direction (push, pull), data-structure (Bitmap, Sorted Queue, Unsorted Queue), Load-Balance (TWC, WM, CM, STRICT, 2D-partition), Stepping (Increase, Decrease, Remain), Kernel Fusion(Standalone, Fused).
The fast optimization transition of GSWITCH is based on a machine learning model trained from 600+ real graphs from the [network repository](http://networkrepository.com). 
The model can be reused by news applications, or be retrained to adapt new architectures.
In addition, GSWITCH provides succinct programming interface which hides all low-level tuning details. Developers can implements their graph application with high performance in just ~100 lines of code.

For more details, please visit our [website](https://jokopi.github.io/GSWITCH/).

## Dependency

 - nvcc 7.5+
 - cmake
 - moderngpu

## Quickstart

```shell
mkdir build
cd build
cmake ..
make
```

## Usage

Here are the basic useages of pre-integrated applications(BFS,CC,PR,SSSP,BC) in GSWITCH.

```shell
./EXE <graph_path> [options]
[-r, --src=<int>]         Choose a root vertex. (Default: ch
                          oose randomly).
[-v, --verbose]           Print verbose per iteration info. 
                          (Default: quiet mode)
[-V, --validation]        Process the CPU reference validati
                          on. (Defaule: no validation)
[-H, --with-header]       Input file has header (e.g. nverte
                          xs, nvertexs, nedges, Default: no 
                          header).
[-W, --with-weight]       Input file has weight.(Default: no
                           weight value)
[-i, --ignore-weight]     Ignore the graph weight.(Default: 
                          false)
[-d, --directed]          Graph is directed.(Default: undire
                          cted)
[-c, --configs=Push-Queue-CM-Fused]
                          Set debug strategies, use - to sep
                          arater them (Default: <empty>).
[-D, --device=<int>]      Choose GPU for testing (Default: 0
                          )
[-j, --json=<string>]     set the json path (Default: 0)
```

## Example

Here is a sample codes of BFS for graph soc-orkut. For more details please visit `./application/bfs.cu`

```c++
#include "gswitch.h"
using G = device_graph_t<CSR, Empty>;

struct BFS:Functor<VC,int,Empty,Empty>{
  __device__ Status filter(int vid, G g){
    int lvl = *wa_of(vid);
    if(lvl == g.get_level()) return Active;
    else if (lvl < 0) return Inactive;
    else return Fixed;
  }
  __device__ int emit(int vid, Empty *w, G g) {return g.get_level();}
  __device__ bool cond(int v, int newv, G g) {return *wa_of(v)==-1;}
  __device__ bool comp(int* v, int newv, G g) {*v=newv; return true;}
  __device__ bool compAtomic(int* v, int newv, G g) {*v=newv; return true;}
};

int main(){
  // load graph
  for(level=0;;level++){
    inspector.inspect(as, g, f, stats, fets, conf);
    if(as.finish(g, f, conf)) break;
    selector.select(stats, fets, conf);
    executor.filter(as, g, f, stats, fets, conf);
    g.update_level();
    executor.expand(as, g, f, stats, fets, conf);
  }
  // copy data back
}
```
run the with `./BFS soc-orkut.mtx --with-header --src=0 --device=0 --verbose`:

![run-bfs](./docs/assets/imgs/run-bfs-orkut-example.png)


## Performance
please visit our [website](https://jokopi.github.io/GSWITCH/)

## License
All the libraryies, examples, and source codes of GSWITCH are released under [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0).
