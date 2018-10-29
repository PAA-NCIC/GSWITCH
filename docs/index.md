---
layout: homepage
---

# GSWITCH
GSWITCH is a pattern-based algorithmic autotuning system that dynamically switched to the suitable optimization variants with negligible overhead.
Specifically, It is a CUDA library targeting the GPU-based graph processing application, it supports both vertex-centric or edge-centric abstractions.
By far, GSWITCH can automatically determine the suitable optimization variants in Direction (push, pull), data-structure (Bitmap, Queue), Load-Balance (TWC, WM, CM, STRICT, 2D-partition), Stepping (Increase, Decrease, Remain), Kernel Fusion (Standalone, Fused).
The fast optimization transition of GSWITCH is based on a machine learning model trained from 600+ real graphs from the [network repository](http://networkrepository.com).
The model can be resued by new applications, or be retrained to adapt to new architectures.
In addition, GSWITCH provides succinct programming interface which hides all low-level tuning details. Developers can implements their graph applications with high performance in just ~100 lines of code.

## Dependency

 - nvcc 7.5+
 - cmake
 - moderngpu

## Quick Start

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
[-r, --src=<int>]      Choose a root vertex. (Default: ch
                       oose randomly).
[-v, --verbose]        Print verbose per iteration info. 
                       (Default: quiet mode)
[-V, --validation]     Process the CPU reference validati
                       on. (Defaule: no validation)
[-H, --with-header]    Input file has header (e.g. nverte
                       xs, nvertexs, nedges, Default: no 
                       header).
[-W, --with-weight]    Input file has weight.(Default: no
                        weight value)
[-i, --ignore-weight]  Ignore the graph weight.(Default: 
                       false)
[-d, --directed]       Graph is directed.(Default: undire
                       cted)
[-c, --configs=Push-Queue-CM-Fused]
                       Set debug strategies, use - to sep
                       arater them (Default: <empty>).
[-D, --device=<int>]   Choose GPU for testing (Default: 0
                       )
[-j, --json=<string>]  set the json path (Default: 0)
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

![run-bfs](./assets/imgs/run-bfs-orkut-example.png)
*Figure 1: BFS example*

## Why GSWTICH

As GPUs provide higher parallelism and memory bandwidth than traditional CPUs, GPUs become a promising hardware to accelerate graph algorithms. Many recent works have explored the potential of using GPUs for data-intensive graph processing. Although the primary optimizations of these works are diverse, we notice that most of them are trying to find a 'one size fits all' solution. This leads to the mismatch and complication issues:

 **Mismatch**: Previous GPU-based graph frameworks may incur performance hits due to suboptimal strategies. Previous works accelerated graph primitives to run truly fast on some particular graphs or algorithms, however, their performance might fell dramatically when facing an unmatched situation. For example, Figure 2 shows that differents graph require different load-balance strategies. Figure 3 shows the performance loss if we only use push model in frontier expansion.

![LB](./assets/imgs/motivation_LB.png)
*Figure 2: Best load-balance for different graph*

![loss](./assets/imgs/motivation_loss.png)
*Figure 3: performance loss*

 **Complication**: Priori knowledge is required for users to make favorable decisions, especially from a mass of choices. A bulk synchronous parallel (BSP)-style graph application achieves its best performance only if correct strategies are chosen in every super-step. Unfortunately, the number of these performance-crucial strategies is very large, or worse yet, various combinations of these strategies form a huge tuning space. Data analysts should not spend their labor on wrestling with the tedious and complex performance tuning. Offloading the decision-making to a fully auto-tuning runtime could be a better choice.

## APIs

<table>
  <thead>
    <tr>
      <th><strong>APIs</strong></th>
      <th><strong>Description</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="52%"><code class="highlighter-rouge">filter(int vidx, G g)</code></td>
      <td>required, stream all the vertices(or edges) and filter out active ones; then update their value.</td>
    </tr>
    <tr>
      <td><code class="highlighter-rouge">WA emit(int vidx, E* e, G g)</code></td>
      <td>required, describe the message from one vertex to another.</td>
    </tr>
    <tr>
      <td><code class="highlighter-rouge">comp(WA* vdata, WA msg, G g)</code></td>
      <td>required, describe how the message is processed in the target vertex.</td>
    </tr>
    <tr>
      <td><code class="highlighter-rouge">compAtomic(WA* vdata, WA msg, G g)</code></td>
      <td>ditto, but an atomic version.</td>
    </tr>
    <tr>
      <td><code class="highlighter-rouge">cond(int vidx, E* e, G g)</code></td>
      <td>optional, help to omit useless updates.</td>
    </tr>
    <tr>
      <td><code class="highlighter-rouge">exit(int vidx, E* e, G g)</code></td>
      <td>optional, customed exit condition.</td>
    </tr>
  </tbody>
</table>

## Applications

Here we describe five typical graph applications in our framework: the Breadth-first search algorithm (BFS), the connected components algorithm (CC), the PageRank algorithm (PR), the single source shortest path algorithm (SSSP), and the betweenness centrality algorithm (BC). These five benchmarks can cover the majority of real-world graph applications.

**Breadth-First Search**. A BFS algorithm generates the breadth-first search tree of a graph from a source vertex *root* and compute the shortest jump hops from the *root* to all vertices reachable from it. Both of each vertex and each edge will be processed at most once. In BFS, any race condition between edges visiting a vertex is benign, thus we can perform pruning when we find a vertex that has been touched in the current iteration. 

In our framework, The BFS algorithm is implemented in vertex-centric abstraction. We use a `filter` function to mark a vertex whose *label* equals to the BFS depth as active and `emit` the *label+1* to the inactive vertices. Because the update process of BFS is idempotent, both of the `comp` and `compAtomic` function can do the update without atomic operations.

**Connected Components**. For undirected graphs, we say a group of vertices as a connected component when all vertices in the group can reach one another. Soman shows that the edge-centric implementation may be a better choice. In the edge-centric mode, each edge of the graph tries to assign its two end vertices with the same ID. A pointer-jumping is used to reduce the multi-level tree into star-like, which ensures that once the two end vertices of an edge have the same ID, they will remain the same in the following iteration. By repeating these two steps until no vertex changes, the algorithm terminates and outputs the result.

We implement the CC algorithm also as edge-centric in GSWITCH. At the beginning, all the edges are active, and we use a `emit` function to make the lower vertex writes its ID to the higher vertex in the expansion step. After expansion, we add an extra pointer-jumping phase to perform the root-finding procedure. At last, the `filter` function will mark the edge whose two end vertices have the same ID as fixed and mark the other edges as active. When all the edges become fixed, the whole algorithm terminates.


**PageRank**. Google first introduced the PageRank algorithm to calculate the relative importance of webpages. Given a graph *G=<V,E>*, a damping factor *d (0<d<1)*, the PR value of a vertex *v* is calculated by:

$$ PR[v] = \frac{1-d}{|V|} + d\sum_{u\in V^-(v)}{\frac{PR[u]}{dg^+(u)}} $$

At first, the PR values of all the vertices are set all the same and then use the equation above to compute until all of the differences of PR values between iterations drop to below a threshold value. GraphLab has introduced a Delta-PageRank algorithm to only send their changes (deltas) in PR values to their neighbors, thus not all the vertices are active during each iteration, which means less data movement.

In GSWITCH, we implement both of the traditional PageRank and the Delta-PageRank. In traditional PageRank, the `emit` function send the PR value of a vertex to its neighbors, and the new PR value is calculated in the `filter` function. In Delta-PageRank, the `emit` function only send the deltas in the last iteration, and the `filter` function will accumulate the sum of delta contributions from their neighbors. When the accumulated delta is large enough, the vertex is marked as active and updates its PR value. Both of the two algorithms have the same `comp` and `compAtomic` functions. They accumulate the messages sent from neighbors.


**Single-Source Shortest Path**. An SSSP problem takes a weighted graph and a rooted vertex as input, and compute the shortest path distance from the root vertex to each vertex reachable from it. For graphs with negative weight, Bellman-Ford algorithm can report the existence of a negative cycle. For graphs with non-negative weight, the distance of each vertex decrease monotonously, the delta-stepping algorithm is used to make a balance between work-efficiency and parallelism. As described in a recent work (Davidson2014Work), they used a Near-Far classification to preferentially update a group of vertices based on distance scoring heuristic.

We implement the sssp algorithm with our dynamic window optimization to overcome the drawback of the irregular workloads across iterations. For an edge *e* connected vertices of *v* and *u*, the `emit` function send the sum of the distance value of *v* and the weight of *e*. Then we compare the received message and the distance of *u* in the `comp` and `compAtomic` function. Note that in `compAtomic`, we use `atomicMin` to ensure atomicity. Finally, we use a `filter` function to choose the vertices whose distance fall in the dynamic window. 

**Betweenness Centrality**. Centrality is widely used to indecate the importance of nodes in social graphs. The commonest formulation of the BC problem is in Brandes's work. For a graph *G=<V, E>*, and a root vertex *root*. Let the *nsp[v]* be the number of the shortest paths from *root* to *v*. The BC value of the parent vertex $p$ of the $v$ in the BFS tree rooted as *root* can be calculated according to:

$$ BC[p] = \sum_{v\in child(p)}{\frac{nsp[p]}{nsp[v]}(BC[v]+1)} $$

The algorithm is composed of two phases: a forward phase and a backward phase. The forward phase computes the number of the shortest paths from the $root$ to each vertex in BFS order. The backward phase computes the BC value using the equation above in a reversed BFS order.

In GSWITCH, We also implement the forward and backward phases. In the forward phase, we use a `filter` function similar with BFS to choose the active vertices and use the emit function to send the number of the shortest paths to inactive vertices. The `comp` and `compAtomic` functions are used to sum the numbers of the shortest paths. In the backward phase. A `filter` function will choose and update the vertices according to the level computed in the forward phase. A `emit` function will send BC value to its neighbors, while the `comp` and `compAtomic` function do the sum work.

## Performance

Here we show 100 cases for each application (BFS,CC,PR,SSSP,BC) compared with [Gunrock](https://github.com/gunrock/gunrock) on k40m,P100,and V100. (P.S. Note that some graph below have the same names with the graph in other well-known dataset such as SNAP, but **THEY ARE DIFFERENT**.)

### K40m

**DOBFS** (both without tuning parameters)

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |
|cvxqp3.mtx| 5.87296485901| 1.6582|
|SOPF_FS_b39_c7.mtx| 6.72578811646| 1.77515|
|bundle1.mtx| 5.12003898621| 1.36597|
|scircuit.mtx| 16.7441368103| 8.01489|
|fe-ocean.mtx| 29.777765274| 12.519|
|socfb-Vanderbilt48.mtx| 2.52294540405| 0.858887|
|sme3Da.mtx| 25.4499912262| 5.25195|
|neos1.mtx| 2.70509719849| 0.912842|
|crystk02.mtx| 70.0058898926| 10.2581|
|3Dspectralwave2.mtx| 5.04207611084| 2.27588|
|3D_51448_3D.mtx| 32.0639610291| 4.72998|
|ibm_matrix_2.mtx| 36.4141464233| 4.84082|
|g7jac140sc.mtx| 18.6948776245| 2.54028|
|case39.mtx| 9.09209251404| 2.06079|
|socfb-UMass92.mtx| 4.33206558228| 0.902344|
|fp.mtx| 5.09595870972| 1.87793|
|neos2.mtx| 2.60519981384| 0.83667|
|c-73.mtx| 5.66005706787| 1.91406|
|frb50-23-1.mtx| 0.723123550415| 0.360107|
|TF17.mtx| 13.6611461639| 1.479|
|bas1lp.mtx| 4.16803359985| 0.969971|
|mri1.mtx| 3.86190414429| 1.73706|
|g7jac160sc.mtx| 31.3901901245| 3.23877|
|shar_te2-b2.mtx| 12.5088691711| 1.56421|
|socfb-Mississippi66.mtx| 3.36003303528| 0.858887|
|g7jac180.mtx| 16.1709785461| 2.86084|
|g7jac180sc.mtx| 15.9480571747| 2.87598|
|in-2004.mtx| 18.1181430817| 8.17505|
|bcsstk35.mtx| 31.7490100861| 8.92505|
|nemeth24.mtx| 104.316947937| 25.0598|
|g7jac200sc.mtx| 12.3958587646| 2.83691|
|turon_m.mtx| 18.4290409088| 8.60693|
|gas_sensor.mtx| 17.7128314972| 7.93408|
|ASIC_680ks.mtx| 12.7611160278| 3.99805|
|sgpf5y6.mtx| 4.42910194397| 2.02197|
|pds-70.mtx| 4.63199615479| 2.25073|
|TSOPF_FS_b162_c3.mtx| 7.78794288635| 1.67993|
|invextr1_new.mtx| 25.9199142456| 7.88403|
|rajat21.mtx| 66.0350341797| 7.09497|
|HFE18_96_in.mtx| 2.601146698| 1.13599|
|cage12.mtx| 6.62398338318| 3.09692|
|onf5_4-8x8-20.mtx| 41.0130004883| 2.5918|
|conf6_0-8x8-20.mtx| 6.90817832947| 2.52881|
|conf6_0-8x8-30.mtx| 37.868976593| 2.59497|
|conf6_0-8x8-80.mtx| 8.60190391541| 2.56421|
|frb59-26-3.mtx| 1.97196006775| 0.406006|
|ca-dblp-2012.mtx| 23.7967967987| 2.88403|
|nemsemm1.mtx| 11.0490322113| 3.70801|
|wave.mtx| 13.2989883423| 6.13721|
|144.mtx| 7.17806816101| 3.52515|
|pkustk05.mtx| 33.2131385803| 14.8279|
|helm2d03.mtx| 66.8899993896| 31.7368|
|TSOPF_FS_b162_c4.mtx| 9.97090339661| 1.91211|
|Maragal_7.mtx| 4.57882881165| 0.829834|
|pkustk07.mtx| 6.46114349365| 2.88281|
|TSOPF_FS_b300_c2.mtx| 153.057098389| 51.9648|
|socfb-UIllinois.mtx| 2.41589546204| 0.942871|
|Ga19As19H42.mtx| 6.99901580811| 2.89795|
|matrix-new_3.mtx| 32.1168899536| 3.60107|
|sc-nasasrb.mtx| 59.9720458984| 28.8|
|pct20stif.mtx| 154.378173828| 5.68921|
|ramage02.mtx| 34.6400756836| 3.79004|
|m_t1.mtx| 25.3880023956| 9.48096|
|mouse_gene.mtx| 5.56111335754| 1.49414|
|roadNet-PA.mtx| 238.497024536| 35.9468|
|TSOPF_FS_b39_c30.mtx| 27.0621776581| 4.07202|
|mc2depi.mtx| 247.810836792| 115.603|
|3dtube.mtx| 65.5870437622| 16.9741|
|TF18.mtx| 5.56802749634| 2.27197|
|av41092.mtx| 4.41408157349| 1.573|
|cont1_l.mtx| 7.62796401978| 1.72119|
|watson_2.mtx| 7.61318206787| 3.54785|
|web-Stanford.mtx| 64.7459030151| 29.74|
|offshore.mtx| 13.2050514221| 4.51709|
|ecology2.mtx| 276.221038818| 138.065|
|pkustk04.mtx| 89.2648696899| 15.6208|
|gupta2.mtx| 40.7769699097| 10.949|
|t3dh_e.mtx| 29.1409492493| 5.25488|
|TSOPF_FS_b300.mtx| 10.1289749146| 1.60498|
|Si87H76.mtx| 12.4158859253| 5.67993|
|ship_001.mtx| 38.3911132812| 17.387|
|c8_mat11.mtx| 5.42902946472| 0.848145|
|s3dkq4m2.mtx| 57.1749191284| 25.7554|
|TF19.mtx| 9.84597206116| 2.64185|
|pattern1.mtx| 11.3041400909| 1.03394|
|gupta3.mtx| 9.68503952026| 0.89502|
|LargeRegFile.mtx| 229.742050171| 5.1499|
|tp-6.mtx| 19.4079875946| 2.04297|
|sc-pwtk.mtx| 107.097862244| 45.554|
|soc-digg.mtx| 5.20586967468| 2.4541|
|hugetrace-00000.mtx| 414.326202393| 179.371|
|web-it-2004.mtx| 13.1268501282| 4.60498|
|frb100-40.mtx| 12.2449398041| 0.401123|
|hugetric-00000.mtx| 446.645019531| 212.517|
|adaptive.mtx| 954.504943848| 415.347|
|co-papers-citeseer.mtx| 25.2740383148| 6.47705|
|packing-500x100x100-b050.mtx| 121.479034424| 50.6799|
|socfb-A-anon.mtx| 14.4929885864| 5.31885|
|delaunay_n23.mtx| 229.893920898| 101.023|
|channel-500x100x100-b050.mtx| 157.440185547| 56.637|
|delaunay_n24.mtx| 345.646148682| 164.176|


**CC** (Both are edge-centric)

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |
|ct20stif.mtx| 2.77304649353| 0.63208|
|cbuckle.mtx| 1.98793411255| 0.968994|
|cvxbqp1.mtx| 2.50697135925| 1.03979|
|ch7-8-b4.mtx| 2.80904769897| 1.14209|
|darcy003.mtx| 1.67512893677| 0.755615|
|ch7-9-b3.mtx| 2.39992141724| 1.12109|
|dawson5.mtx| 2.69603729248| 0.722168|
|ch7-9-b5.mtx| 9.52792167664| 1.46704|
|dbir1.mtx| 1.47414207458| 0.62207|
|ch8-8-b3.mtx| 1.36804580688| 0.646729|
|dbir2.mtx| 1.76286697388| 0.635986|
|ch8-8-b4.mtx| 1.38902664185| 0.657959|
|dblp-2010.mtx| 1.62696838379| 0.733154|
|dc1.mtx| 4.0328502655| 1.6709|
|heart1.mtx| 2.78997421265| 1.38892|
|germany_osm.mtx| 5.25307655334| 1.14502|
|goodwin.mtx| 2.57897377014| 0.677002|
|graham1.mtx| 3.44300270081| 1.34717|
|graphics.mtx| 2.92897224426| 0.953125|
|great-britain_osm.mtx| 2.4049282074| 1.14014|
|heart3.mtx| 2.35795974731| 1.15308|
|gupta1.mtx| 2.17509269714| 0.86499|
|gupta2.mtx| 5.2318572998| 2.11401|
|gupta3.mtx| 3.25798988342| 0.824219|
|hood.mtx| 4.09007072449| 1.32397|
|email-EuAll.mtx| 3.44395637512| 0.971924|
|gyro.mtx| 4.87303733826| 1.03076|
|fem_filter.mtx| 3.52478027344| 1.44897|
|halfb.mtx| 2.84600257874| 1.15796|
|lp_ken_18.mtx| 1.88207626343| 0.719727|
|kkt_power.mtx| 5.60092926025| 1.89819|
|kneser_10_4_1.mtx| 3.20506095886| 1.44092|
|g7jac200.mtx| 2.00390815735| 0.779053|
|laminar_duct3D.mtx| 1.64699554443| 0.73291|
|g7jac200sc.mtx| 2.83908843994| 0.811035|
|landmark.mtx| 3.20196151733| 1.26562|
|misc-IMDB-bi.mtx| 3.06010246277| 1.52905|
|mixtank_new.mtx| 3.10087203979| 1.45825|
|mk12-b4.mtx| 3.42392921448| 0.9021|
|pwtk.mtx| 5.30791282654| 1.48877|
|raefsky2.mtx| 12.0389461517| 5.05908|
|raefsky4.mtx| 8.21399688721| 3.27686|
|raefsky5.mtx| 4.82296943665| 1.76099|
|rail4284.mtx| 4.67991828918| 2.03809|
|rail_79841.mtx| 12.1219158173| 4.26196|
|rajat15.mtx| 13.5071277618| 5.70386|
|rajat18.mtx| 6.10113143921| 2.2019|
|rajat21.mtx| 3.91912460327| 1.93018|
|rajat22.mtx| 6.81781768799| 1.96606|
|rajat24.mtx| 3.51905822754| 1.17627|
|rajat25.mtx| 3.85785102844| 1.83301|
|rajat26.mtx| 4.24194335938| 2.05713|
|rajat29.mtx| 3.48806381226| 1.69604|
|rajat30.mtx| 5.11598587036| 1.64795|
|rajat31.mtx| 3.94797325134| 1.33203|
|ramage02.mtx| 3.20506095886| 1.31421|
|rel8.mtx| 3.7841796875| 1.31104|
|relat8.mtx| 2.83789634705| 1.20581|
|relat9.mtx| 4.83107566833| 2.33887|
|rim.mtx| 7.58290290833| 3.13306|
|roadNet-PA.mtx| 4.99796867371| 1.77295|
|roadNet-TX.mtx| 3.13305854797| 1.2561|
|s1rmq4m1.mtx| 4.13298606873| 1.3811|
|s2rmq4m1.mtx| 3.08012962341| 1.30396|
|s2rmt3m1.mtx| 12.5458240509| 1.44409|
|s3rmq4m1.mtx| 4.00805473328| 1.81982|
|s3rmt3m1.mtx| 16.0698890686| 3.58105|
|s4dkt3m2.mtx| 15.3570175171| 7.37305|
|scircuit.mtx| 15.6872272491| 7.22681|
|shallow_water1.mtx| 4.41002845764| 1.79883|
|shallow_water2.mtx| 9.36603546143| 2.34106|
|shar_te2-b3.mtx| 4.53901290894| 1.55615|
|ship_001.mtx| 3.2639503479| 1.48096|
|ship_003.mtx| 5.72085380554| 1.68018|
|shipsec1.mtx| 6.16502761841| 1.90625|
|shipsec8.mtx| 6.31785392761| 1.8772|
|shyy161.mtx| 9.32383537292| 3.2749|
|sinc12.mtx| 5.74588775635| 1.43604|
|sinc15.mtx| 5.76305389404| 2.42798|
|sinc18.mtx| 3.32999229431| 1.39185|
|sls.mtx| 3.99708747864| 1.46387|
|sme3Db.mtx| 3.82280349731| 1.47192|
|sme3Dc.mtx| 3.29279899597| 1.58179|
|soc-LiveJournal1.mtx| 3.3700466156| 1.4751|
|t0331-4l.mtx| 16.6139602661| 2.94287|
|soc-Slashdot0811.mtx| 13.1039619446| 4.91895|
|soc-Slashdot0902.mtx| 24.491071701| 6.62109|
|soc-sign-Slashdot090216.mtx| 16.2551403046| 7.62207|
|soc-sign-Slashdot090221.mtx| 5.95283508301| 2.33105|
|t3dh.mtx| 3.51691246033| 1.54297|
|turon_m.mtx| 5.90896606445| 2.58496|
|twotone.mtx| 6.28089904785| 2.56299|
|vanbody.mtx| 8.50200653076| 3.26782|
|venkat01.mtx| 8.58092308044| 3.37793|
|venkat25.mtx| 8.06283950806| 2.68311|
|vfem.mtx| 45.9461212158| 18.7581|
|socfb-Columbia2.mtx| 10.1180076599| 3.64478|
|viscorocks.mtx| 26.5548229218| 6.375|
|socfb-Cornell5.mtx| 36.0109806061| 17.2588|
|water_tank.mtx| 18.089056015| 6.35889|
|socfb-Duke14.mtx| 63.472032547| 19.2708|

**PageRank** (with the same threshold)

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |
|3D_51448_3D.mtx| 11.2380981445| 4.72192|
|g7jac140.mtx| 12.6740932465| 5.61499|
|hamming10-2.mtx| 10.3750228882| 3.84302|
|Raj1.mtx| 41.9518947601| 8.49902|
|Maragal_6.mtx| 13.1239891052| 6.34595|
|lp1.mtx| 183.020114899| 32.4751|
|bcsstk37.mtx| 11.9321346283| 4.4729|
|patents_main.mtx| 83.2149982452| 8.35303|
|frb50-23-2.mtx| 11.0521316528| 3.91821|
|hvdc2.mtx| 12.9191875458| 6.06494|
|soc-LiveJournal1.mtx| 635.165929794| 19.156|
|stat96v1.mtx| 216.727018356| 7.70361|
|pds-50.mtx| 103.451013565| 7.41699|
|rail2586.mtx| 239.165067673| 7.85083|
|cage13.mtx| 174.88694191| 6.36206|
|tmt_unsym.mtx| 30.3120613098| 5.32202|
|g7jac160sc.mtx| 13.3528709412| 6.36816|
|shar_te2-b2.mtx| 280.174016953| 5.56519|
|tech-RL-caida.mtx| 18.2960033417| 8.03223|
|thermomech_dM.mtx| 36.1630916595| 5.78906|
|boyd2.mtx| 72.9038715363| 24.9932|
|nemeth22.mtx| 14.3570899963| 4.4812|
|soc-twitter-follows.mtx| 113.15202713| 16.3601|
|TSOPF_RS_b2052_c1.mtx| 13.6208534241| 5.01294|
|pds-60.mtx| 127.330064774| 9.78198|
|rgg_n_2_17_s0.mtx| 15.4240131378| 6.97412|
|torso3.mtx| 101.810932159| 6.40479|
|nemeth23.mtx| 11.8088722229| 4.698|
|ASIC_320ks.mtx| 59.8199367523| 11.7651|
|Lin.mtx| 36.6859436035| 5.61621|
|NotreDame_www.mtx| 143.15199852| 10.0251|
|delaunay_n18.mtx| 43.7529087067| 6.88916|
|language.mtx| 176.179885864| 12.626|
|qa8fm.mtx| 14.3330097198| 6.28003|
|rajat24.mtx| 103.203058243| 15.6089|
|shar_te2-b3.mtx| 14.6470069885| 7.27686|
|dblp-2010.mtx| 108.412981033| 11.563|
|mk13-b5.mtx| 63.9169216156| 6.79297|
|ca-MathSciNet.mtx| 210.743188858| 12.063|
|ASIC_680ks.mtx| 77.1999359131| 12.1309|
|socfb-Harvard1.mtx| 15.7029628754| 7.302|
|venkat50.mtx| 13.2749080658| 5.90308|
|ns3Da.mtx| 14.6100521088| 7.1272|
|sgpf5y6.mtx| 84.9740505219| 7.58691|
|pds-70.mtx| 144.984006882| 12.1709|
|mono_500Hz.mtx| 16.0708427429| 7.51318|
|cfd1.mtx| 14.0058994293| 6.60376|
|rajat29.mtx| 166.009902954| 23.2021|
|rajat21.mtx| 115.41891098| 20.0891|
|pds-80.mtx| 161.581993103| 13.7842|
|darcy003.mtx| 65.4871463776| 9.15601|
|mario002.mtx| 68.2969093323| 9.23975|
|cage12.mtx| 18.6719894409| 8.12598|
|coAuthorsDBLP.mtx| 45.9468364716| 12.2029|
|mixtank_new.mtx| 13.867855072| 6.58618|
|kneser_10_4_1.mtx| 68.2787895203| 11.219|
|c-big.mtx| 276.998996735| 11.4849|
|atmosmodd.mtx| 49.9241352081| 7.31396|
|atmosmodj.mtx| 57.893037796| 7.31396|
|pds-90.mtx| 177.029132843| 14.313|
|conf6_0-8x8-80.mtx| 14.4059658051| 6.8999|
|neos.mtx| 580.847024918| 15.8191|
|frb59-26-2.mtx| 18.6479091644| 5.78711|
|watson_1.mtx| 106.513023376| 9.35425|
|dbic1.mtx| 22.0100879669| 10.8318|
|web-NotreDame.mtx| 25.908946991| 12.437|
|pds-100.mtx| 190.760850906| 15.6812|
|Freescale1.mtx| 554.043054581| 15.637|
|socfb-MSU24.mtx| 22.9361057281| 9.96118|
|connectus.mtx| 503.82900238| 33.916|
|dbir2.mtx| 20.4219818115| 9.23389|
|helm2d03.mtx| 59.7171783447| 10.2249|
|thermal2.mtx| 94.3360328674| 11.5869|
|soc-delicious.mtx| 479.265928268| 19.262|
|flickr.mtx| 320.264101028| 22.3999|
|circuit5M.mtx| 662.358999252| 22.2041|
|nlpkkt120.mtx| 543.653011322| 10.186|
|Si41Ge41H72.mtx| 22.9659080505| 11.3999|
|pkustk03.mtx| 17.9071426392| 8.92993|
|inf-roadNet-PA.mtx| 101.7100811| 15.4719|
|roadNet-PA.mtx| 113.131999969| 16.2612|
|ljournal-2008.mtx| 1490.26417732| 24.2891|
|inf-belgium_osm.mtx| 117.704153061| 19.1069|
|belgium_osm.mtx| 109.932899475| 19.188|
|delaunay_n19.mtx| 83.4739208221| 12.1411|
|mc2depi.mtx| 72.3390579224| 10.1758|
|parabolic_fem.mtx| 88.7501239777| 11.2102|
|Hamrle3.mtx| 136.647939682| 12.6802|
|Rucci1.mtx| 780.635118484| 16.3708|
|great-britain_osm.mtx| 132.483005524| 22.6492|
|cont1_l.mtx| 702.295064926| 130.175|
|GL7d17.mtx| 621.984004974| 26.2939|
|t2em.mtx| 89.1511440277| 13.2307|
|watson_2.mtx| 229.428052902| 18.553|
|roadNet-TX.mtx| 132.482051849| 19.7939|
|GL7d22.mtx| 442.041873932| 20.688|
|GL7d16.mtx| 628.221988678| 25.3667|
|ecology2.mtx| 101.673126221| 13.885|
|ecology1.mtx| 98.906993866| 13.8862|
|webbase-1M.mtx| 560.477018356| 30.0222|
|apache2.mtx| 93.6460494995| 13.0732|

**SSSP** (Both enable stepping)

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |
|para-6.mtx| 1.99699401855| 0.929932|
|para-9.mtx| 2.05898284912| 1.01611|
|hvdc1.mtx| 7.55000114441| 3.28394|
|bauru5727.mtx| 15.6869888306| 4.87988|
|cavity17.mtx| 45.8748321533| 7.22021|
|cavity18.mtx| 14.4829750061| 5.86914|
|cavity20.mtx| 13.1051540375| 5.89819|
|cavity22.mtx| 13.090133667| 5.979|
|cavity24.mtx| 14.9431228638| 6.05029|
|cavity26.mtx| 14.4040584564| 5.62134|
|Kemelmacher.mtx| 4.13799285889| 2.06104|
|graphics.mtx| 15.5980587006| 2.51416|
|c-61.mtx| 5.62405586243| 2.32593|
|s1rmq4m1.mtx| 15.4628753662| 7.125|
|coater2.mtx| 18.4071063995| 7.71191|
|fem_hifreq_circuit.mtx| 28.4621715546| 13.9224|
|c-56.mtx| 12.2361183167| 4.72998|
|ncvxqp5.mtx| 8.85891914368| 4.23413|
|helm3d01.mtx| 13.8421058655| 4.64893|
|onetone2.mtx| 9.84501838684| 4.61597|
|graham1.mtx| 12.6340389252| 5.04199|
|inlet.mtx| 134.886032104| 27.9131|
|ncvxqp3.mtx| 8.66389274597| 4.15625|
|ex40.mtx| 35.7789993286| 9.49316|
|c-67.mtx| 18.7311172485| 6.06299|
|deltaX.mtx| 20.9989547729| 3.05615|
|c-68.mtx| 21.7459201813| 7.40894|
|epb3.mtx| 100.191833496| 48.063|
|fxm4_6.mtx| 8.71682167053| 4.24194|
|FEM_3D_thermal2.mtx| 74.4819641113| 33.832|
|mark3jac120.mtx| 20.9710597992| 7.92773|
|mark3jac120sc.mtx| 20.5562114716| 7.9209|
|c-69.mtx| 15.9959793091| 7.14575|
|c-70.mtx| 28.1360149384| 5.41309|
|c-72.mtx| 14.356136322| 6.28101|
|mark3jac140sc.mtx| 21.8830108643| 8.83105|
|email-EuAll.mtx| 18.7590122223| 5.49683|
|heart2.mtx| 13.9570236206| 3.14014|
|image_interp.mtx| 93.9931869507| 46.0579|
|RFdevice.mtx| 8.47601890564| 3.99292|
|dc2.mtx| 18.0118083954| 3.23193|
|flower_8_4.mtx| 8.55183601379| 4.20117|
|c-71.mtx| 19.1378593445| 8.46826|
|nemeth19.mtx| 87.0599746704| 41.218|
|fe_ocean.mtx| 52.4818878174| 23.8398|
|cont-300.mtx| 134.707931519| 59.4211|
|fe-tooth.mtx| 23.0369567871| 9.19873|
|fe_tooth.mtx| 22.8610038757| 9.2251|
|2D_54019_highK.mtx| 34.0430755615| 13.021|
|Dubcova2.mtx| 38.459777832| 15.2349|
|3Dspectralwave2.mtx| 25.2449512482| 5.37671|
|case39.mtx| 16.0081386566| 4.35571|
|c-73.mtx| 22.5808620453| 5.95605|
|frb50-23-1.mtx| 7.50207901001| 2.1499|
|bas1lp.mtx| 20.1640129089| 5.13013|
|rail2586.mtx| 23.0309963226| 8.84619|
|cage13.mtx| 14.0771865845| 6.42993|
|boyd2.mtx| 34.9929351807| 17.1699|
|li.mtx| 20.7870006561| 10.0059|
|TSOPF_RS_b39_c19.mtx| 20.1978683472| 4.65405|
|vfem.mtx| 31.6431522369| 13.033|
|soc-twitter-follows.mtx| 49.0310211182| 7.83105|
|rgg_n_2_17_s0.mtx| 138.471130371| 65.7761|
|598a.mtx| 22.7079391479| 10.314|
|d_pretok.mtx| 97.1269607544| 24.9761|
|turon_m.mtx| 51.9452095032| 21.1318|
|TSOPF_RS_b162_c4.mtx| 11.5170478821| 4.1748|
|CO.mtx| 14.4340991974| 6.22021|
|language.mtx| 28.7320613861| 13.5547|
|venkat01.mtx| 29.2029380798| 13.386|
|venkat25.mtx| 28.9130210876| 13.729|
|venkat50.mtx| 28.5458564758| 12.2239|
|cfd1.mtx| 35.8240585327| 17.271|
|appu.mtx| 11.3160610199| 4.98608|
|darcy003.mtx| 283.935058594| 103.494|
|mario002.mtx| 232.207061768| 104.589|
|cage12.mtx| 20.2748775482| 9.62085|
|net100.mtx| 13.2060050964| 6.22876|
|TSC_OPF_1047.mtx| 25.377035141| 9.29712|
|af_shell2.mtx| 63.159942627| 31.4221|
|atmosmodj.mtx| 81.4990997314| 36.4758|
|water_tank.mtx| 37.1270179749| 14.741|
|conf5_4-8x8-05.mtx| 45.0170059204| 4.90186|
|conf5_4-8x8-10.mtx| 38.7840270996| 4.85083|
|conf5_4-8x8-15.mtx| 15.22397995| 4.8501|
|conf5_4-8x8-20.mtx| 15.4058933258| 4.8667|
|conf6_0-8x8-20.mtx| 19.6619033813| 4.89722|
|conf6_0-8x8-30.mtx| 57.4040412903| 4.82007|
|conf6_0-8x8-80.mtx| 12.1150016785| 4.84912|
|H2O.mtx| 18.8748836517| 7.59497|
|Hook_1498.mtx| 29.0341377258| 11.9731|
|connectus.mtx| 14.9919986725| 5.77002|
|crashbasis.mtx| 353.230010986| 42.5957|
|majorbasis.mtx| 106.11390686| 42.4351|
|helm2d03.mtx| 166.405914307| 56.8552|
|mac_econ_fwd500.mtx| 166.723007202| 42.0881|
|cop20k_A.mtx| 57.6269607544| 17.2839|
|filter3D.mtx| 67.2008972168| 33.5322|
|ct20stif.mtx| 25.7580280304| 9.23901|
|pct20stif.mtx| 21.7549800873| 9.0769|
|socfb-Penn94.mtx| 12.9871368408| 6.48999|

**BC** 

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |
|g7jac040.mtx| 18.385887146| 3.14819|
|g7jac040sc.mtx| 18.0320739746| 3.09229|
|graphics.mtx| 3.72290611267| 1.65112|
|shallow_water2.mtx| 94.0580368042| 45.4141|
|chipcool1.mtx| 15.0260925293| 7.50708|
|circuit_4.mtx| 9.19985771179| 4.55908|
|c-61.mtx| 6.90698623657| 2.52686|
|raefsky1.mtx| 9.58585739136| 4.63477|
|poisson3Da.mtx| 16.6900157928| 7.98193|
|garon2.mtx| 24.6660709381| 11.2969|
|helm3d01.mtx| 11.8319988251| 4.46387|
|c-59.mtx| 21.595954895| 3.54321|
|lhr10c.mtx| 35.4959945679| 3.38306|
|lhr11c.mtx| 8.07499885559| 3.45264|
|c-67.mtx| 5.12003898621| 2.29785|
|c-67b.mtx| 5.1441192627| 2.271|
|deltaX.mtx| 4.64200973511| 2.04517|
|c-62.mtx| 14.5308971405| 2.21826|
|nd6k.mtx| 11.3220214844| 5.41382|
|ncvxqp7.mtx| 12.7639770508| 4.04565|
|lung2.mtx| 113.388061523| 54.4592|
|bayer01.mtx| 5.33008575439| 2.61475|
|sinc12.mtx| 13.4980678558| 3.07568|
|EAT_SR.mtx| 77.9559631348| 2.26709|
|lhr14c.mtx| 30.8728218079| 4.05615|
|mk12-b4.mtx| 4.11581993103| 1.93848|
|n4c6-b12.mtx| 4.47702407837| 2.10889|
|heart2.mtx| 5.54609298706| 2.70483|
|pds-30.mtx| 5.62596321106| 2.79321|
|g7jac100sc.mtx| 33.3449859619| 4.36548|
|socfb-Cal65.mtx| 17.3230171204| 2.83228|
|lp_ken_18.mtx| 8.3920955658| 3.24927|
|soc-slashdot.mtx| 8.59308242798| 4.16309|
|socfb-Bingham82.mtx| 6.3648223877| 2.8877|
|RFdevice.mtx| 5.85889816284| 2.34131|
|dc2.mtx| 20.0400352478| 1.92773|
|dc1.mtx| 4.13012504578| 1.94727|
|flower_8_4.mtx| 5.85007667542| 2.74097|
|lhr17.mtx| 25.7549285889| 3.50903|
|psmigr_2.mtx| 4.82797622681| 1.66772|
|psmigr_3.mtx| 15.5189037323| 2.08545|
|cit-HepPh.mtx| 12.1970176697| 4.4082|
|socfb-Vanderbilt48.mtx| 6.68096542358| 2.67383|
|socfb-UCF52.mtx| 6.24513626099| 2.698|
|g7jac120.mtx| 25.1448154449| 5.43384|
|stormg2-125.mtx| 7.38501548767| 3.41333|
|fome21.mtx| 8.3920955658| 3.73193|
|socfb-GWU54.mtx| 15.0549411774| 3.22876|
|scc_twitter-copen.mtx| 8.60500335693| 2.59009|
|socfb-JMU79.mtx| 12.5648975372| 3.22998|
|socfb-Northwestern25.mtx| 13.1771564484| 3.24756|
|socfb-Duke14.mtx| 14.0700340271| 3.49585|
|case39.mtx| 12.2880935669| 3.68408|
|socfb-UMass92.mtx| 18.9130306244| 2.90063|
|socfb-UC33.mtx| 13.1318569183| 5.91504|
|fp.mtx| 9.06300544739| 4.48364|
|socfb-NotreDame57.mtx| 6.90412521362| 3.19897|
|boyd1.mtx| 7.52687454224| 3.11206|
|frb50-23-2.mtx| 5.78784942627| 1.42261|
|frb50-23-5.mtx| 3.18813323975| 1.46704|
|TF17.mtx| 8.5551738739| 3.17822|
|stat96v1.mtx| 28.0420780182| 3.61108|
|g7jac160.mtx| 29.0489196777| 5.70093|
|shar_te2-b2.mtx| 17.4961090088| 7.35889|
|socfb-UConn.mtx| 28.5861492157| 3.41406|
|socfb-UConn91.mtx| 11.9259357452| 3.38623|
|tech-RL-caida.mtx| 18.9678668976| 5.9502|
|socfb-Mississippi66.mtx| 6.03199005127| 2.75195|
|socfb-BU10.mtx| 18.3501243591| 3.23022|
|SiO.mtx| 18.1999206543| 4.69629|
|socfb-MU78.mtx| 32.1409683228| 2.87305|
|Trec13.mtx| 4.96912002563| 2.44385|
|socfb-Baylor93.mtx| 9.31692123413| 3.28613|
|bibd_17_8.mtx| 5.17201423645| 1.77124|
|socfb-UPenn7.mtx| 12.5470161438| 3.36426|
|socfb-Virginia63.mtx| 13.099193573| 3.31812|
|ch7-8-b4.mtx| 7.75694847107| 3.16602|
|frb53-24-4.mtx| 3.2069683075| 1.40503|
|frb53-24-2.mtx| 3.32713127136| 1.39526|
|frb53-24-3.mtx| 6.68001174927| 1.37012|
|socfb-NYU9.mtx| 17.2910690308| 3.24414|
|n4c6-b6.mtx| 6.61706924438| 3.03979|
|socfb-Maryland58.mtx| 28.785943985| 2.65503|
|socfb-UCLA.mtx| 5.82218170166| 2.84351|
|socfb-UCLA26.mtx| 6.05010986328| 2.89697|
|g7jac200sc.mtx| 14.6560668945| 6.68311|
|socfb-Tennessee95.mtx| 8.18705558777| 3.05298|
|m133-b3.mtx| 18.2960033417| 3.93774|
|ca-MathSciNet.mtx| 17.8940296173| 6.7312|
|rel8.mtx| 26.0708332062| 4.00903|
|socfb-Harvard1.mtx| 15.2611732483| 3.58618|
|fem_filter.mtx| 162.341125488| 24.7131|
|n4c6-b11.mtx| 7.31992721558| 3.60474|
|socfb-Wisconsin87.mtx| 14.2209529877| 3.97876|
|nw14.mtx| 7.80200958252| 2.82007|
|socfb-Auburn71.mtx| 8.47315788269| 3.10718|
|C2000-5.mtx| 14.2478942871| 1.75098|
|conf5_4-8x8-10.mtx| 30.2400588989| 7.21802|
|conf6_0-8x8-80.mtx| 18.1341171265| 6.35425|
|socfb-FSU53.mtx| 19.3450450897| 3.05591|
|net4-1.mtx| 61.3079071045| 12.9727|

### P100

**DOBFS** (both without tuning parameters)

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |


**CC** (Both are edge-centric)

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |


**PageRank** (with the same threshold)

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |


**SSSP** (Both enable stepping)

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |

**BC** 

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |
### V100

**DOBFS** (both without tuning parameters)

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |


**CC** (Both are edge-centric)

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |


**PageRank** (with the same threshold)

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |


**SSSP** (Both enable stepping)

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |

**BC** 

| **Dataset** | **Gunrock** | **GSWITCH** |
| ------- | ------- | ------- |


## License
All the libraryies, examples, and source codes of GSWITCH are released under [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0).

