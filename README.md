
# Learning-Augmented Heuristics: Simple yet Smart, Robust and Interpretable Cache Eviction

This repo contains code for OSDI'26 paper: [Learning-Augmented Heuristics: Simple yet Smart, Robust and Interpretable Cache Eviction](https://haochengxia.com/publication/osdi-s4fifo.pdf)

<div style="text-align: center;">
  <img src="/doc/diagram/overview.svg" alt="diagram" width="480"/>
</div>

<!-- ![S3-FIFO diagram](/doc/diagram/diagram_s3fifo.svg) -->


## Abstract
Caching is widely used across the system stack to improve performance and efficiency, with eviction algorithms at its core. Existing cache eviction policies fall into two broad categories: static heuristics (e.g., 2Q, S3-FIFO) and smart algorithms (e.g., ARC, LRB). Smart caches can adapt to workloads and have the potential to achieve higher efficiency and robustness than static heuristics. However, we find that existing smart caches suffer from objective mismatches and instability.

We introduce Learning-Augmented Heuristics (LAH), a framework that learns the cache-level parameters of static heuristics. By decoupling the data and control planes, LAH supports simple, high-speed data reads and writes on the data plane, while performing occasional asynchronous learning on the control plane using cache-level metrics.

We demonstrate the effectiveness of LAH through S4-FIFO, a Smart S3-FIFO cache eviction algorithm. We pre-train a single model on 4140 production traces and embed it in S4-FIFO to learn optimal cache parameters. On 1035 evaluation traces, S4-FIFO improves the mean efficiency by 26% compared to S3-FIFO and by 8% compared to 3L-Cache, the best state-of-the-art algorithm. S4-FIFO is also robust---increasing miss ratio over FIFO by 0.8% on the worst trace, whereas 3L-Cache increases FIFO's miss ratio by 8.8%. Finally, learning-augmented heuristics also enable good interpretability: a language model can clearly explain why specific parameters are updated.

## Repo structure 
The repo is a snapshot of [libCacheSim](https://github.com/cacheMon/libCacheSim), modified [cachelib](https://github.com/facebook/cachelib/), and [distComp](https://github.com/1a1a11a/distComp). 


### How to use libCacheSim
You can compile libCacheSim, which will provide a `cachesim` binary, then you can run simulations with
```bash
# compile libcachesim
pushd libCacheSim/scripts && bash install_dependency.sh && bash install_libcachesim.sh && popd;
```

Use cacheSim to run cache simulations

```bash
# ./cachesim DATAPATH TRACE_FORMAT EVICTION_ALGO CACHE_SIZE [OPTION...]
./cachesim DATA oracleGeneral fifo,arc,lecar,s3fifo,s4fifo 0 --ignore-obj-size 1
```
Detailed instructions can be found at [libCacheSim](https://github.com/cacheMon/libCacheSim).

### How to use cachelib

***[TBD]***

### How to use distComp
If you need to scale up the computation by using more nodes, you would need to use distComp. See [here](https://github.com/1a1a11a/distComp) for more details. 


## Instructions for reproducing results (artifact evaluation)
Please see [artifact evaluation](/doc/AE.md) for detailed instructions.


## Traces
The traces we used can be downloaded [here](https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/).

The binary traces are [zstd](https://github.com/facebook/zstd) compressed and have the following format:
```c
struct {
    uint32_t timestamp;
    uint64_t obj_id;
    uint32_t obj_size;
    int64_t next_access_vtime;  // -1 if no next access
}
```
The compressed traces can be used with libCacheSim without decompression. And libCacheSim provides a tracePrint tool to print the trace in human-readable format.


### Acknowledgement
We greatly thank the following people and organizations that made this work possible. 

***[TBD]***