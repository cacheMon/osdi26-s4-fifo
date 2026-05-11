# S4-FIFO: Artifact Reproduction Guide

This guide describes how to reproduce the key results from the OSDI 2026 paper "Learning-Augmented Heuristics: Simple, yet Smart, Robust and Interpretable Cache Eviction."

## Artifact Overview

This artifact contains:

| Component | Location | Description |
|-----------|----------|-------------|
| S4-FIFO implementation | `libCacheSim/libCacheSim/cache/eviction/S4FIFO*.c` | Cache simulator with S4-FIFO, S4-FIFO-base, and S4-FIFO-verify |
| CacheLib prototype | `CacheLib/` | Meta CacheLib integration |
| Model training scripts | `analysis/` | LightGBM training, feature analysis, evaluation, plotting |
| Infrastructure | `Dockerfile`, `Makefile`, `requirements.txt` | Build and Docker setup |

### Supported Algorithms

The simulator includes the following eviction algorithms for comparison: S4-FIFO (`s4fifo`), S4-FIFO-base (`s4fifo-base`), S4-FIFO-verify (`s4fifo-verify`), S3-FIFO (`s3fifo`), LRU (`lru`), 2Q (`2q`), ARC (`arc`), LeCaR (`lecar`), LIRS (`lirs`), Sieve (`sieve`), LFU (`lfu`), FIFO (`fifo`), and others.

## Quick Smoke Test (5 minutes)

This verifies the artifact builds and runs correctly.

```bash
# Build
make build

# Run S4-FIFO on the included sample trace (100K requests)
make test

# Test all three S4-FIFO variants
libCacheSim/build/bin/cachesim libCacheSim/data/cloudPhysicsIO.vscsi vscsi s4fifo 1GB -n 100000
libCacheSim/build/bin/cachesim libCacheSim/data/cloudPhysicsIO.vscsi vscsi s4fifo-base 1GB -n 100000
libCacheSim/build/bin/cachesim libCacheSim/data/cloudPhysicsIO.vscsi vscsi s4fifo-verify 1GB -n 100000
```

All three variants should report **miss ratio ~0.48** on this sample trace.

## Phase-by-Phase Reproduction

The paper's evaluation uses 5,175 production traces from 14 sources. Because many of these traces contain proprietary data, they are available upon request. Below we describe each phase so evaluators can verify the methodology and, where traces are available, reproduce results.

### Phase 1: Dataset Preparation

**What the paper does**: Collect 5,175 production traces from 14 sources (block storage, key-value stores, CDN caches). Filter out traces with fewer than 1 million objects. Randomly split into 4,140 training and 1,035 evaluation traces.

**Trace sources** (Table 4 in the paper):

| Dataset | Year | Cache Type | Traces |
|---------|------|-----------|--------|
| MSR | 2007 | Block | 14 |
| FIU | 2008-11 | Block | 7 |
| CloudPhysics | 2015 | Block | 104 |
| Systor | 2017 | Block | 6 |
| CDN 1 | 2018 | Object | 163 |
| Tencent Photo | 2018 | Object | 2 |
| WikiMedia CDN | 2016-19 | Object | 4 |
| Tencent CBS | 2020 | Block | 3,370 |
| Alibaba | 2020 | Block | 552 |
| Twitter | 2020 | KV | 50 |
| CDN 2 | 2021 | Object | 890 |
| Meta Storage | 2022 | Block | 5 |
| Meta KV | 2022 | KV | 5 |
| Meta CDN | 2023 | Object | 3 |

**To reproduce**: Place traces in a directory and use `cachesim` with the appropriate trace type flag (`vscsi`, `oracleGeneralBin`, `csv`, `txt`, `twr`). The included `cloudPhysicsIO.vscsi` trace in `libCacheSim/data/` can be used for functional verification.

### Phase 2: Parameter Search and Feature Collection

**What the paper does**: For each of 4,140 training traces, run S4-FIFO with 168 parameter configurations (grid search over discretized parameters). Record the best configuration (lowest miss ratio) as the training label. Separately, run S4-FIFO with default parameters on each trace to collect the 73-dimensional feature vector.

**Parameter search space (168 configurations)**:

| Parameter | Symbol | Values |
|-----------|--------|--------|
| Small queue ratio | ρS | 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90 |
| Ghost queue ratio | ρG | 3x, 6x |
| Small-to-main threshold | τS | 1, 2 |
| Ghost-to-main threshold | τG | 0, 1 |
| Skip ratio | κ | 0.00, 0.25 |

**Feature vector (73 dimensions)**:

| Feature Category | Dims | Description |
|-----------------|------|-------------|
| Small histogram | 20 | 20-bin hit-position distribution in small queue |
| Main histogram | 20 | 20-bin hit-position distribution in main queue |
| Ghost histogram | 20 | 20-bin hit-position distribution in ghost queue |
| Queue hit ratios | 3 | Proportion of hits in each queue (S, M, G) |
| Log cache size | 1 | log(cache capacity) |
| Utility gap | 1 | \|H_main - H_small\| / H_total |
| Filtering efficiency | 1 | H_small / H_main |
| Ghost pressure | 1 | H_ghost / (H_ghost + H_total) |
| Tail heaviness | 1 | Sum of hits in last 10 bins of main histogram |
| Decay rate | 1 | Slope between first two bins of small histogram |
| One-hit ratio | 1 | # freq=1 evictions / # small evictions |
| Unique ratio | 1 | # unique objects / # requests |
| Scan intensity | 1 | Unique ratio normalized by request rate |
| Thrashing risk | 1 | Combined signal for working set exceeding capacity |

**To reproduce**: Use the feature collection built into S4-FIFO:

```bash
# Collect features and dump to file
cachesim trace.vscsi vscsi s4fifo 1GB \
  -e "collect-features=true,feature-collect-reqs=100000,dump-file=features.txt"
```

### Phase 3: Configuration Discretization

**What the paper does**: From the 168 grid-search configurations, select 18 representative parameter sets using a greedy set-cover algorithm. This reduces the classification space while maintaining near-optimal coverage.

**To reproduce**: See `analysis/create_grid_optimal.py`.

### Phase 4: Model Training

**What the paper does**: Train a cost-sensitive LightGBM classifier on the (features, best-config) pairs from Phase 2.

**Model configuration**:
- Algorithm: LightGBM (Gradient Boosted Decision Trees)
- Trees: 20, max depth: 9
- Input: 73-dimensional feature vector
- Output: Probability distribution over 18 configurations
- Objective: Cost-sensitive classification with FIFO-anchored asymmetric cost matrix

**Cost matrix**: L[k][j] = E_train[(MR(config_k) - MR(config_j)) / MR_FIFO], where MR_FIFO is FIFO's miss ratio used as anchor for robustness.

**To reproduce**: See training scripts in `analysis/`:

```bash
# Train the model
python3 analysis/train_xgb_18class.py

# Export to dependency-free code (C++, Python, Go, Rust, etc.)
python3 analysis/export_models_multilang.py
```

### Phase 5: Evaluation

**What the paper does**: For each of 1,035 evaluation traces:

1. Run S4-FIFO with default parameters for the first 20% of the trace (warmup + feature collection)
2. Make one prediction using the pre-trained model
3. Apply the predicted parameters and run the remaining 80% of the trace
4. Compare against baselines: S3-FIFO, 3L-Cache, LRB, LHD, GL-Cache, LeCaR, ARC, 2Q, LIRS

Cache sizes tested: 0.1%, 1%, and 10% of each trace's working set size.

**To reproduce**:

```bash
# Run S4-FIFO with prediction (20% warmup, then predicted params)
cachesim trace.vscsi vscsi s4fifo 1GB \
  -e "collect-features=true,feature-collect-reqs=50000"

# Run baselines for comparison
cachesim trace.vscsi vscsi s3fifo 1GB
cachesim trace.vscsi vscsi lru 1GB
cachesim trace.vscsi vscsi arc 1GB
```

**Expected results** (from the paper):

| Metric | Result |
|--------|--------|
| Mean improvement over S3-FIFO | +26% miss ratio reduction |
| Mean improvement over 3L-Cache | +8% miss ratio reduction |
| Worst-case vs FIFO (large cache) | +0.8% miss ratio increase |
| Worst-case vs FIFO (small cache) | +0.2% miss ratio increase |
| 3L-Cache worst-case vs FIFO | +8.8% miss ratio increase |

### Phase 6: Throughput Benchmarking

**What the paper does**: Measure S4-FIFO throughput using CacheBench from Meta's CacheLib. Compare against highly optimized LRU, 2Q, S3-FIFO, and TinyLFU implementations.

**Setup**: 48 threads, CacheBench CDN + four Graph Cache workloads.

**Expected results**: S4-FIFO throughput is comparable to S3-FIFO and outperforms LRU, even with continuous feature collection (worst-case). Feature collection runs outside the critical section and does not affect correctness.

**To reproduce**: See `CacheLib/README.md` for CacheLib build and benchmark instructions.

### Phase 7: Generalization and Analysis

**What the paper does**:

- **Cross-dataset generalization**: Train on CDN2 (object cache), evaluate on Twitter (KV cache). The model generalizes because caching patterns (scanning, looping, thrashing) transcend specific datasets.
- **Feature importance**: Histogram features contribute ~75% of total importance. Queue-level locality shape is the strongest signal.
- **Training data scaling**: Top-1 accuracy ~60%, top-3 ~80% with several thousand training traces.

**To reproduce**: See `analysis/cross_dataset_simple.py` and `analysis/export_feature_importance.py`.

## S4-FIFO Parameters Reference

Parameters are set via the `-e` flag:

```bash
cachesim trace.vscsi vscsi s4fifo 1GB \
  -e "small-size-ratio=0.20,ghost-size-ratio=6,move-to-main-threshold=1"
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `small-size-ratio` | Fraction of cache for the small queue | 0.10 |
| `ghost-size-ratio` | Ghost queue size as multiple of cache | 0.90 |
| `move-to-main-threshold` | Hits before promoting to main queue | 2 |
| `small-skip-ratio` | Skip ratio for small queue frequency update | 0 |
| `ghost-to-main-threshold` | Ghost hits before admitting to main | 0 |
| `collect-features` | Enable feature collection | false |
| `feature-collect-reqs` | Requests to collect features before prediction | 100000 |
| `prediction-interval` | Re-predict every N requests (0 = one-shot) | 0 |
| `dump-file` | Dump collected features to file | (none) |

Use `-e "print=true"` to print the current parameter values.

## Troubleshooting

**Build fails with missing headers**: Ensure the `s4-fifo` branch of libCacheSim is checked out: `git submodule update --init --recursive`

**Trace file not found**: The included `cloudPhysicsIO.vscsi` is a sample trace. Production traces are available upon request from the authors.

**Model accuracy below expected**: Verify that feature collection is enabled and the trace has enough requests for warmup (>100K recommended). Use `dump-file` to inspect the collected features.

**Results differ from paper**: Check cache size (must be relative to working set), trace type flag, and that the correct S4-FIFO variant is used (`s4fifo` for the main variant).

## System Requirements

**Minimum**: Linux x86_64 (Ubuntu 20.04+), gcc 11+, cmake 3.12+, Python 3.9+, 4 cores, 16GB RAM

**Recommended for full reproduction**: 32 cores, 192GB RAM, 500GB disk (for grid search over all traces)

**Docker alternative**: `docker build -t s4fifo . && docker run -it s4fifo bash`

## Citation

```bibtex
@inproceedings{xia2026s4fifo,
  title={Learning-Augmented Heuristics: Simple, yet Smart, Robust and Interpretable Cache Eviction},
  author={Xia, Haocheng and Nixon, William and Marthen, Bintang Dwi and Bhandari, Pranav and Yang, Juncheng},
  booktitle={Proceedings of the 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year={2026}
}
```
