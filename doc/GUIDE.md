# S4-FIFO: Artifact Guide

## Paper Overview

**Title**: Learning-Augmented Heuristics: Simple, yet Smart, Robust and Interpretable Cache Eviction

**Authors**: Haocheng Xia (Harvard & UIUC), William Nixon (U Chicago), Bintang Dwi Marthen (ITB), Pranav Bhandari (Meta), Juncheng Yang (Harvard)

**Venue**: OSDI 2026

## Key Contributions

S4-FIFO introduces **Learning-Augmented Heuristics (LAH)**, a framework that augments simple cache heuristics with machine learning:

1. **Classification**: Identifies periodic cache-level learning as unexplored sweet spot
2. **LAH Framework**: Decouples data path (simple heuristic) and control path (ML model)
3. **S4-FIFO Implementation**: Learning-augmented S3-FIFO on pre-trained GBDT
4. **Strong Results**: 26% efficiency improvement, 0.8% robustness guarantee

## Quick Summary

### Results
- **Efficiency**: +26% mean improvement over S3-FIFO, +8% over 3L-Cache
- **Robustness**: Only 0.8% worst-case miss ratio increase vs FIFO (vs 3L-Cache's 8.8%)
- **Performance**: Throughput parity with traditional heuristics
- **Interpretability**: Clear cache-level parameters with operational semantics

### Key Insight
Most smart caches either:
- **Object-level, per-miss**: Suffer from objective mismatch and instability
- **Cache-level, per-miss**: Have delayed reward and magic parameters

**S4-FIFO (Cache-level, periodic)**: None of the above!

## What's in This Artifact?

### Documentation

1. **GUIDE.md** (this file): Overview and quick start
2. **REPRODUCTION.md**: Detailed reproduction guide

### Data & Models

- Production traces (5,175 from 14 sources) - Contact for access
- Pre-trained S4-FIFO GBDT model (4,140 training traces)
- Feature engineering specification (73 dimensions)
- Configuration discretization (168 → 18 classes)

### Code

- libCacheSim simulator with S4-FIFO
- Model training pipeline (LightGBM)
- Evaluation scripts (vs. 8 baselines)
- CacheLib integration for production deployment

## Reproduction Overview

### Phase 1: Data (Week 1-2)
Prepare 5,175 production traces (4,140 training, 1,035 evaluation)

### Phase 2: Features (Week 3-4)
Grid search 168 parameter configs, extract 73 features per trace
- 4,140 traces × 168 configs = ~694K simulations
- Each trace produces best-config label

### Phase 3: Training (Week 5-6)
Train cost-sensitive GBDT model
- Input: 73 features per trace
- Output: 18-class configuration selection
- Cost matrix: asymmetric penalties anchored to FIFO robustness

### Phase 4: Evaluation (Week 7)
Test on 1,035 held-out traces vs. 8 baselines
- Report efficiency and robustness

### Phase 5-10: Analysis & Deployment (Week 8-14)
- Comparative analysis, throughput benchmarking
- Generalization study, interpretability
- Production CacheLib implementation
- Open source release

**See [REPRODUCTION.md](REPRODUCTION.md) for detailed steps**

## Technical Overview

### Architecture

```
┌─────────────────────────────────┐
│ Control Plane (Async)           │
│ Pre-trained GBDT Model          │
│ Updates parameters periodically │
└─────────────┬───────────────────┘
              │ Parameter updates
              ▼
┌─────────────────────────────────┐
│ Data Plane (Fast, O(1))         │
│ S3-FIFO with learned params     │
│ Simple deterministic heuristic  │
└─────────────────────────────────┘
```

### S4-FIFO Algorithm

**4 FIFO Queues + 6 Learnable Parameters**:

| Parameter | Default | Range |
|-----------|---------|-------|
| ρS (small queue ratio) | 0.10 | [0.05-0.90] |
| ρM (main queue ratio) | 0.90 | [0.10-0.95] |
| ρG (ghost queue ratio) | 3x | {3x, 6x} |
| τS (small-to-main threshold) | 2 | {1, 2} |
| τG (ghost-to-main threshold) | 0 | {0, 1} |
| κ (skip ratio) | 0 | {0, 0.25} |

### Features (73 dimensions)

**Histogram Features (60 dims)**: 20-bin hit position distributions for each queue

**Queue Ratios (3 dims)**: Proportion of hits in each queue

**Composite Features (10 dims)**:
- Utility gap, filtering efficiency, ghost pressure
- Tail heaviness, decay rate, one-hit ratio
- Unique ratio, scan intensity, thrashing risk
- Cache size (log-transformed)

### Model

- **Algorithm**: LightGBM (Gradient Boosting Decision Trees)
- **Trees**: 20 trees, depth 9
- **Input**: 73-dimensional feature vector
- **Output**: Probability distribution over 18 configurations
- **Objective**: Cost-sensitive classification (asymmetric penalties)
- **Export**: Dependency-free C++/Python/Go/Rust/JS

## Expected Results

### Efficiency
- Mean improvement over S3-FIFO: **+26%**
- Mean improvement over 3L-Cache: **+8%**

### Robustness
- Worst-case vs FIFO: **+0.8%** miss ratio increase
- 3L-Cache worst-case: +8.8% (for comparison)

### Performance
- Throughput: Parity with S3-FIFO (≥95%)
- Latency: Feature collection <0.1%, model inference <2ms
- Overhead: ~10-50 KB total (vs. 200+ bytes/object for 3L-Cache)

## System Requirements

**Minimum**:
- Linux x86_64 (Ubuntu 20.04+)
- Python 3.9+
- 4 CPU cores, 16GB RAM, 100GB disk

**Recommended**:
- Linux x86_64 (Ubuntu 22.04)
- 8+ CPU cores, 32GB+ RAM, 200GB+ disk

**For production CacheLib deployment**:
- C++ compiler (g++ or clang++)
- libCacheSim (for testing)

## Getting Started

### 1. Setup Environment
```bash
git clone <this-repo>
cd osdi26-s4-fifo
pip install -r requirements.txt
```

### 2. Quick Evaluation (1 hour)
```bash
# Evaluate on pre-trained model with sample traces
python scripts/quick_eval.py
```

**Output**: Performance table comparing S4-FIFO vs baselines

### 3. Full Reproduction (14 weeks)
Follow the detailed steps in [REPRODUCTION.md](REPRODUCTION.md)

### 4. Production Deployment
See the "Build CacheLib (Advanced)" section in the main [README.md](../README.md) for CacheLib build instructions.

## Key Files

```
osdi26-s4-fifo/
├── doc/
│   ├── GUIDE.md                 # This file
│   └── REPRODUCTION.md          # Detailed reproduction guide
├── libCacheSim/                 # Cache simulator with S4-FIFO
├── CacheLib/                    # Meta CacheLib integration
├── analysis/                    # Model training and plotting scripts
│   └── evaluate.py              # Test evaluation
└── README.md                    # Overview
```

## Citation

```bibtex
@inproceedings{xia2026s4fifo,
  title={Learning-Augmented Heuristics: Simple, yet Smart, Robust and Interpretable Cache Eviction},
  author={Xia, Haocheng and Nixon, William and Marthen, Bintang Dwi and Bhandari, Pranav and Yang, Juncheng},
  booktitle={Proceedings of the 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year={2026}
}
```

## Baselines & Comparisons

S4-FIFO is compared against:

1. **S3-FIFO**: Static heuristic baseline
2. **3L-Cache**: State-of-the-art learning-based
3. **LRU**: Traditional algorithm
4. **2Q**: Traditional algorithm
5. **ARC**: Cache-level adaptive
6. **LeCaR**: Learning-based per-miss adaptation
7. **LRB**: Learning reuse distance
8. **GL-Cache**: Periodic object-level learning

## Contact

For questions or issues:
1. Check [REPRODUCTION.md](REPRODUCTION.md) troubleshooting section
2. Create GitHub issue with:
   - Error message/output
   - Configuration used
   - Steps to reproduce
   - System details

## License

Apache License 2.0

## Acknowledgments

- Production traces provided by participating companies
- S3-FIFO design from Juncheng Yang (OSDI 2024)
- CacheLib team at Meta for infrastructure support
