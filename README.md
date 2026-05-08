# S4-FIFO: Learning-Augmented Heuristics for Cache Eviction

This is the artifact repository for the paper:

## Learning-Augmented Heuristics: Simple, yet Smart, Robust and Interpretable Cache Eviction

**Venue**: OSDI 2026

**Authors**: Haocheng Xia (Harvard & UIUC), William Nixon (U Chicago), Bintang Dwi Marthen (ITB), Pranav Bhandari (Meta), Juncheng Yang (Harvard)

## Overview

S4-FIFO is a **learning-augmented cache eviction algorithm** that achieves:
- **26% mean efficiency improvement** over S3-FIFO
- **8% improvement** over 3L-Cache (previous SOTA)
- **0.8% robustness guarantee** - worst-case miss ratio increase vs FIFO
- **Throughput parity** with traditional heuristics like LRU
- **Interpretable decisions** based on cache-level parameters

### Key Innovation

Unlike existing smart caches that suffer from instability or objective mismatch, S4-FIFO uses:
- **Periodic cache-level learning** (not per-miss object-level)
- **Pre-trained foundation model** on 4,140 production traces
- **Separation of concerns**: simple data path + async control path
- **Cost-sensitive learning** anchored to robustness

## Quick Start

**For a 1-hour quick evaluation**:
```bash
python scripts/quick_eval.py
```

**For full reproduction (14 weeks)**:
See [Detailed Reproduction Guide](docs/REPRODUCTION.md)

**For paper/architecture overview**:
See [Artifact Guide](docs/GUIDE.md)

## Documentation

- **[GUIDE.md](docs/GUIDE.md)** - Overview, architecture, and quick start
- **[REPRODUCTION.md](docs/REPRODUCTION.md)** - Complete 10-phase reproduction guide (14 weeks)
- **[CASE_STUDIES.md](docs/CASE_STUDIES.md)** - Related projects and implementations
- **[PROJECTS.md](docs/PROJECTS.md)** - Implementation details and file structure

## Core Concept

### Learning-Augmented Heuristics (LAH) Framework

```
Offline (Once):
  4,140 production traces → Grid search 168 configs → Train GBDT model

Online (Periodic):
  Cache features → Pre-trained model → Select best configuration

Data Path (Always):
  Simple S3-FIFO heuristic with learned parameters
```

### S4-FIFO Algorithm

- **4 FIFO Queues**: Small, Main, Ghost, Skip
- **6 Learnable Parameters**: Queue sizes, promotion thresholds
- **73 Features**: Cache metrics and workload characteristics
- **GBDT Model**: Trained on 4,140 traces, 18-class output

## Reproduction Roadmap

| Phase | Duration | Task |
|-------|----------|------|
| 1 | Week 1-2 | Collect and prepare 5,175 production traces |
| 2 | Week 3-4 | Grid search: 4,140 traces × 168 configs → features |
| 3 | Week 5-6 | Train cost-sensitive GBDT model |
| 4 | Week 7 | Evaluate on 1,035 test traces vs 8 baselines |
| 5 | Week 8 | Comparative analysis and visualization |
| 6 | Week 9 | Throughput and latency benchmarking |
| 7 | Week 10 | Generalization study (cross-workload) |
| 8 | Week 11 | Interpretability analysis with LLM |
| 9 | Week 12-13 | Production CacheLib implementation |
| 10 | Week 14 | Open source release and documentation |

**See [REPRODUCTION.md](docs/REPRODUCTION.md) for detailed instructions for each phase.**

## Key Results

### Efficiency (Figure 1)
- **S4-FIFO vs S3-FIFO**: +26% mean miss ratio reduction
- **S4-FIFO vs 3L-Cache**: +8% mean improvement
- **S4-FIFO vs traditional (LRU, 2Q)**: +15-20% improvement

### Robustness
- **S4-FIFO worst case**: +0.8% vs FIFO
- **3L-Cache worst case**: +8.8% vs FIFO
- Robustness anchored to FIFO via cost matrix

### Performance
- **Throughput**: ≥95% parity with S3-FIFO
- **Feature collection**: <0.1% overhead
- **Model inference**: <2ms, async
- **Storage**: ~10-50 KB total

## What's Included

**Data**:
- 5,175 production traces (metadata provided, contact for traces)
- Pre-trained GBDT model
- Feature specifications and statistics

**Code**:
- libCacheSim simulator with S4-FIFO implementation
- Model training pipeline (LightGBM)
- Evaluation scripts vs 8 baselines
- CacheLib integration (in development)

**Documentation**:
- Architecture and design decisions
- Feature engineering details
- Configuration discretization
- Step-by-step reproduction guide

## System Requirements

**Minimum**:
- Linux x86_64 (Ubuntu 20.04+)
- Python 3.9+
- 4 CPU cores
- 16 GB RAM
- 100 GB disk

**Recommended**:
- Ubuntu 22.04 LTS
- 8+ cores
- 32 GB RAM
- 200 GB disk

**For production**:
- C++ compiler (g++/clang++)
- Meta CacheLib library

## Citation

```bibtex
@inproceedings{xia2026s4fifo,
  title={Learning-Augmented Heuristics: Simple, yet Smart, Robust and Interpretable Cache Eviction},
  author={Xia, Haocheng and Nixon, William and Marthen, Bintang Dwi and Bhandari, Pranav and Yang, Juncheng},
  booktitle={Proceedings of OSDI 2026},
  year={2026}
}
```

## Baseline Comparisons

S4-FIFO is evaluated against:
1. **S3-FIFO** - Static heuristic baseline
2. **3L-Cache** - State-of-the-art learning-based
3. **LRU** - Traditional algorithm
4. **2Q** - Traditional algorithm
5. **ARC** - Cache-level adaptive
6. **LeCaR** - Learning per-miss with regret minimization
7. **LRB** - Learning reuse distance at object-level
8. **GL-Cache** - Periodic learning at group-level

## Files Structure

```
osdi26-s4-fifo/
├── README.md                    # This file
├── docs/
│   ├── GUIDE.md                 # Artifact overview and quick start
│   ├── REPRODUCTION.md          # Detailed 10-phase reproduction
│   ├── CASE_STUDIES.md          # Related case studies
│   └── PROJECTS.md              # Implementation details
├── data/
│   ├── traces/                  # Production traces (request on access)
│   └── models/                  # Pre-trained models
├── src/
│   ├── simulator/               # Cache simulator
│   ├── training/                # Model training
│   └── evaluation/              # Benchmarking
├── scripts/
│   ├── run_grid_search.py       # Phase 2: Parameter search
│   ├── train_model.py           # Phase 3: Model training
│   ├── evaluate.py              # Phase 4: Evaluation
│   └── quick_eval.py            # Quick 1-hour evaluation
├── requirements.txt             # Python dependencies
├── Dockerfile                   # For reproducibility
└── LICENSE                      # Apache 2.0
```

## License

Apache License 2.0

See LICENSE file for details.

## Acknowledgments

- Production traces from participating organizations
- S3-FIFO design by Juncheng Yang (OSDI 2024)
- Meta CacheLib team for infrastructure
- Collaborators from Harvard, U Chicago, ITB, Meta

## Support

For questions or issues:
1. Check [REPRODUCTION.md troubleshooting](docs/REPRODUCTION.md#troubleshooting)
2. Create GitHub issue with:
   - Error message
   - Configuration
   - Steps to reproduce
   - System details (OS, Python version)