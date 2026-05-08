# S4-FIFO

This repository contains the artifacts for the paper  
**“Learning-Augmented Heuristics: Simple, Smart, Robust, and Interpretable Cache Eviction.”**

<div style="text-align: center;">
  <img src="/doc/diagram/overview.svg" alt="S4-FIFO overview diagram" width="480"/>
</div>

## Learning-Augmented Heuristics Framework

```mermaid
graph TD
    subgraph "Offline Training"
        A[4,140 production traces] --> B[Grid search over 168 configurations]
        B --> C[Train GBDT model]
    end

    subgraph "Online Adaptation"
        D[Cache features] --> E[Pre-trained model]
        E --> F[Select the best configuration]
    end

    subgraph "Data Path"
        G[Simple S4-FIFO heuristic with learned parameters]
    end
```

## S4-FIFO Algorithm

S4-FIFO is a learning-augmented cache eviction algorithm that combines a simple FIFO-based data path with an offline-trained model for adaptive parameter selection.

Key components include:

* **Three FIFO queues**: Small, Main, and Ghost
* **Five learnable parameters**: skip ratio, queue sizes, and promotion thresholds
* **Seventy-three features**: cache metrics and workload characteristics
* **GBDT model**: trained on 4,140 production traces with an 18-class output space

## Artifact Evaluation

For simulator-based experiments, see [`libCacheSim`](./libCacheSim/).

For real-cache experiments, see [`CacheLib`](./CacheLib/).

## License

This project is licensed under the Apache License 2.0.

See [`LICENSE`](./LICENSE) for details.

## Support

For questions or issues:

1. Check the [troubleshooting section in `REPRODUCTION.md`](doc/REPRODUCTION.md#troubleshooting).
2. Open a GitHub issue with detailed steps to reproduce the problem.
