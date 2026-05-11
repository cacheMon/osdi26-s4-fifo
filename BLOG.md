# S4-FIFO: When Simple Heuristics Meet Smart Learning

**Haocheng Xia** (Harvard & UIUC), **William Nixon** (U Chicago), **Bintang Dwi Marthen** (ITB), **Pranav Bhandari** (Meta), **Juncheng Yang** (Harvard)

---

Caching is one of the most impactful levers in systems performance. From storage stacks and databases to large-scale web services, caches absorb repeated accesses and reduce backend load. At the heart of every cache is an eviction algorithm — the policy that decides which objects stay and which go when memory runs out.

Despite years of research into sophisticated learned eviction policies, production systems overwhelmingly rely on simple heuristics like LRU, 2Q, and S3-FIFO. They're fast, predictable, and easy to debug. But they leave significant performance on the table — a static configuration that works well for one workload can be far from optimal for another.

Meanwhile, the "smart" caches designed to close this gap — LRB, 3L-Cache, LHD, GL-Cache — haven't seen widespread adoption. The reasons are both fundamental and practical: objective mismatch (better predictions don't always mean fewer misses), instability (reacting to noisy per-request signals), and overhead (hundreds of bytes of metadata per object and expensive inference on every cache miss).

We asked a simple question: **what if we don't learn a new eviction policy at all, but instead learn to configure a simple one?**

The result is **S4-FIFO** (Smart S3-FIFO), a cache eviction algorithm that augments a simple FIFO-based heuristic with a pre-trained machine learning model. S4-FIFO achieves a **26% higher miss ratio reduction** over S3-FIFO and **8% over 3L-Cache** — the best prior algorithm — while never increasing FIFO's miss ratio by more than **0.8%** on the worst-case workload. It does this at throughput parity with LRU and S3-FIFO, with a total learning overhead of just tens of kilobytes.

S4-FIFO will appear at OSDI 2026. Both the implementation and pre-trained model are open source.

## The Problem: Why Smart Caches Aren't Winning

To understand why S4-FIFO takes a different approach, it helps to look at where existing smart caches fall short.

We can classify smart eviction algorithms along two axes: **learning granularity** (do they reason about individual objects or the whole cache?) and **prediction frequency** (do they predict on every miss, or periodically?).

Most learned caches — LRB, 3L-Cache, LHD — operate at the **object level**, predicting reuse distance or utility scores for individual objects on **every miss**. This sounds natural but has three recurring problems:

1. **Objective mismatch.** These systems optimize surrogate metrics like L2 loss on reuse distance prediction. But a smaller prediction loss does not guarantee fewer cache misses. You can have a model with lower prediction error that makes worse eviction decisions.

2. **Instability and noise.** Per-request signals are noisy. Cache workloads exhibit huge variance at fine time granularity — the miss ratio measured over 10-second intervals can swing wildly. Reacting to every miss means chasing noise rather than learning signal.

3. **Overhead and opacity.** Object-level learning requires maintaining feature vectors (200+ bytes) per cached object and running model inference on every miss. This reduces throughput and makes the system harder to reason about — when miss ratio goes up, operators can't explain why.

On the other axis, **cache-level** algorithms like ARC and LeCaR tune global parameters (queue sizes, expert weights) rather than scoring individual objects. They avoid objective mismatch because their parameters directly affect miss ratio. But they update on **every miss** (every ghost hit), which makes them unstable — chasing individual ghost-hit signals rather than learning the workload's true structure.

There's one algorithm that uses **periodic** prediction: GL-Cache. But it still operates at the object level, and its predictions go stale quickly because the usefulness of object groups changes over time.

This analysis reveals an empty quadrant in the design space: **periodic, cache-level learning**. No one has explored it. And it offers an appealing combination — directly optimizing cache parameters for miss ratio reduction, without the noise and overhead of per-miss or per-object approaches.

S4-FIFO fills that gap.

## The Insight: Learn to Configure, Not to Evict

S4-FIFO is built on a principle we call **Learning-Augmented Heuristics (LAH)**: instead of embedding complex learning logic into the critical path, cleanly separate the cache's **data plane** from its **control plane**.

The data plane is a simple, parameterized heuristic — S3-FIFO extended with a few more knobs. It handles all reads and writes at full speed with deterministic O(1) operations. No model inference, no per-object metadata, no neural networks on the critical path.

The control plane runs asynchronously and infrequently. It collects lightweight cache-level features (hit-position histograms, queue hit ratios, workload characteristics) and feeds them to a pre-trained GBDT model that selects the best parameter configuration. This happens once after warmup — or periodically, say daily, in production — and the cache continues serving traffic uninterrupted.

Think of it like this: the heuristic is the engine, and the model is the driver adjusting the gears. You don't rebuild the engine for every hill — you just shift.

## How S4-FIFO Works

### The Data Plane: Four FIFO Queues

S4-FIFO extends S3-FIFO's three-queue design with an implicit **skip queue**:

- **Small FIFO**: Filters one-hit wonders and new arrivals. Newly inserted objects start here.
- **Main FIFO**: Retains popular objects via reinsertion on access. This is where the hot data lives.
- **Ghost FIFO**: Metadata-only queue that tracks recently evicted objects. A hit here means the cache should have kept that object.
- **Skip zone**: The first fraction of the small queue where frequency counters are not incremented. This prevents bursty correlated references from polluting the main queue — effectively acting as a fourth queue.

All operations — insert, lookup, evict, promote — are O(1). No sampling, no sorting, no complex data structures.

### The Five Learnable Knobs

S3-FIFO uses a fixed configuration: small queue at 10% of cache, ghost at 90%, promotion threshold of 2. S4-FIFO makes these parameters learnable:

| Parameter | Meaning | Default | Search Range |
|-----------|---------|---------|-------------|
| ρS (small queue ratio) | Fraction of cache for the small queue | 0.10 | 0.05 – 0.90 |
| ρG (ghost queue ratio) | Ghost queue size relative to cache | 0.90 | 0.90 – 6.00 |
| κ (skip ratio) | Fraction of small queue that skips counting | 0.00 | 0.00, 0.25 |
| τS (small-to-main threshold) | Hits needed to promote from small to main | 2 | 1, 2 |
| τG (ghost-to-main threshold) | Ghost hits needed to admit directly to main | 0 | 0, 1 |

We discretize these into 168 candidate configurations and then use a greedy set-cover algorithm to identify **18 representative parameter sets** — the ones that collectively cover the best configuration for the most workloads.

### The Control Plane: 73 Features, 1 Model, <2ms Inference

During warmup, S4-FIFO runs with default parameters and collects a compact **73-dimensional feature vector**:

- **60 histogram dimensions**: 20-bin hit-position distributions for each of the small, main, and ghost queues. These capture the shape of locality — where in each queue are hits concentrated? Too many hits at the tail of the small queue suggest the promotion threshold is too aggressive; too many ghost hits suggest the small queue is undersized.
- **3 queue ratio dimensions**: Proportion of total hits landing in each queue.
- **10 composite features**: Derived metrics like utility gap (divergence between small and main queue effectiveness), ghost pressure (how many evicted objects are being re-requested), thrashing risk (working set vs. cache capacity), scan intensity, one-hit ratio, and decay rate.

These features are collected with **O(1) overhead per access** — incrementing a histogram bin counter. No per-object metadata, no complex computation.

The model itself is a **LightGBM gradient-boosted decision tree** with 20 trees of depth 9. We chose GBDT over deep neural networks for two reasons: (1) it handles heterogeneous, nonlinear features well without normalization, and (2) it can be compiled into dependency-free conditional branches and exported to C/C++, Go, Rust, Java, and JavaScript. Our CacheLib prototype uses the same model trained in the simulator, compiled into a single C++ header file.

Inference takes less than 2 milliseconds and happens asynchronously off the critical path. The parameter switch doesn't require pausing the cache — we're just changing queue size targets and threshold values, not moving data.

### Training: A Foundation Model for Caching

We pre-train a single model on **4,140 production traces** from 14 sources spanning block storage, key-value stores, and CDN caches (including Tencent, Alibaba, Twitter, CloudPhysics, Meta Storage, Meta KV, and Meta CDN). The model performs **zero-shot prediction** — no per-deployment retraining needed.

This works because caching patterns are governed by universal structural behaviors — scanning, looping, thrashing — that transcend specific datasets. By extracting content-oblivious features (histograms and ratios rather than object IDs), the model learns to recognize workload patterns and map them to optimal configurations. It's effectively a **foundation model for caching**: knowledge from thousands of traces transfers to new, unseen environments.

The training objective is a **cost-sensitive classification** that explicitly penalizes configurations leading to worse miss ratios than FIFO. This FIFO-anchored cost function is key to S4-FIFO's robustness — the model learns that some mistakes (choosing a scan-hostile configuration for a scan-heavy workload) are far costlier than others.

## Results

We evaluated S4-FIFO on **1,035 held-out production traces**, comparing against S3-FIFO, 3L-Cache, LRB, LHD, GL-Cache, LeCaR, ARC, 2Q, and LIRS at cache sizes ranging from 0.1% to 10% of each trace's working set.

### Efficiency: +26% Over S3-FIFO, +8% Over 3L-Cache

At large cache sizes (10% of working set), S4-FIFO achieves a **26% higher mean miss ratio reduction over FIFO** compared to S3-FIFO, and **7-8% higher** than 3L-Cache. At small cache sizes (0.1% of working set), S4-FIFO is slightly behind 3L-Cache in mean efficiency but still substantially better than all other algorithms.

Notably, S4-FIFO with predicted parameters achieves nearly the same efficiency as **offline S4-FIFO** (grid-search optimal) — at most 0.2% difference in mean miss ratio reduction. The model is selecting nearly the best configuration most of the time.

Why does 3L-Cache have an edge at small cache sizes? When the cache is small relative to the working set, nearly all cached objects are highly popular, providing rich signals for object-level learning. But this advantage comes at enormous cost: 3L-Cache is **17.3× slower** on average in our simulator (up to 274× slower) and requires 200+ bytes of metadata per object.

### Robustness: 0.8% Worst Case

This is where S4-FIFO truly shines. Every other smart cache has pathological workloads where it performs worse than a simple FIFO:

- **LRB**: up to 72% worse than FIFO on its worst trace
- **LIRS**: up to 20% worse
- **3L-Cache**: 8.8% worse on its worst trace
- **2Q**: 4.3% worse

**S4-FIFO: only 0.8% worse than FIFO** on its worst-case workload at large cache sizes (0.2% at small cache sizes). This is not a coincidence — it's a direct consequence of three design choices:

1. Static FIFO queues avoid the pathological behaviors of adaptive algorithms.
2. Learning steers away from adversarial configurations.
3. The FIFO-anchored cost function explicitly optimizes for robustness.

For operators, this is the difference between "this algorithm might improve things" and "this algorithm will not break my system." That trust is essential for production adoption.

### Throughput: Parity with Heuristics

We measured throughput using **CacheBench** from Meta's CacheLib, comparing S4-FIFO against highly optimized implementations of LRU, 2Q, S3-FIFO, and TinyLFU. Even with continuous feature collection on every request (a worst-case scenario — in production, features would be collected periodically or via sampling), S4-FIFO sustains throughput comparable to S3-FIFO and outperforms LRU.

The key insight is that feature collection happens **outside the critical section** — incrementing a histogram counter doesn't affect cache correctness, so it doesn't contend with cache operations. TinyLFU, by contrast, must update its count-min sketch inside the critical section, causing significant throughput degradation.

### Why a Foundation Model Works

A natural question: can a single model really work across all these diverse workloads? We investigated this from several angles:

**Feature importance.** The histogram features contribute 75% of total importance in the trained model. The shape of hit distributions across queues — how locality manifests — is the strongest signal for predicting good configurations. This is content-oblivious: it captures structural patterns, not specific object IDs.

**Training data scaling.** Top-1 accuracy approaches 60%, top-3 near 80% with several thousand training traces. Accuracy continues to improve with more data, suggesting the model would benefit from even larger trace corpora.

**Cross-dataset generalization.** We trained a model on CDN2 (object cache traces) and evaluated it on Twitter (key-value traces) — a completely different cache type. The model generalizes effectively, confirming that the learned patterns are structural rather than dataset-specific.

**Interpretability.** Because both the features (hit distributions, ghost pressure, thrashing risk) and the outputs (queue sizes, promotion thresholds) have clear operational semantics, S4-FIFO's decisions are interpretable. An LLM can be prompted with the feature values and produce a reasonable explanation for why a particular configuration was chosen — something that's fundamentally impossible with opaque per-object scoring.

## Production Deployment

S4-FIFO runs in both libCacheSim for simulation and Meta's CacheLib for production evaluation. Both implementations share the same pre-trained model, exported from LightGBM to dependency-free C++ headers using m2cgen. No Python runtime, no ML framework dependencies — just a few hundred lines of conditional branches compiled into the cache binary.

The total storage overhead from learning is on the order of tens of kilobytes: the 73 global feature counters, the histogram arrays, and the compiled model. The dominant overhead remains the ghost queue metadata (8 bytes per entry for object ID and insertion timestamp), which is inherited from S3-FIFO.

In practice, S4-FIFO's deployment workflow is:

1. Start the cache with default parameters.
2. During warmup, collect features using the built-in feature collector.
3. After warmup (when the cache reaches steady state), run inference asynchronously.
4. Apply the predicted parameters — queue size targets and thresholds adjust seamlessly without pausing traffic.
5. Optionally, repeat periodically (e.g., daily) to adapt to workload shifts.

## Open Source

S4-FIFO is open source. The repository includes:

- **libCacheSim implementation** with S4-FIFO, S4-FIFO-base (minimal variant), and S4-FIFO-verify (verification with histograms)
- **CacheLib integration** for production deployment
- **Model training pipeline** (LightGBM with cost-sensitive classification)
- **Analysis scripts** for feature engineering, evaluation, and plotting
- **Pre-trained model** exported to C/C++, Python, Go, Rust, Java, and JavaScript
- **Docker-based artifact evaluation package** for reproducing all results

We believe learning-augmented heuristics represent a practical path to deploying ML-driven caching in production. By keeping the data plane simple and fast, and moving intelligence to an asynchronous control plane, S4-FIFO achieves the efficiency of learned caches without sacrificing the robustness and performance that operators require.

Try it on your workloads — we'd love to hear what you find.

---

*Haocheng Xia is a PhD student at Harvard and UIUC. Pranav Bhandari is an engineer at Meta. Juncheng Yang is a PhD student at Harvard. This work will appear at OSDI 2026.*
