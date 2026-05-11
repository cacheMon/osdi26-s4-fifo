# Analysis Scripts

This directory contains the model training, feature analysis, and evaluation scripts for S4-FIFO.

## Directory Structure

```
analysis/
├── cleaned/              # Cleaned evaluation results and figures
│   ├── 0.001/            # Results at 0.1% cache ratio
│   ├── 0.01/             # Results at 1% cache ratio
│   ├── 0.1/              # Results at 10% cache ratio
│   ├── feature_distribution/
│   └── individual_plots/ # Per-trace comparison plots
├── gpu/                  # GPU-accelerated training scripts
│   └── train_deep_classifier.py   # Neural network classifier (PyTorch)
├── feature_order_search/ # Feature order ablation study
├── xgb_18class_results/  # XGBoost 18-class model results
├── xgb_results/          # XGBoost training results
├── features_*.csv        # Feature datasets (73-dim features per trace)
└── *.py                  # Training and evaluation scripts
```

## Key Scripts

### Model Training

| Script | Description |
|--------|-------------|
| `train_xgb_18class.py` | Train XGBoost 18-class configuration selector |
| `train_xgb_18class_lite_logc.py` | Lightweight XGBoost with log-cost loss |
| `train_xgb_18class_wss.py` | XGBoost with working set size features |
| `train_xgb_joint_rmi.py` | Joint regression + multi-input model |
| `train_predict_params.py` | Direct parameter prediction (regression) |
| `train_per_ratio.py` | Train separate models per cache size ratio |
| `gpu/train_deep_classifier.py` | Neural network classifier (PyTorch) |

### Feature Analysis

| Script | Description |
|--------|-------------|
| `export_feature_importance.py` | Export XGBoost feature importance rankings |
| `export_models_multilang.py` | Export trained models to multiple formats |
| `plot_feature_distribution.py` | Plot feature distributions across datasets |
| `plot_feature_importance.py` | Visualize feature importance |

### Evaluation

| Script | Description |
|--------|-------------|
| `evaluate_model.py` | Evaluate trained model on test traces |
| `cross_dataset_simple.py` | Cross-dataset generalization test |
| `cross_dataset_optimized.py` | Optimized cross-dataset evaluation |
| `check_tail_performance.py` | Analyze tail (worst-case) performance |
| `analyze_class_distribution.py` | Analyze predicted class distribution |
| `learning_curve_experiment.py` | Learning curve (data efficiency) experiment |
| `create_improvement_report.py` | Generate improvement report over baselines |
| `create_grid_optimal.py` | Create grid-search optimal configurations |
| `inspect_traces.py` | Inspect and validate trace files |
| `test_c_consistency.py` | Test C implementation consistency |

### Parameter Search

| Script | Description |
|--------|-------------|
| `xgb_predict_params.py` | Predict parameters using XGBoost model |
| `find_best_order.py` | Find optimal feature ordering |

### Visualization

| Script | Description |
|--------|-------------|
| `plot_cleaned_data.py` | Plot cleaned evaluation data |
| `plot_individual_figures_wss.py` | Individual trace plots (working set size) |
| `learning_curve_v2_replot.py` | Re-plot learning curve with updated data |

## Data Files

| File | Description |
|------|-------------|
| `features_10pct.csv` | 73-dim features extracted at 10% cache ratio |
| `features_20pct.csv` | 73-dim features extracted at 20% cache ratio |
| `features_20pct_with_meta.csv` | Features with trace metadata |
| `aggregated_results.csv` | Aggregated simulation results |
| `cleaned/*.csv` | Per-ratio evaluation statistics |

## Usage

```bash
# Install dependencies
pip install -r ../requirements.txt

# Train XGBoost 18-class model
python3 train_xgb_18class.py

# Evaluate model
python3 evaluate_model.py

# Generate plots
python3 plot_feature_importance.py
```

## Feature Engineering

S4-FIFO uses 73 features per trace, computed from a short warmup simulation:

- **Histogram features (60 dims)**: 20-bin histograms for each of the 3 FIFO queues
- **Queue hit ratios (3 dims)**: small/main/ghost hit ratios
- **Composite features (10 dims)**: utility gap, filtering efficiency, ghost pressure, etc.

The feature extraction is implemented in `libCacheSim/libCacheSim/cache/eviction/S4FIFO_features.h`.

## Model Details

- **Algorithm**: XGBoost (GBDT)
- **Input**: 73 features per trace
- **Output**: 18-class configuration label
- **Training data**: 4,140 production traces
- **Configuration space**: 168 grid-search configurations reduced to 18 classes via greedy set cover
