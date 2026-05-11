#!/usr/bin/env python3
"""
Improved XGBoost model to predict optimal S4FIFO parameters.
Goal: Match Ground Truth performance through comprehensive feature engineering.

Strategy:
1. Predict optimal parameters (s_param, m_param, t_param, g_param) 
2. Look up actual miss_ratio from grid_full based on predicted parameters
3. Focus on tail cases relative to FIFO using custom loss weighting
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_data():
    """Load features, grid_full, and grid_optimal"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load features
    features_list = []
    for pct in [5, 10, 20]:
        feat = pd.read_csv(os.path.join(script_dir, f'features_{pct}pct.csv'))
        features_list.append(feat)
    
    features_df = pd.concat(features_list, ignore_index=True)
    print(f"Loaded features: {len(features_df)} samples, {features_df['trace'].nunique()} traces")
    
    # Load grid_optimal (labels)
    cleaned_dir = os.path.join(script_dir, 'cleaned')
    optimal_list = []
    grid_full_list = []
    fifo_list = []
    
    for ratio_str in ['0.001', '0.01', '0.1']:
        # Optimal labels
        opt_path = os.path.join(cleaned_dir, ratio_str, 'grid_optimal.csv')
        opt = pd.read_csv(opt_path)
        optimal_list.append(opt)
        
        # Grid full for evaluation
        full_path = os.path.join(cleaned_dir, ratio_str, 'grid_full.csv')
        full = pd.read_csv(full_path)
        grid_full_list.append(full)
        
        # FIFO baseline
        base_path = os.path.join(cleaned_dir, ratio_str, 'baselines.csv')
        baselines = pd.read_csv(base_path)
        fifo = baselines[baselines['algorithm'] == 'FIFO'][['trace', 'ratio', 'miss_ratio']].copy()
        fifo = fifo.rename(columns={'miss_ratio': 'fifo_miss_ratio'})
        fifo_list.append(fifo)
    
    optimal_df = pd.concat(optimal_list, ignore_index=True)
    grid_full_df = pd.concat(grid_full_list, ignore_index=True)
    fifo_df = pd.concat(fifo_list, ignore_index=True)
    
    print(f"Loaded optimal labels: {len(optimal_df)} samples")
    print(f"Loaded grid_full: {len(grid_full_df)} configurations")
    print(f"Loaded FIFO baseline: {len(fifo_df)} samples")
    
    return features_df, optimal_df, grid_full_df, fifo_df

def engineer_features(df):
    """Enhanced feature engineering"""
    df = df.copy()
    
    # Basic derived features
    df['H_total'] = df['H_g'] + df['H_m'] + df['H_s']
    df['H_ratio_gm'] = df['H_g'] / (df['H_m'] + 1e-10)
    df['H_ratio_sm'] = df['H_s'] / (df['H_m'] + 1e-10)
    df['H_ratio_gs'] = df['H_g'] / (df['H_s'] + 1e-10)
    
    # Hit ratios
    df['total_hit_count'] = df['hits_ghost'] + df['hits_main'] + df['hits_small']
    df['hit_ratio_ghost'] = df['hits_ghost'] / (df['total_hit_count'] + 1e-10)
    df['hit_ratio_main'] = df['hits_main'] / (df['total_hit_count'] + 1e-10)
    df['hit_ratio_small'] = df['hits_small'] / (df['total_hit_count'] + 1e-10)
    
    # Overall metrics
    df['overall_hit_rate'] = df['total_hits'] / (df['total_reqs'] + 1e-10)
    df['overall_miss_rate'] = df['total_misses'] / (df['total_reqs'] + 1e-10)
    
    # Reuse features
    df['rho_ratio'] = df['rho_onehit'] / (df['rho_unique'] + 1e-10)
    df['log_rho_onehit'] = np.log1p(df['rho_onehit'])
    df['log_rho_unique'] = np.log1p(df['rho_unique'])
    
    # Histogram statistics
    ghost_cols = [c for c in df.columns if c.startswith('hist_ghost_')]
    main_cols = [c for c in df.columns if c.startswith('hist_main_')]
    small_cols = [c for c in df.columns if c.startswith('hist_small_')]
    
    for prefix, cols in [('ghost', ghost_cols), ('main', main_cols), ('small', small_cols)]:
        df[f'{prefix}_hist_mean'] = df[cols].mean(axis=1)
        df[f'{prefix}_hist_std'] = df[cols].std(axis=1)
        df[f'{prefix}_hist_max'] = df[cols].max(axis=1)
        df[f'{prefix}_hist_min'] = df[cols].min(axis=1)
        df[f'{prefix}_hist_skew'] = df[cols].skew(axis=1)
        
        # Early vs late
        early = [c for c in cols if int(c.split('_')[-1]) < 10]
        late = [c for c in cols if int(c.split('_')[-1]) >= 10]
        df[f'{prefix}_early_sum'] = df[early].sum(axis=1)
        df[f'{prefix}_late_sum'] = df[late].sum(axis=1)
        df[f'{prefix}_early_late_ratio'] = df[f'{prefix}_early_sum'] / (df[f'{prefix}_late_sum'] + 1e-10)
    
    # Combined features (small + ghost)
    for i in range(20):
        df[f'sg_combined_{i}'] = df[f'hist_small_{i}'] + df[f'hist_ghost_{i}']
        df[f'sg_ratio_{i}'] = df[f'hist_small_{i}'] / (df[f'hist_ghost_{i}'] + 1e-10)
    
    # Weighted history (more recent = more important)
    weights = np.exp(-np.arange(20) / 5.0)
    for prefix, cols in [('ghost', ghost_cols), ('main', main_cols), ('small', small_cols)]:
        weighted = sum(df[col] * weights[int(col.split('_')[-1])] for col in cols)
        df[f'{prefix}_weighted'] = weighted
    
    # Ratio features
    df['log_ratio'] = np.log10(df['ratio'])
    df['sqrt_ratio'] = np.sqrt(df['ratio'])
    df['log_C_squared'] = df['log_C'] ** 2
    df['log_C_ratio'] = df['log_C'] * df['log_ratio']
    
    # Interaction features
    df['hits_ghost_entropy'] = df['hits_ghost'] * df['H_g']
    df['hits_main_entropy'] = df['hits_main'] * df['H_m']
    df['hits_small_entropy'] = df['hits_small'] * df['H_s']
    
    print(f"Feature engineering: {len(df.columns)} total features")
    return df

def prepare_data(features_df, optimal_df, fifo_df):
    """Merge and prepare training data"""
    # Merge features with optimal parameters
    merged = pd.merge(features_df, optimal_df, on=['trace', 'ratio'], how='inner')
    print(f"After merging features + optimal: {len(merged)} samples")
    
    # Merge with FIFO
    merged = pd.merge(merged, fifo_df, on=['trace', 'ratio'], how='inner')
    print(f"After merging with FIFO: {len(merged)} samples")
    
    # Calculate tail weight (worse FIFO = higher weight)
    merged['tail_weight'] = np.clip(merged['fifo_miss_ratio'] / merged['fifo_miss_ratio'].median(), 0.5, 10.0)
    
    return merged

def split_by_trace(df, test_size=0.2, random_state=42):
    """Split ensuring no trace overlap"""
    unique_traces = df['trace'].unique()
    np.random.seed(random_state)
    
    shuffled = np.random.permutation(unique_traces)
    split_idx = int(len(shuffled) * (1 - test_size))
    
    train_traces = shuffled[:split_idx]
    test_traces = shuffled[split_idx:]
    
    train_df = df[df['trace'].isin(train_traces)].copy()
    test_df = df[df['trace'].isin(test_traces)].copy()
    
    print(f"\nTrain: {len(train_traces)} traces, {len(train_df)} samples")
    print(f"Test: {len(test_traces)} traces, {len(test_df)} samples")
    print(f"No overlap: {len(set(train_traces) & set(test_traces)) == 0}")
    
    return train_df, test_df

def train_param_models(train_df, feature_cols, param_targets=['s_param', 'm_param', 't_param', 'g_param']):
    """Train separate XGBoost models for each parameter"""
    models = {}
    label_encoders = {}
    
    X_train = train_df[feature_cols]
    weights = train_df['tail_weight'].values
    
    for param in param_targets:
        print(f"\nTraining model for {param}...")
        y_train = train_df[param]
        
        # Check if classification or regression
        unique_vals = y_train.nunique()
        print(f"  Unique values: {unique_vals}")
        print(f"  Value distribution: {y_train.value_counts().head()}")
        
        if unique_vals <= 10:  # Classification
            # Encode labels to 0, 1, 2, ...
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            label_encoders[param] = le
            
            params = {
                'objective': 'multi:softprob',
                'num_class': unique_vals,
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train_encoded, sample_weight=weights, verbose=False)
        else:  # Regression
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, sample_weight=weights, verbose=False)
        
        models[param] = model
        print(f"  Model trained")
    
    return models, label_encoders

def predict_and_evaluate(models, label_encoders, test_df, feature_cols, grid_full_df, fifo_df):
    """Predict parameters and evaluate performance"""
    X_test = test_df[feature_cols]
    
    # Predict each parameter
    predicted_params = {}
    for param, model in models.items():
        if param in label_encoders:  # Classification with label encoding
            pred_encoded = model.predict(X_test)
            pred = label_encoders[param].inverse_transform(pred_encoded)
        elif hasattr(model, 'predict_proba'):  # Classification without encoding
            pred = model.predict(X_test)
        else:  # Regression
            pred = model.predict(X_test)
            # Round/clip to valid ranges
            if param == 's_param':
                pred = np.clip(np.round(pred, 2), 0.05, 0.9)
            elif param == 'm_param':
                pred = np.clip(np.round(pred), 1, 2).astype(int)
            elif param == 't_param':
                pred = np.clip(np.round(pred), 0, 1).astype(int)
            elif param == 'g_param':
                pred = np.clip(pred, 0.9, 6.0)
        
        predicted_params[param] = pred
        
        # Accuracy for discrete params
        actual = test_df[param].values
        if param in ['s_param', 'm_param', 't_param', 'g_param']:
            if param in ['m_param', 't_param']:
                acc = (pred == actual).mean()
            else:
                acc = (np.abs(pred - actual) < 0.01).mean()  # Allow small float diff
            print(f"{param} accuracy: {acc:.2%}")
    
    # Create results dataframe
    results = test_df[['trace', 'ratio', 's_param', 'm_param', 't_param', 'g_param', 'miss_ratio', 'fifo_miss_ratio']].copy()
    results = results.rename(columns={'miss_ratio': 'optimal_miss_ratio'})
    
    for param in predicted_params:
        results[f'pred_{param}'] = predicted_params[param]
    
    # Look up miss_ratio from grid_full for predicted parameters
    print("\nLooking up predicted miss ratios from grid_full...")
    predicted_mrs = []
    
    for idx, row in results.iterrows():
        # Find matching configuration in grid_full
        mask = (
            (grid_full_df['trace'] == row['trace']) &
            (grid_full_df['ratio'] == row['ratio']) &
            (grid_full_df['s_param'] == row['pred_s_param']) &
            (grid_full_df['m_param'] == row['pred_m_param']) &
            (grid_full_df['t_param'] == row['pred_t_param']) &
            (np.abs(grid_full_df['g_param'] - row['pred_g_param']) < 0.01)
        )
        
        matches = grid_full_df[mask]
        
        if len(matches) > 0:
            predicted_mrs.append(matches.iloc[0]['miss_ratio'])
        else:
            # If exact match not found, find closest configuration
            trace_grid = grid_full_df[
                (grid_full_df['trace'] == row['trace']) &
                (grid_full_df['ratio'] == row['ratio'])
            ]
            
            if len(trace_grid) > 0:
                # Find closest by parameter distance
                trace_grid = trace_grid.copy()
                trace_grid['dist'] = (
                    (trace_grid['s_param'] - row['pred_s_param'])**2 +
                    (trace_grid['m_param'] - row['pred_m_param'])**2 +
                    (trace_grid['t_param'] - row['pred_t_param'])**2 +
                    ((trace_grid['g_param'] - row['pred_g_param'])/5.0)**2
                )
                closest = trace_grid.nsmallest(1, 'dist')
                predicted_mrs.append(closest.iloc[0]['miss_ratio'])
            else:
                predicted_mrs.append(np.nan)
    
    results['predicted_miss_ratio'] = predicted_mrs
    
    # Remove NaNs
    results = results.dropna()
    print(f"Valid predictions: {len(results)} / {len(test_df)}")
    
    # Calculate metrics
    results['optimal_vs_fifo_pct'] = (results['fifo_miss_ratio'] - results['optimal_miss_ratio']) / results['fifo_miss_ratio'] * 100
    results['predicted_vs_fifo_pct'] = (results['fifo_miss_ratio'] - results['predicted_miss_ratio']) / results['fifo_miss_ratio'] * 100
    results['gap_to_optimal_pct'] = (results['predicted_miss_ratio'] - results['optimal_miss_ratio']) / results['optimal_miss_ratio'] * 100
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOverall Performance vs FIFO:")
    print(f"  Ground Truth: {results['optimal_vs_fifo_pct'].mean():.2f}% ± {results['optimal_vs_fifo_pct'].std():.2f}%")
    print(f"  Predicted: {results['predicted_vs_fifo_pct'].mean():.2f}% ± {results['predicted_vs_fifo_pct'].std():.2f}%")
    
    print(f"\nTail Cases:")
    print(f"  Ground Truth P99: {np.percentile(results['optimal_vs_fifo_pct'], 1):.2f}%")
    print(f"  Predicted P99: {np.percentile(results['predicted_vs_fifo_pct'], 1):.2f}%")
    print(f"  Ground Truth worst: {results['optimal_vs_fifo_pct'].min():.2f}%")
    print(f"  Predicted worst: {results['predicted_vs_fifo_pct'].min():.2f}%")
    
    print(f"\nGap to Ground Truth:")
    print(f"  Mean: {results['gap_to_optimal_pct'].mean():.2f}%")
    print(f"  Median: {results['gap_to_optimal_pct'].median():.2f}%")
    print(f"  P90: {np.percentile(results['gap_to_optimal_pct'], 90):.2f}%")
    print(f"  P99: {np.percentile(results['gap_to_optimal_pct'], 99):.2f}%")
    
    print(f"\nBy Cache Ratio:")
    for ratio in sorted(results['ratio'].unique()):
        ratio_data = results[results['ratio'] == ratio]
        print(f"  Ratio {ratio}:")
        print(f"    GT vs FIFO: {ratio_data['optimal_vs_fifo_pct'].mean():.2f}%")
        print(f"    Pred vs FIFO: {ratio_data['predicted_vs_fifo_pct'].mean():.2f}%")
        print(f"    Gap to GT: {ratio_data['gap_to_optimal_pct'].mean():.2f}% (median: {ratio_data['gap_to_optimal_pct'].median():.2f}%)")
    
    return results

def plot_results(results, output_dir):
    """Create visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Performance comparison by ratio
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ratios = sorted(results['ratio'].unique())
    x = np.arange(len(ratios))
    width = 0.35
    
    gt_means = [results[results['ratio'] == r]['optimal_vs_fifo_pct'].mean() for r in ratios]
    pred_means = [results[results['ratio'] == r]['predicted_vs_fifo_pct'].mean() for r in ratios]
    
    bars1 = ax.bar(x - width/2, gt_means, width, label='Ground Truth', color='#FFD700', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, pred_means, width, label='Predicted', color='#1f77b4', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Cache Ratio', fontsize=14, fontweight='bold')
    ax.set_ylabel('Improvement over FIFO (%)', fontsize=14, fontweight='bold')
    ax.set_title('Parameter Prediction Performance', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ratios)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_ratio.png'), dpi=300)
    print(f"\nSaved: {os.path.join(output_dir, 'performance_by_ratio.png')}")
    plt.close()
    
    # 2. Gap distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, ratio in enumerate(sorted(results['ratio'].unique())):
        ratio_data = results[results['ratio'] == ratio]
        
        axes[idx].hist(ratio_data['gap_to_optimal_pct'], bins=50, alpha=0.7, edgecolor='black')
        axes[idx].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[idx].set_xlabel('Gap to Ground Truth (%)', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'Ratio {ratio}', fontsize=14, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gap_distribution.png'), dpi=300)
    print(f"Saved: {os.path.join(output_dir, 'gap_distribution.png')}")
    plt.close()

def main():
    print("="*80)
    print("XGBoost Parameter Prediction for Optimal S4FIFO Configuration")
    print("="*80)
    
    # 1. Load data
    print("\n[1] Loading data...")
    features_df, optimal_df, grid_full_df, fifo_df = load_all_data()
    
    # 2. Feature engineering
    print("\n[2] Feature engineering...")
    features_eng = engineer_features(features_df)
    
    # 3. Prepare data
    print("\n[3] Preparing data...")
    merged_df = prepare_data(features_eng, optimal_df, fifo_df)
    
    # 4. Split
    print("\n[4] Splitting train/test...")
    train_df, test_df = split_by_trace(merged_df, test_size=0.2)
    
    # 5. Get feature columns
    exclude_cols = ['trace', 'ratio', 's_param', 'm_param', 't_param', 'g_param', 'k_param',
                   'miss_ratio', 'fifo_miss_ratio', 'tail_weight']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"\n[5] Training with {len(feature_cols)} features")
    
    # 6. Train models for each parameter
    print("\n[6] Training parameter models...")
    models, label_encoders = train_param_models(train_df, feature_cols)
    
    # 7. Evaluate
    print("\n[7] Evaluating...")
    results = predict_and_evaluate(models, label_encoders, test_df, feature_cols, grid_full_df, fifo_df)
    
    # 8. Save results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgb_param_results')
    os.makedirs(output_dir, exist_ok=True)
    
    results.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print(f"\nSaved: {os.path.join(output_dir, 'predictions.csv')}")
    
    # 9. Visualize
    print("\n[8] Creating visualizations...")
    plot_results(results, output_dir)
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)

if __name__ == '__main__':
    main()
