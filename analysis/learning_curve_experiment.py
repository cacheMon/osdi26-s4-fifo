#!/usr/bin/env python3
"""
Learning Curve Experiment: Fixed Test Set, Varying Training Size
Measures Top-1, Top-2, Top-3 accuracy for 18-class XGBoost classifier.
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# Configuration from train_xgb_18class.py
SELECTED_CONFIGS = [
    (0.20, 1, 0, 3.0), (0.05, 1, 0, 0.9), (0.50, 1, 0, 0.9), (0.20, 1, 0, 0.9),
    (0.05, 2, 0, 6.0), (0.10, 2, 1, 3.0), (0.30, 2, 0, 3.0), (0.05, 2, 0, 3.0),
    (0.10, 2, 0, 0.9), (0.70, 1, 1, 0.9), (0.20, 1, 1, 0.9), (0.05, 1, 1, 0.9),
    (0.30, 1, 0, 6.0), (0.20, 2, 0, 0.9), (0.90, 2, 0, 3.0), (0.10, 2, 0, 6.0),
    (0.30, 2, 1, 3.0), (0.05, 2, 0, 0.9),
]
CONFIG_TO_ID = {c: i for i, c in enumerate(SELECTED_CONFIGS)}

# Font settings for paper
plt.rcParams.update({
    'font.size': 18, 'axes.labelsize': 22, 'xtick.labelsize': 18,
    'ytick.labelsize': 18, 'legend.fontsize': 16, 'font.weight': 'bold',
    'axes.labelweight': 'bold',
})


def normalize_trace(t):
    t = str(t)
    if '.oracleGeneral' in t:
        t = t.split('.oracleGeneral')[0]
    for ext in ['.zst', '.csv', '.txt']:
        if t.endswith(ext): t = t[:-len(ext)]
    return t


def load_all_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    features_df = pd.read_csv(os.path.join(script_dir, 'features_20pct.csv'))
    features_df['trace'] = features_df['trace'].apply(normalize_trace)
    
    cleaned_dir = os.path.join(script_dir, 'cleaned')
    optimal_list, grid_full_list, fifo_list = [], [], []
    
    for ratio_str in ['0.001', '0.01', '0.1']:
        opt = pd.read_csv(os.path.join(cleaned_dir, ratio_str, 'grid_optimal.csv'))
        opt['trace'] = opt['trace'].apply(normalize_trace)
        optimal_list.append(opt)
        
        full = pd.read_csv(os.path.join(cleaned_dir, ratio_str, 'grid_full.csv'))
        full['trace'] = full['trace'].apply(normalize_trace)
        grid_full_list.append(full)
        
        baselines = pd.read_csv(os.path.join(cleaned_dir, ratio_str, 'baselines.csv'))
        fifo = baselines[baselines['algorithm'] == 'FIFO'][['trace', 'ratio', 'miss_ratio']].copy()
        fifo['trace'] = fifo['trace'].apply(normalize_trace)
        fifo = fifo.rename(columns={'miss_ratio': 'fifo_miss_ratio'})
        fifo_list.append(fifo)
    
    return (pd.concat(optimal_list, ignore_index=True), 
            pd.concat(grid_full_list, ignore_index=True),
            pd.concat(fifo_list, ignore_index=True),
            features_df)


def engineer_features(df, ratio_col=None):
    df = df.copy()
    df['probation_efficiency'] = df['hits_small'] / (df['hits_main'] + 1e-6)
    df['total_hits_with_ghost'] = df['total_hits'] + df['hits_ghost']
    df['ghost_pressure'] = df['hits_ghost'] / (df['total_hits_with_ghost'] + 1e-6)
    df['entropy_gap'] = df['H_m'] - df['H_s']
    
    if 'hist_small_0' in df.columns and 'hist_small_1' in df.columns:
        df['decay_rate_small'] = df['hist_small_0'] - df['hist_small_1']
    
    main_tail_cols = [f'hist_main_{i}' for i in range(10, 20)]
    existing_tail_cols = [c for c in main_tail_cols if c in df.columns]
    if existing_tail_cols:
        df['tail_heaviness'] = df[existing_tail_cols].sum(axis=1)
    
    if ratio_col and ratio_col in df.columns:
        df['thrashing_risk'] = df['rho_unique'] / (df[ratio_col] * 100 + 1e-6)
        df['scan_intensity'] = df['rho_onehit'] * (1.0 - df[ratio_col])
    
    return df


def prepare_dataset():
    """Load and prepare full dataset. Returns tuned_traces as separate set."""
    print("Loading data...")
    optimal_df, grid_full_df, fifo_df, features_df = load_all_data()
    
    # Load tuned traces (1022 unique traces for test set)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tuned_df = pd.read_csv(os.path.join(script_dir, 'xgb_18class_results/tune.csv'))
    tuned_traces = set(tuned_df['trace'].apply(normalize_trace).unique())
    print(f"Tuned traces (test set): {len(tuned_traces)}")
    
    # Engineer features on ALL traces
    features_eng = engineer_features(features_df)
    
    # Merge ALL data
    merged = pd.merge(features_eng, optimal_df, on=['trace', 'ratio'], how='inner')
    merged = pd.merge(merged, fifo_df, on=['trace', 'ratio'], how='inner')
    
    # Post-merge features
    merged['thrashing_risk'] = merged['rho_unique'] / (merged['ratio'] * 100 + 1e-6)
    merged['scan_intensity'] = merged['rho_onehit'] * (1.0 - merged['ratio'])
    
    # Find best selected config
    grid_full_df['config'] = list(zip(
        grid_full_df['s_param'], 
        grid_full_df['m_param'].astype(int),
        grid_full_df['t_param'].astype(int), 
        grid_full_df['g_param'].round(1)
    ))
    
    selected_set = set(SELECTED_CONFIGS)
    grid_selected = grid_full_df[grid_full_df['config'].apply(lambda c: c in selected_set)]
    
    best_configs = grid_selected.groupby(['trace', 'ratio']).apply(
        lambda x: x.loc[x['miss_ratio'].idxmin(), 'config'], include_groups=False
    ).reset_index(name='best_config')
    
    merged = pd.merge(merged, best_configs, on=['trace', 'ratio'], how='left')
    merged['class_id'] = merged['best_config'].map(CONFIG_TO_ID)
    merged = merged.dropna(subset=['class_id'])
    merged['class_id'] = merged['class_id'].astype(int)
    
    # Feature columns
    keep_features = [
        'probation_efficiency', 'ghost_pressure', 'entropy_gap', 
        'decay_rate_small', 'tail_heaviness', 'thrashing_risk', 'scan_intensity',
        'rho_unique', 'rho_onehit', 'total_reqs', 'H_m', 'H_s', 'H_g',
    ]
    hist_cols = [c for c in merged.columns if 'hist_' in c]
    feature_cols = list(set(keep_features + hist_cols) & set(merged.columns))
    
    all_traces = set(merged['trace'].unique())
    print(f"Total samples: {len(merged)}, Features: {len(feature_cols)}")
    print(f"All unique traces: {len(all_traces)}, Training pool: {len(all_traces - tuned_traces)}")
    
    # Build miss ratio lookup for tolerance
    grid_selected['class_id'] = grid_selected['config'].map(CONFIG_TO_ID)
    mr_lookup = {}
    for _, row in grid_selected.iterrows():
        if not pd.isna(row['class_id']):
            key = (row['trace'], row['ratio'], int(row['class_id']))
            mr_lookup[key] = row['miss_ratio']
    
    return merged, feature_cols, tuned_traces, mr_lookup


def run_experiment(merged, feature_cols, tuned_traces, mr_lookup, train_fractions, n_runs=3):
    """Run learning curve experiment with fixed test set (1022 tuned traces).
    
    Uses miss ratio tolerance for all Top-k: if predicted config has same MR as optimal, count as correct.
    """
    
    # Test set = 1022 tuned traces (FIXED)
    # Training pool = all OTHER traces
    all_traces = set(merged['trace'].unique())
    test_traces = tuned_traces
    train_pool_traces = list(all_traces - test_traces)
    
    test_df = merged[merged['trace'].isin(test_traces)].reset_index(drop=True)
    train_pool_df = merged[merged['trace'].isin(train_pool_traces)]
    
    X_test = np.nan_to_num(test_df[feature_cols].values, nan=0.0)
    y_test = test_df['class_id'].values
    
    print(f"\nFixed test set (tuned traces): {len(test_df)} samples ({len(test_traces)} traces)")
    print(f"Training pool (other traces): {len(train_pool_df)} samples ({len(train_pool_traces)} traces)")
    
    def calc_topk_with_tolerance(probs, y_true, test_df, mr_lookup):
        """Calculate Top-k accuracy with miss ratio tolerance for ALL k."""
        sorted_idx = np.argsort(-probs, axis=1)
        
        top1_correct, top2_correct, top3_correct = 0, 0, 0
        
        for i in range(len(y_true)):
            true_class = y_true[i]
            trace = test_df.iloc[i]['trace']
            ratio = test_df.iloc[i]['ratio']
            true_mr = round(mr_lookup.get((trace, ratio, true_class), 999), 4)
            
            found_at_k = None
            for k in range(3):  # Check top 3
                pred = sorted_idx[i, k]
                pred_mr = round(mr_lookup.get((trace, ratio, pred), 998 - k), 4)
                
                if pred == true_class or pred_mr == true_mr:
                    found_at_k = k
                    break
            
            if found_at_k is not None:
                if found_at_k == 0:
                    top1_correct += 1
                    top2_correct += 1
                    top3_correct += 1
                elif found_at_k == 1:
                    top2_correct += 1
                    top3_correct += 1
                else:  # found_at_k == 2
                    top3_correct += 1
        
        total = len(y_true)
        return 100 * top1_correct / total, 100 * top2_correct / total, 100 * top3_correct / total
    
    results = []
    
    for frac in train_fractions:
        n_train_traces = max(1, int(len(train_pool_traces) * frac))
        
        top1_list, top2_list, top3_list = [], [], []
        
        for run in range(n_runs):
            # Sample training traces
            np.random.seed(42 + run * 100)
            sampled_traces = np.random.choice(train_pool_traces, n_train_traces, replace=False)
            train_df = train_pool_df[train_pool_df['trace'].isin(sampled_traces)]
            
            X_train = np.nan_to_num(train_df[feature_cols].values, nan=0.0)
            y_train = train_df['class_id'].values
            
            # Train ensemble (smaller for speed)
            n_models = 10
            probs = np.zeros((len(X_test), len(SELECTED_CONFIGS)))
            
            for i in range(n_models):
                model = lgb.LGBMClassifier(
                    objective='multiclass',
                    num_class=len(SELECTED_CONFIGS),
                    num_leaves=31 + i * 5,
                    max_depth=6 + i % 4,
                    learning_rate=0.05,
                    n_estimators=200,
                    random_state=42 + i * 11 + run * 1000,
                    verbose=-1
                )
                model.fit(X_train, y_train)
                
                # Handle case where training set is missing some classes
                pred_probs = model.predict_proba(X_test)
                if pred_probs.shape[1] < len(SELECTED_CONFIGS):
                    # Pad with zeros for missing classes
                    full_probs = np.zeros((len(X_test), len(SELECTED_CONFIGS)))
                    for j, cls in enumerate(model.classes_):
                        full_probs[:, cls] = pred_probs[:, j]
                    probs += full_probs
                else:
                    probs += pred_probs
            
            probs /= n_models
            
            # Calculate top-k accuracy WITH TOLERANCE
            top1, top2, top3 = calc_topk_with_tolerance(probs, y_test, test_df, mr_lookup)
            
            top1_list.append(top1)
            top2_list.append(top2)
            top3_list.append(top3)
        
        # Average over runs
        result = {
            'train_fraction': frac,
            'n_train_traces': n_train_traces,
            'n_train_samples': len(train_df),
            'top1_mean': np.mean(top1_list),
            'top1_std': np.std(top1_list),
            'top2_mean': np.mean(top2_list),
            'top2_std': np.std(top2_list),
            'top3_mean': np.mean(top3_list),
            'top3_std': np.std(top3_list),
        }
        results.append(result)
        
        print(f"Train {frac*100:.0f}% ({n_train_traces} traces): "
              f"Top1={result['top1_mean']:.1f}±{result['top1_std']:.1f}%, "
              f"Top2={result['top2_mean']:.1f}±{result['top2_std']:.1f}%, "
              f"Top3={result['top3_mean']:.1f}±{result['top3_std']:.1f}%")
    
    return pd.DataFrame(results)


def plot_results(results_df, output_dir):
    """Plot learning curve."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = results_df['n_train_traces'].values
    
    # Plot with error bars
    ax.errorbar(x, results_df['top1_mean'], yerr=results_df['top1_std'], 
                marker='o', markersize=10, linewidth=2, capsize=5, label='Top-1')
    ax.errorbar(x, results_df['top2_mean'], yerr=results_df['top2_std'], 
                marker='s', markersize=10, linewidth=2, capsize=5, label='Top-2')
    ax.errorbar(x, results_df['top3_mean'], yerr=results_df['top3_std'], 
                marker='^', markersize=10, linewidth=2, capsize=5, label='Top-3')
    
    ax.set_xlabel('Number of Training Traces')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir}/learning_curve.png")
    
    # Save data
    results_df.to_csv(f'{output_dir}/learning_curve_data.csv', index=False)
    print(f"Saved: {output_dir}/learning_curve_data.csv")


def main():
    print("="*60)
    print("Learning Curve Experiment: 18-Class XGBoost")
    print("="*60)
    
    merged, feature_cols, tuned_traces, mr_lookup = prepare_dataset()
    
    # Training fractions to try
    train_fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    print("\nRunning experiment...")
    results_df = run_experiment(merged, feature_cols, tuned_traces, mr_lookup, train_fractions, n_runs=3)
    
    plot_results(results_df, 'xgb_18class_results')
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
