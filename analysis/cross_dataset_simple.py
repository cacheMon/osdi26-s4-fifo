#!/usr/bin/env python3
"""
Simple cross-dataset generalization experiment:
Train on tencentBlock, test on each other dataset separately.
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 22,
    'xtick.labelsize': 14,
    'ytick.labelsize': 18,
    'legend.fontsize': 14,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

SELECTED_CONFIGS = [
    (0.20, 1, 0, 3.0), (0.05, 1, 0, 0.9), (0.50, 1, 0, 0.9), (0.20, 1, 0, 0.9),
    (0.05, 2, 0, 6.0), (0.10, 2, 1, 3.0), (0.30, 2, 0, 3.0), (0.05, 2, 0, 3.0),
    (0.10, 2, 0, 0.9), (0.70, 1, 1, 0.9), (0.20, 1, 1, 0.9), (0.05, 1, 1, 0.9),
    (0.30, 1, 0, 6.0), (0.20, 2, 0, 0.9), (0.90, 2, 0, 3.0), (0.10, 2, 0, 6.0),
    (0.30, 2, 1, 3.0), (0.05, 2, 0, 0.9),
]
CONFIG_TO_ID = {c: i for i, c in enumerate(SELECTED_CONFIGS)}


def normalize_trace(t):
    t = str(t)
    if '.oracleGeneral' in t:
        t = t.split('.oracleGeneral')[0]
    for ext in ['.zst', '.csv', '.txt']:
        if t.endswith(ext): t = t[:-len(ext)]
    return t


def build_trace_dataset_map():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trace_to_dataset = {}
    with open(os.path.join(script_dir, 'trace.txt'), 'r') as f:
        for line in f:
            line = line.strip()
            if '/oracleReuse/' in line:
                parts = line.split('/oracleReuse/')[-1].split('/')
                dataset = parts[0]
                trace_name = normalize_trace(os.path.basename(line))
                trace_to_dataset[trace_name] = dataset
    return trace_to_dataset


def load_and_prepare_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load features
    features_df = pd.read_csv(os.path.join(script_dir, 'features_20pct.csv'))
    features_df['trace'] = features_df['trace'].apply(normalize_trace)
    
    # Load optimal configs and grid data
    cleaned_dir = os.path.join(script_dir, 'cleaned')
    optimal_list, grid_full_list = [], []
    
    for ratio_str in ['0.001', '0.01', '0.1']:
        opt = pd.read_csv(os.path.join(cleaned_dir, ratio_str, 'grid_optimal.csv'))
        opt['trace'] = opt['trace'].apply(normalize_trace)
        optimal_list.append(opt)
        
        full = pd.read_csv(os.path.join(cleaned_dir, ratio_str, 'grid_full.csv'))
        full['trace'] = full['trace'].apply(normalize_trace)
        grid_full_list.append(full)
    
    optimal_df = pd.concat(optimal_list, ignore_index=True)
    grid_full_df = pd.concat(grid_full_list, ignore_index=True)
    
    # Engineer features
    features_df['probation_efficiency'] = features_df['hits_small'] / (features_df['hits_main'] + 1e-6)
    features_df['ghost_pressure'] = features_df['hits_ghost'] / (features_df['total_hits'] + features_df['hits_ghost'] + 1e-6)
    features_df['entropy_gap'] = features_df['H_m'] - features_df['H_s']
    
    # Merge
    merged = pd.merge(features_df, optimal_df, on=['trace', 'ratio'], how='inner')
    merged['thrashing_risk'] = merged['rho_unique'] / (merged['ratio'] * 100 + 1e-6)
    
    # Get best config
    grid_full_df['config'] = list(zip(
        grid_full_df['s_param'], 
        grid_full_df['m_param'].astype(int),
        grid_full_df['t_param'].astype(int), 
        grid_full_df['g_param'].round(1)
    ))
    
    selected_set = set(SELECTED_CONFIGS)
    grid_selected = grid_full_df[grid_full_df['config'].apply(lambda c: c in selected_set)].copy()
    grid_selected['class_id'] = grid_selected['config'].map(CONFIG_TO_ID)
    
    best_configs = grid_selected.groupby(['trace', 'ratio']).apply(
        lambda x: x.loc[x['miss_ratio'].idxmin(), 'config'], include_groups=False
    ).reset_index(name='best_config')
    
    merged = pd.merge(merged, best_configs, on=['trace', 'ratio'], how='left')
    merged['class_id'] = merged['best_config'].map(CONFIG_TO_ID)
    merged = merged.dropna(subset=['class_id'])
    merged['class_id'] = merged['class_id'].astype(int)
    
    # Add dataset label
    trace_to_dataset = build_trace_dataset_map()
    merged['dataset'] = merged['trace'].map(trace_to_dataset).fillna('unknown')
    
    # Feature columns
    keep_features = [
        'probation_efficiency', 'ghost_pressure', 'entropy_gap', 'thrashing_risk',
        'rho_unique', 'rho_onehit', 'total_reqs', 'H_m', 'H_s', 'H_g',
    ]
    hist_cols = [c for c in merged.columns if 'hist_' in c]
    feature_cols = list(set(keep_features + hist_cols) & set(merged.columns))
    
    return merged, feature_cols, grid_selected


def calc_topk_acc(probs, y_true, miss_ratio_matrix=None):
    """Calculate top-k accuracy with miss ratio tolerance for Top-1."""
    sorted_idx = np.argsort(-probs, axis=1)
    
    # Top-1 with tolerance
    if miss_ratio_matrix is not None:
        top1_correct = 0
        for i in range(len(y_true)):
            pred_class = sorted_idx[i, 0]
            true_class = y_true[i]
            if pred_class == true_class:
                top1_correct += 1
            else:
                pred_mr = round(miss_ratio_matrix[i, pred_class], 4)
                true_mr = round(miss_ratio_matrix[i, true_class], 4)
                if pred_mr == true_mr:
                    top1_correct += 1
        top1 = 100 * top1_correct / len(y_true)
    else:
        top1 = 100 * (sorted_idx[:, 0] == y_true).sum() / len(y_true)
    
    top2 = 100 * sum(y_true[i] in sorted_idx[i, :2] for i in range(len(y_true))) / len(y_true)
    top3 = 100 * sum(y_true[i] in sorted_idx[i, :3] for i in range(len(y_true))) / len(y_true)
    
    return top1, top2, top3


def build_miss_ratio_matrix(test_df, grid_selected):
    """Build miss ratio matrix for test samples."""
    miss_ratio_matrix = np.full((len(test_df), len(SELECTED_CONFIGS)), np.inf)
    for idx, row in test_df.iterrows():
        trace, ratio = row['trace'], row['ratio']
        sample_grid = grid_selected[(grid_selected['trace'] == trace) & 
                                     (grid_selected['ratio'] == ratio)]
        for _, g_row in sample_grid.iterrows():
            class_id = g_row['class_id']
            if not pd.isna(class_id):
                miss_ratio_matrix[idx, int(class_id)] = g_row['miss_ratio']
    return miss_ratio_matrix


def main():
    print("="*60)
    print("Train on TencentBlock, Test on Other Datasets")
    print("="*60)
    
    merged, feature_cols, grid_selected = load_and_prepare_data()
    
    # Get dataset distribution
    dataset_counts = merged.groupby('dataset')['trace'].nunique().sort_values(ascending=False)
    print(f"\nDataset distribution ({len(dataset_counts)} datasets):")
    for ds, cnt in dataset_counts.items():
        print(f"  {ds}: {cnt} traces")
    
    # Training data: tencentBlock
    train_df = merged[merged['dataset'] == 'tencentBlock'].reset_index(drop=True)
    X_train = np.nan_to_num(train_df[feature_cols].values, nan=0.0)
    y_train = train_df['class_id'].values
    
    print(f"\nTrain on: tencentBlock ({len(train_df)} samples, {train_df['trace'].nunique()} traces)")
    
    # Train ensemble
    print("\nTraining ensemble...")
    n_models = 10
    models = []
    for i in range(n_models):
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(SELECTED_CONFIGS),
            num_leaves=31 + i * 5,
            max_depth=6 + i % 4,
            learning_rate=0.05,
            n_estimators=200,
            random_state=42 + i * 11,
            verbose=-1
        )
        model.fit(X_train, y_train)
        models.append(model)
    
    # Test on each dataset
    results = []
    print("\n" + "="*60)
    print("RESULTS: Test on Each Dataset")
    print("="*60)
    
    # First test on tencentBlock itself (in-dataset)
    test_datasets = ['tencentBlock'] + [ds for ds in dataset_counts.index if ds != 'tencentBlock']
    
    for dataset in test_datasets:
        test_df = merged[merged['dataset'] == dataset].reset_index(drop=True)
        if len(test_df) == 0:
            continue
            
        X_test = np.nan_to_num(test_df[feature_cols].values, nan=0.0)
        y_test = test_df['class_id'].values
        
        # Predict with ensemble
        probs = np.zeros((len(X_test), len(SELECTED_CONFIGS)))
        for model in models:
            pred_probs = model.predict_proba(X_test)
            if pred_probs.shape[1] < len(SELECTED_CONFIGS):
                full_probs = np.zeros((len(X_test), len(SELECTED_CONFIGS)))
                for j, cls in enumerate(model.classes_):
                    full_probs[:, cls] = pred_probs[:, j]
                probs += full_probs
            else:
                probs += pred_probs
        probs /= n_models
        
        # Build miss ratio matrix for tolerance
        miss_ratio_matrix = build_miss_ratio_matrix(test_df, grid_selected)
        
        # Calculate accuracy
        top1, top2, top3 = calc_topk_acc(probs, y_test, miss_ratio_matrix)
        
        n_traces = test_df['trace'].nunique()
        is_train = 'TRAIN' if dataset == 'tencentBlock' else 'TEST'
        
        result = {
            'dataset': dataset,
            'n_traces': n_traces,
            'n_samples': len(test_df),
            'top1': top1,
            'top2': top2,
            'top3': top3,
            'type': is_train
        }
        results.append(result)
        
        print(f"\n[{is_train}] {dataset}: {n_traces} traces, {len(test_df)} samples")
        print(f"       Top1={top1:.1f}%, Top2={top2:.1f}%, Top3={top3:.1f}%")
    
    results_df = pd.DataFrame(results)
    
    # Plot results
    out_dir = 'xgb_18class_results'
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(results_df))
    width = 0.25
    
    colors_top1 = ['#27ae60' if r['type'] == 'TRAIN' else '#3498db' for _, r in results_df.iterrows()]
    colors_top2 = ['#2ecc71' if r['type'] == 'TRAIN' else '#5dade2' for _, r in results_df.iterrows()]
    colors_top3 = ['#82e0aa' if r['type'] == 'TRAIN' else '#85c1e9' for _, r in results_df.iterrows()]
    
    bars1 = ax.bar(x - width, results_df['top1'], width, label='Top-1', color=colors_top1, edgecolor='black')
    bars2 = ax.bar(x, results_df['top2'], width, label='Top-2', color=colors_top2, edgecolor='black')
    bars3 = ax.bar(x + width, results_df['top3'], width, label='Top-3', color=colors_top3, edgecolor='black')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['dataset'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/cross_dataset_simple.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out_dir}/cross_dataset_simple.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out_dir}/cross_dataset_simple.pdf")
    plt.close()
    
    # Save CSV
    results_df.to_csv(f'{out_dir}/cross_dataset_simple.csv', index=False)
    print(f"Saved: {out_dir}/cross_dataset_simple.csv")
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
