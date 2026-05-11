#!/usr/bin/env python3
"""
Optimized cross-dataset transfer with:
1. Combined training from multiple similar datasets
2. Feature normalization (StandardScaler)
3. Class weighting to handle label shift
4. Focus on stable features
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

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
    trace_to_dataset = {}
    with open('trace.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if '/oracleReuse/' in line:
                parts = line.split('/oracleReuse/')[-1].split('/')
                dataset = parts[0]
                trace_name = normalize_trace(os.path.basename(line))
                trace_to_dataset[trace_name] = dataset
    return trace_to_dataset


def load_data():
    features_df = pd.read_csv('features_20pct.csv')
    features_df['trace'] = features_df['trace'].apply(normalize_trace)
    
    optimal_list, grid_full_list = [], []
    for ratio_str in ['0.001', '0.01', '0.1']:
        opt = pd.read_csv(f'cleaned/{ratio_str}/grid_optimal.csv')
        opt['trace'] = opt['trace'].apply(normalize_trace)
        optimal_list.append(opt)
        full = pd.read_csv(f'cleaned/{ratio_str}/grid_full.csv')
        full['trace'] = full['trace'].apply(normalize_trace)
        grid_full_list.append(full)
    
    optimal_df = pd.concat(optimal_list, ignore_index=True)
    grid_full_df = pd.concat(grid_full_list, ignore_index=True)
    
    features_df['probation_efficiency'] = features_df['hits_small'] / (features_df['hits_main'] + 1e-6)
    features_df['ghost_pressure'] = features_df['hits_ghost'] / (features_df['total_hits'] + features_df['hits_ghost'] + 1e-6)
    features_df['entropy_gap'] = features_df['H_m'] - features_df['H_s']
    
    merged = pd.merge(features_df, optimal_df, on=['trace', 'ratio'], how='inner')
    merged['thrashing_risk'] = merged['rho_unique'] / (merged['ratio'] * 100 + 1e-6)
    # Additional engineered features
    merged['scan_intensity'] = merged['rho_onehit'] / (merged['ratio'] + 1e-6)
    merged['log_total_reqs'] = np.log1p(merged['total_reqs'])
    
    grid_full_df['config'] = list(zip(
        grid_full_df['s_param'], 
        grid_full_df['m_param'].astype(int),
        grid_full_df['t_param'].astype(int), 
        grid_full_df['g_param'].round(1)
    ))
    
    grid_selected = grid_full_df[grid_full_df['config'].apply(lambda c: c in set(SELECTED_CONFIGS))].copy()
    grid_selected['class_id'] = grid_selected['config'].map(CONFIG_TO_ID)
    
    best_configs = grid_selected.groupby(['trace', 'ratio']).apply(
        lambda x: x.loc[x['miss_ratio'].idxmin(), 'config'], include_groups=False
    ).reset_index(name='best_config')
    
    merged = pd.merge(merged, best_configs, on=['trace', 'ratio'], how='left')
    merged['class_id'] = merged['best_config'].map(CONFIG_TO_ID)
    merged = merged.dropna(subset=['class_id'])
    merged['class_id'] = merged['class_id'].astype(int)
    
    trace_to_dataset = build_trace_dataset_map()
    merged['dataset'] = merged['trace'].map(trace_to_dataset).fillna('unknown')
    
    return merged, grid_selected


def build_miss_ratio_matrix(test_df, grid_selected):
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


def calc_topk_acc(probs, y_true, miss_ratio_matrix=None):
    sorted_idx = np.argsort(-probs, axis=1)
    
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
    top3 = 100 * sum(y_true[i] in sorted_idx[i, :3] for i in range(len(y_true))) / len(y_test)
    
    return top1, top2, top3


def compute_class_weights(train_df, test_label_dist=None):
    """Compute class weights to handle label shift."""
    train_counts = train_df['class_id'].value_counts().to_dict()
    n_samples = len(train_df)
    n_classes = len(SELECTED_CONFIGS)
    
    # Balanced weights
    weights = {}
    for class_id in range(n_classes):
        count = train_counts.get(class_id, 1)
        weights[class_id] = n_samples / (n_classes * count)
    
    # If we have test distribution info, boost underrepresented classes
    if test_label_dist is not None:
        for class_id, test_freq in test_label_dist.items():
            train_freq = train_counts.get(class_id, 1) / n_samples
            if test_freq > train_freq:
                boost = min(test_freq / train_freq, 3.0)  # Cap at 3x
                weights[class_id] *= boost
    
    return weights


def run_experiment(train_df, test_df, feature_cols, grid_selected, config_name, 
                   use_scaler=False, use_class_weights=False):
    """Run single experiment."""
    X_train = train_df[feature_cols].values
    y_train = train_df['class_id'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['class_id'].values
    
    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Normalization
    if use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Class weights
    sample_weight = None
    if use_class_weights:
        # Estimate test label distribution from cphy (or use uniform)
        test_dist = test_df['class_id'].value_counts(normalize=True).to_dict()
        weights = compute_class_weights(train_df, test_dist)
        sample_weight = np.array([weights.get(y, 1.0) for y in y_train])
    
    # Train ensemble
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
            random_state=42 + i * 11,
            verbose=-1
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        
        pred_probs = model.predict_proba(X_test)
        if pred_probs.shape[1] < len(SELECTED_CONFIGS):
            full_probs = np.zeros((len(X_test), len(SELECTED_CONFIGS)))
            for j, cls in enumerate(model.classes_):
                full_probs[:, cls] = pred_probs[:, j]
            probs += full_probs
        else:
            probs += pred_probs
    probs /= n_models
    
    # Calc accuracy
    miss_ratio_matrix = build_miss_ratio_matrix(test_df, grid_selected)
    top1, top2, top3 = calc_topk_acc(probs, y_test, miss_ratio_matrix)
    
    return top1, top2, top3


if __name__ == '__main__':
    print("="*70)
    print("OPTIMIZED CROSS-DATASET TRANSFER")
    print("="*70)
    
    merged, grid_selected = load_data()
    
    # Define feature columns
    base_features = [
        'probation_efficiency', 'ghost_pressure', 'entropy_gap', 'thrashing_risk',
        'rho_unique', 'rho_onehit', 'log_total_reqs', 'H_m', 'H_s', 'H_g',
        'scan_intensity', 'ratio'
    ]
    hist_cols = [c for c in merged.columns if 'hist_' in c]
    all_features = list(set(base_features + hist_cols) & set(merged.columns))
    
    # Stable features only (low distribution shift)
    stable_features = [
        'thrashing_risk', 'entropy_gap', 'H_m', 'H_s', 'ratio',
        'probation_efficiency', 'log_total_reqs'
    ]
    stable_hist = [c for c in hist_cols if 'hist_main' in c or 'hist_ghost' in c]
    stable_features = list(set(stable_features + stable_hist) & set(merged.columns))
    
    test_df = merged[merged['dataset'] == 'cphy'].reset_index(drop=True)
    y_test = test_df['class_id'].values
    
    print(f"\nTest set: cphy ({len(test_df)} samples, {test_df['trace'].nunique()} traces)")
    
    # Different training configurations
    configs = [
        ('tencentBlock only', ['tencentBlock'], all_features, False, False),
        ('tencentBlock + Scaler', ['tencentBlock'], all_features, True, False),
        ('tencentBlock + Weights', ['tencentBlock'], all_features, False, True),
        ('tencentBlock + Scaler + Weights', ['tencentBlock'], all_features, True, True),
        ('alibabaBlock only', ['alibabaBlock'], all_features, False, False),
        ('alibabaBlock + Scaler + Weights', ['alibabaBlock'], all_features, True, True),
        ('tencent + alibaba', ['tencentBlock', 'alibabaBlock'], all_features, False, False),
        ('tencent + alibaba + Scaler', ['tencentBlock', 'alibabaBlock'], all_features, True, False),
        ('tencent + alibaba + Weights', ['tencentBlock', 'alibabaBlock'], all_features, False, True),
        ('tencent + alibaba + All', ['tencentBlock', 'alibabaBlock'], all_features, True, True),
        ('tencent+alibaba+cf', ['tencentBlock', 'alibabaBlock', 'cf'], all_features, False, False),
        ('tencent+alibaba+cf + All', ['tencentBlock', 'alibabaBlock', 'cf'], all_features, True, True),
        ('Stable features only', ['tencentBlock', 'alibabaBlock'], stable_features, True, True),
    ]
    
    print("\n" + "-"*70)
    print(f"{'Configuration':<35} {'Top-1':>8} {'Top-2':>8} {'Top-3':>8}")
    print("-"*70)
    
    results = []
    for name, datasets, features, use_scaler, use_weights in configs:
        train_df = merged[merged['dataset'].isin(datasets)].reset_index(drop=True)
        
        if len(train_df) < 50:
            continue
        
        top1, top2, top3 = run_experiment(
            train_df, test_df, features, grid_selected, name,
            use_scaler=use_scaler, use_class_weights=use_weights
        )
        
        results.append({'config': name, 'top1': top1, 'top2': top2, 'top3': top3})
        print(f"{name:<35} {top1:>7.1f}% {top2:>7.1f}% {top3:>7.1f}%")
    
    print("-"*70)
    
    # Best result
    results_df = pd.DataFrame(results)
    best_idx = results_df['top1'].idxmax()
    print(f"\n🏆 Best config: {results_df.loc[best_idx, 'config']}")
    print(f"   Top-1: {results_df.loc[best_idx, 'top1']:.1f}%")
    print(f"   Top-2: {results_df.loc[best_idx, 'top2']:.1f}%")
    print(f"   Top-3: {results_df.loc[best_idx, 'top3']:.1f}%")
