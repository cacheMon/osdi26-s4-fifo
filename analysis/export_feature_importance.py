#!/usr/bin/env python3
"""
Export feature importance from 100% training run.
Histogram features are grouped together.
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb

# Configuration from train_xgb_18class.py
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


def engineer_features(df):
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
    
    return df


def main():
    print("="*60)
    print("Feature Importance Export (100% Training)")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    optimal_df, grid_full_df, fifo_df, features_df = load_all_data()
    
    # Load tuned traces
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tuned_df = pd.read_csv(os.path.join(script_dir, 'xgb_18class_results/tune.csv'))
    tuned_traces = set(tuned_df['trace'].apply(normalize_trace).unique())
    
    # Engineer features
    features_eng = engineer_features(features_df)
    
    # Merge
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
    
    # Split: tuned traces = test, others = train
    all_traces = set(merged['trace'].unique())
    train_traces = all_traces - tuned_traces
    
    train_df = merged[merged['trace'].isin(train_traces)]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['class_id'].values
    X_train = np.nan_to_num(X_train, nan=0.0)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Features: {len(feature_cols)}")
    
    # Train model
    print("\nTraining model...")
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(SELECTED_CONFIGS),
        num_leaves=50,
        max_depth=8,
        learning_rate=0.05,
        n_estimators=300,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train, feature_name=feature_cols)
    
    # Get feature importance
    importance = model.feature_importances_
    feat_imp = dict(zip(feature_cols, importance))
    
    # Group histogram features
    grouped_imp = {}
    hist_groups = {
        'hist_main': 0,
        'hist_small': 0,
        'hist_ghost': 0,
    }
    
    for feat, imp in feat_imp.items():
        grouped = False
        for prefix in hist_groups:
            if feat.startswith(prefix):
                hist_groups[prefix] += imp
                grouped = True
                break
        if not grouped:
            grouped_imp[feat] = imp
    
    # Add grouped histogram importance
    for prefix, imp in hist_groups.items():
        if imp > 0:
            grouped_imp[prefix] = imp
    
    # Sort by importance
    sorted_imp = sorted(grouped_imp.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*60)
    print("Feature Importance (Grouped)")
    print("="*60)
    
    total = sum(imp for _, imp in sorted_imp)
    for feat, imp in sorted_imp:
        pct = imp / total * 100
        print(f"  {feat:25s}: {imp:8.1f} ({pct:5.1f}%)")
    
    # Save to CSV
    imp_df = pd.DataFrame(sorted_imp, columns=['feature', 'importance'])
    imp_df['importance_pct'] = imp_df['importance'] / imp_df['importance'].sum() * 100
    imp_df.to_csv('xgb_18class_results/feature_importance.csv', index=False)
    print(f"\nSaved: xgb_18class_results/feature_importance.csv")


if __name__ == '__main__':
    main()
