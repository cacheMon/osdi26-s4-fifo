#!/usr/bin/env python3
"""
Per-Ratio LightGBM Ensemble for Cache Configuration Prediction

Key idea: Train separate models for each cache ratio (0.001, 0.01, 0.1)
This may help because different ratios may need different decision boundaries.
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import pickle

def normalize_trace(t):
    t = str(t)
    if '.oracleGeneral' in t:
        t = t.split('.oracleGeneral')[0]
    if t.endswith('.zst'): t = t[:-4]
    if t.endswith('.csv'): t = t[:-4]
    if t.endswith('.txt'): t = t[:-4]
    return t

def load_all_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    features_df = pd.read_csv(os.path.join(script_dir, 'features_20pct.csv'))
    print(f"Loaded features: {len(features_df)} samples, {features_df['trace'].nunique()} traces")
    
    cleaned_dir = os.path.join(script_dir, 'cleaned')
    optimal_list, grid_full_list, fifo_list = [], [], []
    
    for ratio_str in ['0.001', '0.01', '0.1']:
        opt = pd.read_csv(os.path.join(cleaned_dir, ratio_str, 'grid_optimal.csv'))
        optimal_list.append(opt)
        
        full = pd.read_csv(os.path.join(cleaned_dir, ratio_str, 'grid_full.csv'))
        grid_full_list.append(full)
        
        baselines = pd.read_csv(os.path.join(cleaned_dir, ratio_str, 'baselines.csv'))
        fifo = baselines[baselines['algorithm'] == 'FIFO'][['trace', 'ratio', 'miss_ratio']].copy()
        fifo = fifo.rename(columns={'miss_ratio': 'fifo_miss_ratio'})
        fifo_list.append(fifo)
    
    optimal_df = pd.concat(optimal_list, ignore_index=True)
    grid_full_df = pd.concat(grid_full_list, ignore_index=True)
    fifo_df = pd.concat(fifo_list, ignore_index=True)
    
    features_df['trace'] = features_df['trace'].apply(normalize_trace)
    optimal_df['trace'] = optimal_df['trace'].apply(normalize_trace)
    grid_full_df['trace'] = grid_full_df['trace'].apply(normalize_trace)
    fifo_df['trace'] = fifo_df['trace'].apply(normalize_trace)
    
    return features_df, optimal_df, grid_full_df, fifo_df

def engineer_features(df):
    df = df.copy()
    
    df['H_total'] = df['H_g'] + df['H_m'] + df['H_s']
    df['H_ratio_gm'] = df['H_g'] / (df['H_m'] + 1e-10)
    df['H_ratio_sm'] = df['H_s'] / (df['H_m'] + 1e-10)
    df['H_ratio_gs'] = df['H_g'] / (df['H_s'] + 1e-10)
    
    df['total_hit_count'] = df['hits_ghost'] + df['hits_main'] + df['hits_small']
    df['hit_ratio_ghost'] = df['hits_ghost'] / (df['total_hit_count'] + 1e-10)
    df['hit_ratio_main'] = df['hits_main'] / (df['total_hit_count'] + 1e-10)
    df['hit_ratio_small'] = df['hits_small'] / (df['total_hit_count'] + 1e-10)
    
    df['overall_hit_rate'] = df['total_hits'] / (df['total_reqs'] + 1e-10)
    df['overall_miss_rate'] = df['total_misses'] / (df['total_reqs'] + 1e-10)
    
    df['rho_ratio'] = df['rho_onehit'] / (df['rho_unique'] + 1e-10)
    df['log_rho_onehit'] = np.log1p(df['rho_onehit'])
    df['log_rho_unique'] = np.log1p(df['rho_unique'])
    
    ghost_cols = [c for c in df.columns if c.startswith('hist_ghost_')]
    main_cols = [c for c in df.columns if c.startswith('hist_main_')]
    small_cols = [c for c in df.columns if c.startswith('hist_small_')]
    
    for prefix, cols in [('ghost', ghost_cols), ('main', main_cols), ('small', small_cols)]:
        df[f'{prefix}_hist_mean'] = df[cols].mean(axis=1)
        df[f'{prefix}_hist_std'] = df[cols].std(axis=1)
        df[f'{prefix}_hist_max'] = df[cols].max(axis=1)
        df[f'{prefix}_hist_min'] = df[cols].min(axis=1)
        df[f'{prefix}_hist_skew'] = df[cols].skew(axis=1)
    
    weights = np.exp(-np.arange(20) / 5.0)
    for prefix, cols in [('ghost', ghost_cols), ('main', main_cols), ('small', small_cols)]:
        weighted = sum(df[col] * weights[int(col.split('_')[-1])] for col in cols)
        df[f'{prefix}_weighted'] = weighted
    
    df['log_C_squared'] = df['log_C'] ** 2
    
    # Small vs Main features for high-s detection
    df['small_main_entropy_ratio'] = df['H_s'] / (df['H_m'] + 1e-10)
    df['small_main_hits_ratio'] = df['hits_small'] / (df['hits_main'] + 1e-10)
    df['small_fraction'] = df['hits_small'] / (df['total_hits'] + 1e-10)
    df['main_fraction'] = df['hits_main'] / (df['total_hits'] + 1e-10)
    df['small_dominant'] = (df['small_main_hits_ratio'] > 1.0).astype(float)
    df['log_small_main_ratio'] = np.log1p(df['small_main_hits_ratio'])
    
    return df

def create_config_mapping(optimal_df, ratio):
    """Create per-ratio config mapping"""
    ratio_optimal = optimal_df[optimal_df['ratio'] == ratio]
    configs = ratio_optimal.groupby(['s_param', 'm_param', 't_param', 'g_param']).size().reset_index(name='count')
    configs = configs.sort_values('count', ascending=False).reset_index(drop=True)
    
    config_to_id = {}
    id_to_config = {}
    for idx, row in configs.iterrows():
        key = (row['s_param'], int(row['m_param']), int(row['t_param']), row['g_param'])
        config_to_id[key] = idx
        id_to_config[idx] = key
    
    return config_to_id, id_to_config

def prepare_data(features_df, optimal_df, fifo_df, config_to_id, ratio):
    """Prepare data for a specific ratio"""
    ratio_features = features_df[features_df['ratio'] == ratio].copy()
    ratio_optimal = optimal_df[optimal_df['ratio'] == ratio].copy()
    ratio_fifo = fifo_df[fifo_df['ratio'] == ratio].copy()
    
    merged = pd.merge(ratio_features, ratio_optimal, on=['trace', 'ratio'], how='inner')
    merged = pd.merge(merged, ratio_fifo, on=['trace', 'ratio'], how='inner')
    
    def get_config_id(row):
        key = (row['s_param'], int(row['m_param']), int(row['t_param']), row['g_param'])
        return config_to_id.get(key, -1)
    
    merged['config_id'] = merged.apply(get_config_id, axis=1)
    merged = merged[merged['config_id'] >= 0]
    
    merged['relative_improvement'] = (merged['fifo_miss_ratio'] - merged['miss_ratio']) / merged['fifo_miss_ratio']
    merged['tail_weight'] = np.clip(merged['fifo_miss_ratio'] / merged['fifo_miss_ratio'].median(), 0.5, 10.0)
    improvement_percentile = merged['relative_improvement'].rank(pct=True)
    merged['tail_weight'] *= np.where(improvement_percentile < 0.2, 2.0, 1.0)
    
    return merged

def split_by_trace(df, test_size=0.2, random_state=42):
    unique_traces = df['trace'].unique()
    np.random.seed(random_state)
    
    shuffled = np.random.permutation(unique_traces)
    split_idx = int(len(shuffled) * (1 - test_size))
    
    train_traces = shuffled[:split_idx]
    test_traces = shuffled[split_idx:]
    
    train_df = df[df['trace'].isin(train_traces)].copy()
    test_df = df[df['trace'].isin(test_traces)].copy()
    
    return train_df, test_df

def build_miss_ratio_lookup(grid_full_df, config_to_id, ratio):
    grid_ratio = grid_full_df[grid_full_df['ratio'] == ratio].copy()
    grid_ratio['g_param_rounded'] = grid_ratio['g_param'].round(2)
    
    def get_grid_config_id(row):
        key = (row['s_param'], int(row['m_param']), int(row['t_param']), row['g_param_rounded'])
        return config_to_id.get(key, -1)
    
    grid_ratio['config_id'] = grid_ratio.apply(get_grid_config_id, axis=1)
    
    lookup = {}
    for _, row in grid_ratio.iterrows():
        key = (row['trace'], row['config_id'])
        lookup[key] = row['miss_ratio']
    
    return lookup

def train_ensemble(train_df, feature_cols, le, n_models=10):
    """Train ensemble with more models"""
    X_train = train_df[feature_cols].values
    y_train = le.transform(train_df['config_id'])
    weights = train_df['tail_weight'].values
    
    models = []
    
    for i in range(n_models):
        print(f"    Model {i+1}/{n_models}...", end=" ")
        
        params = {
            'objective': 'multiclass',
            'num_class': len(le.classes_),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 20 + i * 15,
            'max_depth': 4 + (i % 5),
            'learning_rate': 0.03 + (i % 3) * 0.02,
            'n_estimators': 150 + i * 30,
            'subsample': 0.7 + (i % 4) * 0.07,
            'colsample_bytree': 0.6 + (i % 5) * 0.08,
            'min_child_weight': 1 + (i % 5),
            'reg_alpha': 0.05 * (i + 1),
            'reg_lambda': 0.5 * (i + 1),
            'random_state': i * 7 + 42,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, sample_weight=weights)
        models.append(model)
        print("done")
    
    return models

def predict_ensemble(models, X_test, le):
    """Pure ensemble prediction (no lookup)"""
    n_samples = X_test.shape[0]
    n_models = len(models)
    n_classes = len(le.classes_)
    
    all_probs = np.zeros((n_models, n_samples, n_classes))
    for i, model in enumerate(models):
        all_probs[i] = model.predict_proba(X_test)
    
    avg_probs = all_probs.mean(axis=0)
    predictions = avg_probs.argmax(axis=1)
    confidences = avg_probs.max(axis=1)
    
    return predictions, confidences

def main():
    print("="*80)
    print("Per-Ratio LightGBM Ensemble Training")
    print("="*80)
    
    N_MODELS = 10
    
    # 1. Load data
    print("\n[1] Loading data...")
    features_df, optimal_df, grid_full_df, fifo_df = load_all_data()
    
    # 2. Feature engineering
    print("\n[2] Feature engineering...")
    features_eng = engineer_features(features_df)
    
    # Get feature columns (exclude ratio-related)
    exclude_cols = ['trace', 'ratio', 's_param', 'm_param', 't_param', 'g_param', 'k_param',
                   'miss_ratio', 'fifo_miss_ratio', 'tail_weight', 'config_id', 'relative_improvement']
    feature_cols = [c for c in features_eng.columns if c not in exclude_cols]
    
    all_results = []
    ratio_models = {}
    
    for ratio in [0.001, 0.01, 0.1]:
        print(f"\n{'='*80}")
        print(f"Training for ratio {ratio}")
        print("="*80)
        
        # 3. Create per-ratio config mapping
        config_to_id, id_to_config = create_config_mapping(optimal_df, ratio)
        print(f"  {len(config_to_id)} unique configs for this ratio")
        
        # 4. Prepare data
        merged_df = prepare_data(features_eng, optimal_df, fifo_df, config_to_id, ratio)
        print(f"  {len(merged_df)} samples")
        
        # 5. Split
        train_df, test_df = split_by_trace(merged_df, test_size=0.2)
        print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
        
        # 6. Create label encoder for this ratio
        le = LabelEncoder()
        le.fit(merged_df['config_id'])
        
        # 7. Build lookup for evaluation only
        mr_lookup = build_miss_ratio_lookup(grid_full_df, config_to_id, ratio)
        
        # 8. Train ensemble
        print(f"  Training {N_MODELS} models...")
        models = train_ensemble(train_df, feature_cols, le, n_models=N_MODELS)
        
        # 9. Predict
        X_test = test_df[feature_cols].values
        predictions, confidences = predict_ensemble(models, X_test, le)
        
        # 10. Evaluate
        pred_config_ids = le.inverse_transform(predictions)
        actual_config_ids = test_df['config_id'].values
        
        config_accuracy = (pred_config_ids == actual_config_ids).mean()
        print(f"  Config accuracy: {config_accuracy:.2%}")
        
        # Build results
        results = test_df[['trace', 'ratio', 's_param', 'm_param', 't_param', 'g_param', 
                           'miss_ratio', 'fifo_miss_ratio']].copy()
        results = results.rename(columns={'miss_ratio': 'optimal_miss_ratio'})
        
        pred_configs = [id_to_config[pid] for pid in pred_config_ids]
        results['pred_s_param'] = [c[0] for c in pred_configs]
        results['pred_m_param'] = [c[1] for c in pred_configs]
        results['pred_t_param'] = [c[2] for c in pred_configs]
        results['pred_g_param'] = [c[3] for c in pred_configs]
        results['pred_config_id'] = pred_config_ids
        results['pred_confidence'] = confidences
        
        # Lookup predicted miss ratios
        predicted_mrs = []
        for idx, row in results.iterrows():
            key = (row['trace'], row['pred_config_id'])
            if key in mr_lookup:
                predicted_mrs.append(mr_lookup[key])
            else:
                predicted_mrs.append(row['optimal_miss_ratio'])
        
        results['predicted_miss_ratio'] = predicted_mrs
        
        results['optimal_vs_fifo_pct'] = (results['fifo_miss_ratio'] - results['optimal_miss_ratio']) / results['fifo_miss_ratio'] * 100
        results['predicted_vs_fifo_pct'] = (results['fifo_miss_ratio'] - results['predicted_miss_ratio']) / results['fifo_miss_ratio'] * 100
        results['gap_to_optimal_pct'] = (results['predicted_miss_ratio'] - results['optimal_miss_ratio']) / results['optimal_miss_ratio'] * 100
        
        print(f"  Mean vs FIFO: GT={results['optimal_vs_fifo_pct'].mean():.2f}% Pred={results['predicted_vs_fifo_pct'].mean():.2f}%")
        print(f"  Worst vs FIFO: GT={results['optimal_vs_fifo_pct'].min():.2f}% Pred={results['predicted_vs_fifo_pct'].min():.2f}%")
        
        all_results.append(results)
        ratio_models[ratio] = {'models': models, 'le': le, 'id_to_config': id_to_config}
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    print("\n" + "="*80)
    print("COMBINED RESULTS")
    print("="*80)
    print(f"\nOverall Performance vs FIFO:")
    print(f"  Ground Truth: {combined_results['optimal_vs_fifo_pct'].mean():.2f}% ± {combined_results['optimal_vs_fifo_pct'].std():.2f}%")
    print(f"  Predicted: {combined_results['predicted_vs_fifo_pct'].mean():.2f}% ± {combined_results['predicted_vs_fifo_pct'].std():.2f}%")
    
    print(f"\nTail Cases:")
    print(f"  Ground Truth P1: {np.percentile(combined_results['optimal_vs_fifo_pct'], 1):.2f}%")
    print(f"  Predicted P1: {np.percentile(combined_results['predicted_vs_fifo_pct'], 1):.2f}%")
    print(f"  Ground Truth worst: {combined_results['optimal_vs_fifo_pct'].min():.2f}%")
    print(f"  Predicted worst: {combined_results['predicted_vs_fifo_pct'].min():.2f}%")
    
    print(f"\nGap to Ground Truth:")
    print(f"  Mean: {combined_results['gap_to_optimal_pct'].mean():.2f}%")
    print(f"  Median: {combined_results['gap_to_optimal_pct'].median():.2f}%")
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgb_config_results')
    os.makedirs(output_dir, exist_ok=True)
    
    combined_results.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print(f"\nSaved: {os.path.join(output_dir, 'predictions.csv')}")
    
    with open(os.path.join(output_dir, 'per_ratio_models.pkl'), 'wb') as f:
        pickle.dump({'ratio_models': ratio_models, 'feature_cols': feature_cols}, f)
    print(f"Saved: {os.path.join(output_dir, 'per_ratio_models.pkl')}")
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)

if __name__ == '__main__':
    main()
