#!/usr/bin/env python3
"""
Standalone evaluation script - loads saved model and evaluates without retraining
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle

def normalize_trace(t):
    """Normalize trace names for consistent matching across data sources"""
    t = str(t)
    if '.oracleGeneral' in t:
        t = t.split('.oracleGeneral')[0]
    if t.endswith('.zst'): t = t[:-4]
    if t.endswith('.csv'): t = t[:-4]
    if t.endswith('.txt'): t = t[:-4]
    return t

def load_all_data():
    """Load features, grid_full, and grid_optimal"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load features (use 20pct data)
    features_df = pd.read_csv(os.path.join(script_dir, 'features_20pct.csv'))
    print(f"Loaded features: {len(features_df)} samples, {features_df['trace'].nunique()} traces")
    
    # Load grid_optimal (labels), grid_full, and FIFO
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
    
    print(f"Loaded optimal: {len(optimal_df)}, grid_full: {len(grid_full_df)}, FIFO: {len(fifo_df)}")
    
    # Normalize trace names
    features_df['trace'] = features_df['trace'].apply(normalize_trace)
    optimal_df['trace'] = optimal_df['trace'].apply(normalize_trace)
    grid_full_df['trace'] = grid_full_df['trace'].apply(normalize_trace)
    fifo_df['trace'] = fifo_df['trace'].apply(normalize_trace)
    
    return features_df, optimal_df, grid_full_df, fifo_df

def create_config_mapping(optimal_df):
    """Create mapping from parameter tuple to config_id"""
    configs = optimal_df.groupby(['s_param', 'm_param', 't_param', 'g_param']).size().reset_index(name='count')
    configs = configs.sort_values('count', ascending=False).reset_index(drop=True)
    
    config_to_id = {}
    id_to_config = {}
    for idx, row in configs.iterrows():
        key = (row['s_param'], int(row['m_param']), int(row['t_param']), row['g_param'])
        config_to_id[key] = idx
        id_to_config[idx] = key
    
    print(f"Created mapping for {len(config_to_id)} unique configurations")
    return config_to_id, id_to_config

def engineer_features(df):
    """Enhanced feature engineering (same as training)"""
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
        
        early = [c for c in cols if int(c.split('_')[-1]) < 10]
        late = [c for c in cols if int(c.split('_')[-1]) >= 10]
        df[f'{prefix}_early_sum'] = df[early].sum(axis=1)
        df[f'{prefix}_late_sum'] = df[late].sum(axis=1)
        df[f'{prefix}_early_late_ratio'] = df[f'{prefix}_early_sum'] / (df[f'{prefix}_late_sum'] + 1e-10)
    
    for i in range(20):
        df[f'sg_combined_{i}'] = df[f'hist_small_{i}'] + df[f'hist_ghost_{i}']
        df[f'sg_ratio_{i}'] = df[f'hist_small_{i}'] / (df[f'hist_ghost_{i}'] + 1e-10)
    
    weights = np.exp(-np.arange(20) / 5.0)
    for prefix, cols in [('ghost', ghost_cols), ('main', main_cols), ('small', small_cols)]:
        weighted = sum(df[col] * weights[int(col.split('_')[-1])] for col in cols)
        df[f'{prefix}_weighted'] = weighted
    
    df['log_ratio'] = np.log10(df['ratio'])
    df['sqrt_ratio'] = np.sqrt(df['ratio'])
    df['log_C_squared'] = df['log_C'] ** 2
    df['log_C_ratio'] = df['log_C'] * df['log_ratio']
    
    df['hits_ghost_entropy'] = df['hits_ghost'] * df['H_g']
    df['hits_main_entropy'] = df['hits_main'] * df['H_m']
    df['hits_small_entropy'] = df['hits_small'] * df['H_s']
    
    return df

def prepare_data(features_df, optimal_df, fifo_df, config_to_id):
    """Merge and prepare data with config_id target"""
    merged = pd.merge(features_df, optimal_df, on=['trace', 'ratio'], how='inner')
    merged = pd.merge(merged, fifo_df, on=['trace', 'ratio'], how='inner')
    
    def get_config_id(row):
        key = (row['s_param'], int(row['m_param']), int(row['t_param']), row['g_param'])
        return config_to_id.get(key, -1)
    
    merged['config_id'] = merged.apply(get_config_id, axis=1)
    merged = merged[merged['config_id'] >= 0]
    
    merged['relative_improvement'] = (merged['fifo_miss_ratio'] - merged['miss_ratio']) / merged['fifo_miss_ratio']
    merged['tail_weight'] = np.clip(merged['fifo_miss_ratio'] / merged['fifo_miss_ratio'].median(), 0.5, 10.0)
    improvement_percentile = merged['relative_improvement'].rank(pct=True)
    merged['tail_weight'] *= np.where(improvement_percentile < 0.2, 2.0, 1.0)
    
    print(f"Prepared {len(merged)} samples with {merged['config_id'].nunique()} unique configs")
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
    
    print(f"Train: {len(train_traces)} traces, {len(train_df)} samples")
    print(f"Test: {len(test_traces)} traces, {len(test_df)} samples")
    
    return train_df, test_df

def evaluate_model(model, le, test_df, feature_cols, id_to_config, grid_full_df):
    """Evaluate model predictions"""
    X_test = test_df[feature_cols]
    num_classes = len(le.classes_)
    n_samples = len(test_df)
    
    print(f"Evaluating {n_samples} test samples with {num_classes} classes...")
    
    # Predict using DMatrix (model is xgb.Booster)
    dtest = xgb.DMatrix(X_test)
    pred_raw = model.predict(dtest)
    
    print(f"Raw prediction shape: {pred_raw.shape}")
    
    # Handle different output shapes
    if pred_raw.ndim == 1:
        # If 1D, assume it's class predictions or need to reshape
        if len(pred_raw) == n_samples * num_classes:
            pred_logits = pred_raw.reshape(n_samples, num_classes)
        elif len(pred_raw) == n_samples:
            # Model returns class indices directly
            pred_encoded = pred_raw.astype(int)
            pred_probs = np.zeros((n_samples, num_classes))
            pred_probs[np.arange(n_samples), np.clip(pred_encoded, 0, num_classes-1)] = 1.0
        else:
            raise ValueError(f"Unexpected prediction shape: {pred_raw.shape}")
    else:
        pred_logits = pred_raw
    
    if 'pred_logits' in dir():
        # Apply softmax
        pred_exp = np.exp(pred_logits - pred_logits.max(axis=1, keepdims=True))
        pred_probs = pred_exp / pred_exp.sum(axis=1, keepdims=True)
        pred_encoded = pred_probs.argmax(axis=1)
    
    # Decode to original config_id
    pred_config_ids = le.inverse_transform(pred_encoded)
    
    # Config accuracy
    actual_config_ids = test_df['config_id'].values
    config_accuracy = (pred_config_ids == actual_config_ids).mean()
    print(f"\nConfig accuracy: {config_accuracy:.2%}")
    
    # Decode predictions to parameters
    results = test_df[['trace', 'ratio', 's_param', 'm_param', 't_param', 'g_param', 
                       'miss_ratio', 'fifo_miss_ratio']].copy()
    results = results.rename(columns={'miss_ratio': 'optimal_miss_ratio'})
    
    pred_configs = [id_to_config[pid] for pid in pred_config_ids]
    results['pred_s_param'] = [c[0] for c in pred_configs]
    results['pred_m_param'] = [c[1] for c in pred_configs]
    results['pred_t_param'] = [c[2] for c in pred_configs]
    results['pred_g_param'] = [c[3] for c in pred_configs]
    results['pred_config_id'] = pred_config_ids
    results['pred_confidence'] = pred_probs.max(axis=1)
    
    # Look up predicted miss_ratio from grid_full
    print("Looking up predicted miss ratios from grid_full...")
    grid_full_df['g_param_rounded'] = grid_full_df['g_param'].round(2)
    
    lookup_dict = {}
    for _, row in grid_full_df.iterrows():
        key = (row['trace'], row['ratio'], row['s_param'], row['m_param'], 
               int(row['t_param']), row['g_param_rounded'])
        lookup_dict[key] = row['miss_ratio']
    
    predicted_mrs = []
    for idx, row in results.iterrows():
        key = (row['trace'], row['ratio'], row['pred_s_param'], row['pred_m_param'], 
               int(row['pred_t_param']), round(row['pred_g_param'], 2))
        
        if key in lookup_dict:
            predicted_mrs.append(lookup_dict[key])
        else:
            trace_grid = grid_full_df[
                (grid_full_df['trace'] == row['trace']) & 
                (grid_full_df['ratio'] == row['ratio'])
            ]
            if len(trace_grid) > 0:
                dists = (
                    (trace_grid['s_param'] - row['pred_s_param'])**2 +
                    (trace_grid['m_param'] - row['pred_m_param'])**2 +
                    (trace_grid['t_param'] - row['pred_t_param'])**2 +
                    ((trace_grid['g_param'] - row['pred_g_param'])/5.0)**2
                )
                closest_idx = dists.idxmin()
                predicted_mrs.append(trace_grid.loc[closest_idx, 'miss_ratio'])
            else:
                predicted_mrs.append(np.nan)
    
    results['predicted_miss_ratio'] = predicted_mrs
    results = results.dropna()
    
    # Calculate metrics
    results['optimal_vs_fifo_pct'] = (results['fifo_miss_ratio'] - results['optimal_miss_ratio']) / results['fifo_miss_ratio'] * 100
    results['predicted_vs_fifo_pct'] = (results['fifo_miss_ratio'] - results['predicted_miss_ratio']) / results['fifo_miss_ratio'] * 100
    results['gap_to_optimal_pct'] = (results['predicted_miss_ratio'] - results['optimal_miss_ratio']) / results['optimal_miss_ratio'] * 100
    
    # Print evaluation
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOverall Performance vs FIFO:")
    print(f"  Ground Truth: {results['optimal_vs_fifo_pct'].mean():.2f}% ± {results['optimal_vs_fifo_pct'].std():.2f}%")
    print(f"  Predicted: {results['predicted_vs_fifo_pct'].mean():.2f}% ± {results['predicted_vs_fifo_pct'].std():.2f}%")
    
    print(f"\nTail Cases (focus area):")
    print(f"  Ground Truth P1 (worst 1%): {np.percentile(results['optimal_vs_fifo_pct'], 1):.2f}%")
    print(f"  Predicted P1: {np.percentile(results['predicted_vs_fifo_pct'], 1):.2f}%")
    print(f"  Ground Truth worst: {results['optimal_vs_fifo_pct'].min():.2f}%")
    print(f"  Predicted worst: {results['predicted_vs_fifo_pct'].min():.2f}%")
    
    print(f"\nGap to Ground Truth:")
    print(f"  Mean: {results['gap_to_optimal_pct'].mean():.2f}%")
    print(f"  Median: {results['gap_to_optimal_pct'].median():.2f}%")
    print(f"  P90: {np.percentile(results['gap_to_optimal_pct'], 90):.2f}%")
    print(f"  P99: {np.percentile(results['gap_to_optimal_pct'], 99):.2f}%")
    
    print(f"\nBy Cache Ratio:")
    for ratio in sorted(results['ratio'].unique()):
        rd = results[results['ratio'] == ratio]
        print(f"  Ratio {ratio}:")
        print(f"    GT vs FIFO: {rd['optimal_vs_fifo_pct'].mean():.2f}%")
        print(f"    Pred vs FIFO: {rd['predicted_vs_fifo_pct'].mean():.2f}%")
        print(f"    Gap: {rd['gap_to_optimal_pct'].mean():.2f}% (median: {rd['gap_to_optimal_pct'].median():.2f}%)")
    
    return results


def main():
    print("="*80)
    print("Standalone Evaluation - Loading saved model")
    print("="*80)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'xgb_config_results', 'xgboost_config_model.json')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # 1. Load data
    print("\n[1] Loading data...")
    features_df, optimal_df, grid_full_df, fifo_df = load_all_data()
    
    # 2. Create config mapping
    print("\n[2] Creating configuration mapping...")
    config_to_id, id_to_config = create_config_mapping(optimal_df)
    
    # 3. Feature engineering
    print("\n[3] Feature engineering...")
    features_eng = engineer_features(features_df)
    
    # 4. Prepare data
    print("\n[4] Preparing data...")
    merged_df = prepare_data(features_eng, optimal_df, fifo_df, config_to_id)
    
    # 5. Split using same seed to get same test set
    print("\n[5] Splitting train/test...")
    train_df, test_df = split_by_trace(merged_df, test_size=0.2)
    
    # 6. Get feature columns
    exclude_cols = ['trace', 'ratio', 's_param', 'm_param', 't_param', 'g_param', 'k_param',
                   'miss_ratio', 'fifo_miss_ratio', 'tail_weight', 'config_id', 'relative_improvement']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    # 7. Load model
    print(f"\n[6] Loading model from {model_path}...")
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Create label encoder (must match training)
    le = LabelEncoder()
    le.fit(train_df['config_id'])
    print(f"Label encoder has {len(le.classes_)} classes")
    
    # 8. Evaluate
    print("\n[7] Evaluating...")
    results = evaluate_model(model, le, test_df, feature_cols, id_to_config, grid_full_df)
    
    # 9. Save results
    output_dir = os.path.join(script_dir, 'xgb_config_results')
    results.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print(f"\nSaved: {os.path.join(output_dir, 'predictions.csv')}")
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)

if __name__ == '__main__':
    main()
