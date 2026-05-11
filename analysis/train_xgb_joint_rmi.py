#!/usr/bin/env python3
"""
Optimized XGBoost model: Joint Configuration Class Prediction WITH RMI
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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
    def normalize_trace(t):
        t = str(t)
        if '.oracleGeneral' in t:
            t = t.split('.oracleGeneral')[0]
        if t.endswith('.zst'): t = t[:-4]
        if t.endswith('.csv'): t = t[:-4]
        if t.endswith('.txt'): t = t[:-4]
        return t

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
    """Enhanced feature engineering from user script"""
    df = df.copy()
    
    # Basic derived features
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
        if not cols: continue
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
        if f'hist_small_{i}' in df.columns and f'hist_ghost_{i}' in df.columns:
            df[f'sg_combined_{i}'] = df[f'hist_small_{i}'] + df[f'hist_ghost_{i}']
            df[f'sg_ratio_{i}'] = df[f'hist_small_{i}'] / (df[f'hist_ghost_{i}'] + 1e-10)
    
    weights = np.exp(-np.arange(20) / 5.0)
    for prefix, cols in [('ghost', ghost_cols), ('main', main_cols), ('small', small_cols)]:
        if not cols: continue
        weighted = 0
        for c in cols:
            idx = int(c.split('_')[-1])
            if idx < len(weights):
                weighted += df[c] * weights[idx]
        df[f'{prefix}_weighted'] = weighted
    
    df['log_ratio'] = np.log10(df['ratio'])
    df['sqrt_ratio'] = np.sqrt(df['ratio'])
    df['log_C_squared'] = df['log_C'] ** 2
    df['log_C_ratio'] = df['log_C'] * df['log_ratio']
    
    df['hits_ghost_entropy'] = df['hits_ghost'] * df['H_g']
    df['hits_main_entropy'] = df['hits_main'] * df['H_m']
    df['hits_small_entropy'] = df['hits_small'] * df['H_s']
    
    print(f"Feature engineering: {len(df.columns)} total features")
    return df

def prepare_data(features_df, optimal_df, fifo_df, config_to_id):
    """Merge and prepare training data with config_id target"""
    merged = pd.merge(features_df, optimal_df, on=['trace', 'ratio'], how='inner')
    merged = pd.merge(merged, fifo_df, on=['trace', 'ratio'], how='inner')
    
    def get_config_id(row):
        key = (row['s_param'], int(row['m_param']), int(row['t_param']), row['g_param'])
        return config_to_id.get(key, -1)
    
    merged['config_id'] = merged.apply(get_config_id, axis=1)
    merged = merged[merged['config_id'] >= 0]
    
    # Weights for custom objective
    merged['relative_improvement'] = (merged['fifo_miss_ratio'] - merged['miss_ratio']) / merged['fifo_miss_ratio']
    merged['tail_weight'] = np.clip(merged['fifo_miss_ratio'] / merged['fifo_miss_ratio'].median(), 0.5, 10.0)
    improvement_percentile = merged['relative_improvement'].rank(pct=True)
    merged['tail_weight'] *= np.where(improvement_percentile < 0.2, 2.0, 1.0)
    
    print(f"Prepared {len(merged)} samples with {merged['config_id'].nunique()} unique configs")
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

def build_empirical_cost_matrix(train_df, grid_full_df, id_to_config, le):
    """
    Build cost matrix C[k, j] based on ACTUAL performance degradation.
    C[k, j] = Avg Regret when predicting k given True is j.
    Regret = (MR_pred - MR_opt) / FIFO_MR
    """
    print("Building empirical cost matrix for RMI...")
    
    # 1. Map grid_full to config_ids
    config_to_id = {v: k for k, v in id_to_config.items()}
    grid_full_df['g_param_rounded'] = grid_full_df['g_param'].round(2)
    
    def get_grid_cid(row):
        key = (row['s_param'], int(row['m_param']), int(row['t_param']), row['g_param_rounded'])
        return config_to_id.get(key, -1)
        
    grid_full_df['config_id'] = grid_full_df.apply(get_grid_cid, axis=1)
    
    # Filter to only relevant config_ids (those in training set labels)
    valid_original_ids = set(le.classes_)
    # Note: grid_full might not have all configs for all traces, but we iterate over what we have
    relevant_grid = grid_full_df[grid_full_df['config_id'].isin(valid_original_ids)].copy()
    
    original_to_encoded = {original: encoded for encoded, original in enumerate(le.classes_)}
    relevant_grid['encoded_id'] = relevant_grid['config_id'].map(original_to_encoded)
    
    # Pivot: index=[trace, ratio], columns=[encoded_id], values=miss_ratio
    print("  Pivoting grid data...")
    mr_matrix = relevant_grid.pivot_table(
        index=['trace', 'ratio'], 
        columns='encoded_id', 
        values='miss_ratio'
    )
    
    targets = train_df[['trace', 'ratio', 'config_id', 'miss_ratio', 'fifo_miss_ratio']].copy()
    targets['encoded_true_label'] = targets['config_id'].map(original_to_encoded)
    targets = targets.set_index(['trace', 'ratio'])
    
    joined = targets.join(mr_matrix, how='inner')
    
    n_classes = len(le.classes_)
    C = np.zeros((n_classes, n_classes))
    
    print(f"  Calculating cost for {n_classes} classes...")
    
    for j in range(n_classes): # j is encoded true class
        subset = joined[joined['encoded_true_label'] == j]
        if len(subset) == 0:
            continue
            
        fifo_mrs = subset['fifo_miss_ratio'].values
        opt_mrs = subset['miss_ratio'].values
        
        for k in range(n_classes): # k is encoded predicted class
            if k not in subset.columns:
                C[k, j] = 1.0 
                continue
                
            pred_mrs = subset[k].values
            regret = (pred_mrs - opt_mrs) / (fifo_mrs + 1e-9)
            regret = np.maximum(regret, 0)
            C[k, j] = np.mean(regret)
            
    print(f"  Cost Matrix Stats: Min={C.min():.4f}, Max={C.max():.4f}, Mean={C.mean():.4f}")
    return C

def train_config_model(train_df, feature_cols, grid_full_df, id_to_config):
    X_train = train_df[feature_cols]
    y_train = train_df['config_id']
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    num_classes_actual = len(le.classes_)
    
    print(f"\nTraining config classifier with {num_classes_actual} classes (re-encoded)...")
    
    grid_full_df['g_param_rounded'] = grid_full_df['g_param'].round(2)
    config_to_id = {v: k for k, v in id_to_config.items()}
    
    def get_grid_config_id(row):
        key = (row['s_param'], int(row['m_param']), int(row['t_param']), row['g_param_rounded'])
        return config_to_id.get(key, -1)
        
    grid_full_df['config_id'] = grid_full_df.apply(get_grid_config_id, axis=1)
    
    mr_lookup = {}
    for _, row in grid_full_df.iterrows():
        key = (row['trace'], row['ratio'], row['config_id'])
        mr_lookup[key] = row['miss_ratio']
    
    print("Computing cost matrix for FIFO-relative loss (Training)...")
    n_samples = len(train_df)
    train_cost_matrix = np.zeros((n_samples, num_classes_actual))
    
    for i, (idx, row) in enumerate(train_df.iterrows()):
        trace, ratio, fifo_mr = row['trace'], row['ratio'], row['fifo_miss_ratio']
        for encoded_cid in range(num_classes_actual):
            original_cid = le.classes_[encoded_cid]
            key = (trace, ratio, original_cid)
            if key in mr_lookup:
                pred_mr = mr_lookup[key]
                relative_vs_fifo = (pred_mr - fifo_mr) / (fifo_mr + 1e-10)
                train_cost_matrix[i, encoded_cid] = max(0, relative_vs_fifo) ** 2
            else:
                train_cost_matrix[i, encoded_cid] = 1.0
                
    max_costs = train_cost_matrix.max(axis=1)
    sample_weights = 1.0 + 10.0 * max_costs
    
    def fifo_relative_objective(y_pred, dtrain):
        labels = dtrain.get_label().astype(int)
        n = len(labels)
        preds = y_pred.reshape(n, num_classes_actual)
        preds_max = preds.max(axis=1, keepdims=True)
        preds_exp = np.exp(preds - preds_max)
        preds_prob = preds_exp / preds_exp.sum(axis=1, keepdims=True)
        
        cost_weights = 1 + train_cost_matrix[:n]
        grad = preds_prob * cost_weights
        true_mask = np.zeros_like(grad)
        true_mask[np.arange(n), labels] = 1.0
        grad = grad - true_mask * cost_weights
        hess = np.maximum(preds_prob * (1 - preds_prob) * cost_weights, 1e-6)
        
        return grad.reshape(n, num_classes_actual), hess.reshape(n, num_classes_actual)
    
    dtrain = xgb.DMatrix(X_train, label=y_train_encoded, weight=sample_weights)
    
    params = {
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'seed': 42,
        'num_class': num_classes_actual,
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=150,
        obj=fifo_relative_objective,
        verbose_eval=10
    )
    
    return model, le

def evaluate_model_rmi(model, le, test_df, feature_cols, id_to_config, grid_full_df, emp_cost_matrix):
    X_test = test_df[feature_cols]
    dtest = xgb.DMatrix(X_test)
    pred_raw = model.predict(dtest)
    
    num_classes = len(le.classes_)
    pred_logits = pred_raw.reshape(-1, num_classes)
    pred_exp = np.exp(pred_logits - pred_logits.max(axis=1, keepdims=True))
    pred_probs = pred_exp / pred_exp.sum(axis=1, keepdims=True)
    
    print("\nApplying Risk-Minimizing Inference...")
    expected_risk = np.dot(pred_probs, emp_cost_matrix.T)
    pred_encoded_rmi = expected_risk.argmin(axis=1)
    
    pred_encoded_std = pred_probs.argmax(axis=1)
    change_pct = (pred_encoded_rmi != pred_encoded_std).mean() * 100
    print(f"  RMI changed {change_pct:.1f}% of predictions vs Standard")
    
    pred_config_ids = le.inverse_transform(pred_encoded_rmi)
    
    results = test_df[['trace', 'ratio', 's_param', 'm_param', 't_param', 'g_param', 
                       'miss_ratio', 'fifo_miss_ratio']].copy()
    results = results.rename(columns={'miss_ratio': 'optimal_miss_ratio'})
    
    pred_configs = [id_to_config[pid] for pid in pred_config_ids]
    results['pred_s_param'] = [c[0] for c in pred_configs]
    results['pred_m_param'] = [c[1] for c in pred_configs]
    results['pred_t_param'] = [c[2] for c in pred_configs]
    results['pred_g_param'] = [c[3] for c in pred_configs]
    results['pred_config_id'] = pred_config_ids
    
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
                predicted_mrs.append(trace_grid.loc[dists.idxmin(), 'miss_ratio'])
            else:
                predicted_mrs.append(row['fifo_miss_ratio'])

    results['predicted_miss_ratio'] = predicted_mrs
    
    results['optimal_vs_fifo_pct'] = (results['fifo_miss_ratio'] - results['optimal_miss_ratio']) / results['fifo_miss_ratio'] * 100
    results['predicted_vs_fifo_pct'] = (results['fifo_miss_ratio'] - results['predicted_miss_ratio']) / results['fifo_miss_ratio'] * 100
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS (RMI)")
    print("="*80)
    
    print(f"\nOverall Performance vs FIFO:")
    print(f"  Ground Truth: {results['optimal_vs_fifo_pct'].mean():.2f}%")
    print(f"  Predicted:    {results['predicted_vs_fifo_pct'].mean():.2f}%")
    
    print(f"\nTail Cases (focus area):")
    print(f"  Ground Truth P1: {np.percentile(results['optimal_vs_fifo_pct'], 1):.2f}%")
    print(f"  Predicted P1:    {np.percentile(results['predicted_vs_fifo_pct'], 1):.2f}%")
    print(f"  Ground Truth Worst: {results['optimal_vs_fifo_pct'].min():.2f}%")
    print(f"  Predicted Worst:    {results['predicted_vs_fifo_pct'].min():.2f}%")
    
    return results

def main():
    print("="*80)
    print("Optimized XGBoost: Joint Configuration + RMI")
    print("="*80)
    
    features_df, optimal_df, grid_full_df, fifo_df = load_all_data()
    config_to_id, id_to_config = create_config_mapping(optimal_df)
    features_eng = engineer_features(features_df)
    merged_df = prepare_data(features_eng, optimal_df, fifo_df, config_to_id)
    
    train_df, test_df = split_by_trace(merged_df, test_size=0.2)
    
    exclude_cols = ['trace', 'ratio', 's_param', 'm_param', 't_param', 'g_param', 'k_param',
                   'miss_ratio', 'fifo_miss_ratio', 'tail_weight', 'config_id', 'relative_improvement']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    model, le = train_config_model(train_df, feature_cols, grid_full_df, id_to_config)
    
    cost_matrix = build_empirical_cost_matrix(train_df, grid_full_df, id_to_config, le)
    
    results = evaluate_model_rmi(model, le, test_df, feature_cols, id_to_config, grid_full_df, cost_matrix)
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgb_joint_rmi_results')
    os.makedirs(output_dir, exist_ok=True)
    results.to_csv(os.path.join(output_dir, 'predictions_rmi.csv'), index=False)
    print(f"\nSaved RMI predictions to {output_dir}/predictions_rmi.csv")

if __name__ == '__main__':
    main()
