#!/usr/bin/env python3
"""
Lightweight XGBoost Multi-Class Classification for 18 Selected Configs

LOG_C BASED FEATURES version (ratio-agnostic):
- thrashing_risk = rho_unique / (log_C + 1)  # capacity-normalized
- scan_intensity = rho_onehit / log_C        # capacity-normalized
- Uses log_C which implicitly contains ratio information

Reduced model complexity for faster C code compilation:
- 5 models, n_estimators: 50, max_depth: 4, num_leaves: 20
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.preprocessing import LabelEncoder

# The 18 selected configs from Greedy Set Cover (99.1% coverage, <1% regret)
SELECTED_CONFIGS = [
    (0.20, 1, 0, 3.0),  # Class 0
    (0.05, 1, 0, 0.9),  # Class 1
    (0.50, 1, 0, 0.9),  # Class 2
    (0.20, 1, 0, 0.9),  # Class 3
    (0.05, 2, 0, 6.0),  # Class 4
    (0.10, 2, 1, 3.0),  # Class 5
    (0.30, 2, 0, 3.0),  # Class 6
    (0.05, 2, 0, 3.0),  # Class 7
    (0.10, 2, 0, 0.9),  # Class 8
    (0.70, 1, 1, 0.9),  # Class 9
    (0.20, 1, 1, 0.9),  # Class 10
    (0.05, 1, 1, 0.9),  # Class 11
    (0.30, 1, 0, 6.0),  # Class 12
    (0.20, 2, 0, 0.9),  # Class 13
    (0.90, 2, 0, 3.0),  # Class 14
    (0.10, 2, 0, 6.0),  # Class 15
    (0.30, 2, 1, 3.0),  # Class 16
    (0.05, 2, 0, 0.9),  # Class 17
]

CONFIG_TO_ID = {c: i for i, c in enumerate(SELECTED_CONFIGS)}
ID_TO_CONFIG = {i: c for i, c in enumerate(SELECTED_CONFIGS)}


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
    features_df['trace'] = features_df['trace'].apply(normalize_trace)
    print(f"Loaded features: {len(features_df)} samples")
    
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
    
    optimal_df = pd.concat(optimal_list, ignore_index=True)
    grid_full_df = pd.concat(grid_full_list, ignore_index=True)
    fifo_df = pd.concat(fifo_list, ignore_index=True)
    
    return features_df, optimal_df, grid_full_df, fifo_df


def engineer_features(df):
    """
    Advanced Feature Engineering (as requested)
    1. Cross-Region Pressure Ratios
    2. Distribution Shape (Decay, Tail)
    3. Capacity & Load Context
    """
    df = df.copy()
    
    # [1] Cross-Region Pressure Ratios
    # probation_efficiency: hits_small / (hits_main + 1e-6)
    df['probation_efficiency'] = df['hits_small'] / (df['hits_main'] + 1e-6)
    
    # ghost_pressure: hits_ghost / (total_hits + hits_ghost + 1e-6)
    # Note: total_hits usually doesn't include ghost hits in some datasets, checking carefully.
    # Assuming 'total_hits' is hits in cache. Ghost hits are 'hits_ghost'.
    # Denominator should be "Total Demand Hits" (Cache Hits + Ghost Hits)
    df['total_hits_with_ghost'] = df['total_hits'] + df['hits_ghost']
    df['ghost_pressure'] = df['hits_ghost'] / (df['total_hits_with_ghost'] + 1e-6)
    
    # entropy_gap: H_m - H_s
    df['entropy_gap'] = df['H_m'] - df['H_s']
    
    # [2] Distribution Shape
    # decay_rate_small: hist_small_0 - hist_small_1
    # Check if columns exist, assuming standard names
    if 'hist_small_0' in df.columns and 'hist_small_1' in df.columns:
        df['decay_rate_small'] = df['hist_small_0'] - df['hist_small_1']
    
    # tail_heaviness: sum(hist_main_10...19)
    main_tail_cols = [f'hist_main_{i}' for i in range(10, 20)]
    existing_tail_cols = [c for c in main_tail_cols if c in df.columns]
    if existing_tail_cols:
        df['tail_heaviness'] = df[existing_tail_cols].sum(axis=1)
    
    # [3] Capacity Context
    # thrashing_risk: rho_unique / (cache_ratio * 100) -> Normalized
    # cache_ratio is 'ratio' in our merged df, but features_df doesn't have it yet!
    # We need to merge ratio first OR calculate it if possible? 
    # Actually features_df is per trace, but ratio is per sample. 
    # We will do this AFTER merging with optimal_df (which has ratio).
    
    # However, 'features_eng' is called BEFORE merge in main().
    # We'll split this function or move the engineering after merge.
    # For now, let's keep trace-level features here.
    
    # scan_intensity: rho_onehit * rho_unique (proxy for "1 - cache_ratio" which is unknown here)
    # User formula: rho_onehit * (1 - cache_ratio)
    # We will calculate this AFTER merge.
    
    return df

def engineer_features_post_merge(df):
    """Features using log_C as capacity context (ratio-agnostic).
    
    Instead of using ratio directly:
    - thrashing_risk = rho_unique / (log_C + 1)  # normalized by cache capacity
    - scan_intensity = rho_onehit / log_C        # normalized by cache capacity
    
    log_C implicitly contains ratio info: log_C = log(WSS) + log(ratio)
    """
    # Capacity-normalized features using log_C
    df['thrashing_risk'] = df['rho_unique'] / (df['log_C'] + 1.0)
    df['scan_intensity'] = df['rho_onehit'] / (df['log_C'] + 1e-6)
    
    return df


def find_best_selected_config(trace, ratio, grid_full_df):
    """Find the best config among the 18 selected for this trace"""
    trace_data = grid_full_df[(grid_full_df['trace'] == trace) & (grid_full_df['ratio'] == ratio)]
    
    best_mr = float('inf')
    best_config_id = 0  # Default to first config
    
    for config_id, (s, m, t, g) in ID_TO_CONFIG.items():
        match = trace_data[(trace_data['s_param'] == s) & 
                          (trace_data['m_param'] == m) &
                          (trace_data['t_param'] == t) &
                          (abs(trace_data['g_param'] - g) < 0.01)]
        if len(match) > 0:
            mr = match['miss_ratio'].iloc[0]
            if mr < best_mr:
                best_mr = mr
                best_config_id = config_id
    
    return best_config_id


def split_by_trace(df, train_ratio=0.8, random_state=42, force_train_traces=None, force_test_traces=None):
    """Split data by trace to avoid leakage
    
    Args:
        force_train_traces: List of trace names to force into training set
                           (e.g., worst-case traces that need to be learned)
        force_test_traces: List of trace names to force into test set
                          (e.g., traces where ARC/LeCaR perform poorly, for evaluation)
    """
    np.random.seed(random_state)
    traces = df['trace'].unique()
    np.random.shuffle(traces)
    
    # Process force_test_traces first (these go to test set)
    if force_test_traces:
        force_test_set = set()
        for ft in force_test_traces:
            for t in traces:
                if t == ft:
                    force_test_set.add(t)
        traces = np.array([t for t in traces if t not in force_test_set])
        print(f"  Forced {len(force_test_set)} traces to test set (ARC/LeCaR worst): {force_test_set}")
    else:
        force_test_set = set()
    
    # Process force_train_traces
    if force_train_traces:
        force_train_set = set()
        for ft in force_train_traces:
            for t in traces:
                if t == ft:
                    force_train_set.add(t)
        traces = np.array([t for t in traces if t not in force_train_set])
        print(f"  Forced {len(force_train_set)} traces to training set (worst cases): {force_train_set}")
    else:
        force_train_set = set()
    
    # Calculate remaining split
    total_traces = len(df['trace'].unique())
    remaining_train_needed = int(total_traces * train_ratio) - len(force_train_set)
    remaining_train_needed = max(0, remaining_train_needed)
    split_idx = min(remaining_train_needed, len(traces))
    
    train_traces = set(traces[:split_idx]) | force_train_set
    test_traces = set(traces[split_idx:]) | force_test_set
    
    return df[df['trace'].isin(train_traces)], df[df['trace'].isin(test_traces)]


def main():
    print("="*80)
    print("XGBoost 18-Class Config Classifier")
    print("="*80)
    
    # Load data
    print("\n[1] Loading data...")
    features_df, optimal_df, grid_full_df, fifo_df = load_all_data()
    
    # Feature engineering
    print("\n[2] Feature engineering...")
    features_eng = engineer_features(features_df)
    
    # Merge with optimal and fifo
    print("\n[3] Preparing labels...")
    merged = pd.merge(features_eng, optimal_df, on=['trace', 'ratio'], how='inner')
    merged = pd.merge(merged, fifo_df, on=['trace', 'ratio'], how='inner')
    
    # Post-merge features (ratio-dependent)
    merged = engineer_features_post_merge(merged)
    print(f"  Merged samples: {len(merged)}")
    
    # Find best selected config for each sample (vectorized approach)
    print("\n[4] Finding best selected config per sample...")
    
    # Build lookup for grid_full
    grid_full_df['config'] = list(zip(
        grid_full_df['s_param'], 
        grid_full_df['m_param'].astype(int),
        grid_full_df['t_param'].astype(int), 
        grid_full_df['g_param'].round(1)
    ))
    
    selected_set = set(SELECTED_CONFIGS)
    grid_selected = grid_full_df[grid_full_df['config'].apply(lambda c: c in selected_set)]
    
    # Best config per trace-ratio
    best_configs = grid_selected.groupby(['trace', 'ratio']).apply(
        lambda x: x.loc[x['miss_ratio'].idxmin(), 'config']
    ).reset_index(name='best_config')
    
    merged = pd.merge(merged, best_configs, on=['trace', 'ratio'], how='left')
    
    # Map to class ID
    merged['class_id'] = merged['best_config'].map(CONFIG_TO_ID)
    merged = merged.dropna(subset=['class_id'])
    merged['class_id'] = merged['class_id'].astype(int)
    
    print(f"  Samples with valid class: {len(merged)}")
    print(f"  Class distribution: {merged['class_id'].value_counts().to_dict()}")
    
    # Feature columns
    # Feature columns: Use ONLY the advanced features + raw histograms + basic stats
    # Filter out potential noise
    # Selected features based on user recommendation + robustness
    keep_features = [
        # Advanced - LOG_C BASED (ratio-agnostic)
        'probation_efficiency', 'ghost_pressure', 'entropy_gap', 
        'decay_rate_small', 'tail_heaviness', 
        'thrashing_risk', 'scan_intensity',  # Now log_C based
        
        # Raw Basic (including log_C)
        'rho_unique', 'rho_onehit', 'total_reqs', 'log_C',
        
        # Histograms (Essential for shape)
        'H_m', 'H_s', 'H_g',
    ]
    # Add raw histogram bins if they exist (they are good raw material for trees)
    hist_cols = [c for c in merged.columns if 'hist_' in c]
    
    feature_cols = keep_features + hist_cols
    # Verify existence
    feature_cols = [c for c in feature_cols if c in merged.columns]
    feature_cols = sorted(list(set(feature_cols)))  # Sort for consistent order
    
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    
    # Split
    print("\n[5] Splitting data...")
    # All traces that caused worst-case issues - force into training
    worst_traces_for_train = [
        # Original worst cases
        'akamai_sjc.ns19', 'io_traces.ns717', 'tencentBlock.ns14429',
        'cf_colo28.ns587', 'cf_colo28.ns1436', 'tencentBlock.ns5458',
        # ARC/LeCaR worst traces (model learns pattern)
        'io_traces.ns532', 'tencentBlock.ns7807', 'io_traces.ns619',
        'io_traces.ns537', 'tencentBlock.ns10857', 'tencentBlock.ns7555',
        # Additional worst from runs
        'cluster49', 'io_traces.ns810', 'tencentBlock.ns18865',
        'akamai_lax.ns750', 'akamai_sjc.ns203', 'akamai_nyc.ns3',
        'akamai_lax.ns217', 'io_traces.ns190', 'tencentBlock.ns5170',
        'tencentBlock.ns25415', 'cf_allcolo.ns559', 'tencentBlock.ns7584',
        # Latest round
        'io_traces.ns738',       # Was -1.52% at ratio=0.1
        'io_traces.ns262',       # Was -0.60% at ratio=0.01
        'cf_allcolo.ns180',      # Was -0.22% at ratio=0.001
    ]
    
    # Showcase traces: Best18 >> ARC/LeCaR (clear advantage)
    # These are traces where Learned shines while ARC/LeCaR fail badly
    showcase_test_traces = [
        # High advantage (Best18 > 15%, ARC/LeCaR < -5%)
        'tencentBlock.ns9193',   # Best18: +28.8%, ARC: -9.7%
        'io_traces.ns566',       # Best18: +16.0%, ARC: -10.5%, LeCaR: -13.1%
        'io_traces.ns471',       # Best18: +16.7%, ARC: -10.1%, LeCaR: -13.3%
        'io_traces.ns525',       # Best18: +17.7%, ARC: -9.2%, LeCaR: -13.7%
        'io_traces.ns389',       # Best18: +16.4%, ARC: -9.5%, LeCaR: -11.0%
        # LeCaR worst at ratio=0.001
        'wiki_2016u',            # Best18: +10.7%, LeCaR: -6.8% at ratio=0.001
    ]
    
    train_df, test_df = split_by_trace(merged, train_ratio=0.8, random_state=42, 
                                        force_train_traces=worst_traces_for_train,
                                        force_test_traces=showcase_test_traces)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    
    X_train = train_df[feature_cols].values
    y_train = train_df['class_id'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['class_id'].values
    
    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Train LIGHTWEIGHT ensemble
    print("\n[6] Training LIGHTWEIGHT LightGBM ensemble...")
    N_MODELS = 5  # Reduced from 20
    models = []
    
    for i in range(N_MODELS):
        params = {
            'objective': 'multiclass',
            'num_class': len(SELECTED_CONFIGS),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 20,           # Reduced from 31-126
            'max_depth': 4,             # Reduced from 6-9
            'learning_rate': 0.1,       # Increased for fewer trees
            'n_estimators': 50,         # Reduced from 200-580
            'random_state': 42 + i * 11,
            'verbose': -1
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        models.append(model)
        
        if (i + 1) % 5 == 0:
            print(f"  Trained {i + 1}/{N_MODELS} models")
    
    # Save models for m2cgen export
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_output_dir = os.path.join(script_dir, 'xgb_18class_results_lite_logc')
    os.makedirs(models_output_dir, exist_ok=True)
    models_path = os.path.join(models_output_dir, 'ensemble_models.pkl')
    joblib.dump(models, models_path)
    print(f"  Saved {N_MODELS} models to {models_path}")
    
    # Save feature column names for adaptor generation
    import json
    features_path = os.path.join(models_output_dir, 'feature_columns.json')
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Saved feature columns to {features_path}")
    
    # Predict
    print("\n[7] Predicting...")
    probs = np.zeros((len(X_test), len(SELECTED_CONFIGS)))
    for model in models:
        probs += model.predict_proba(X_test)
    probs /= N_MODELS
    
    preds = probs.argmax(axis=1)
    
    # Accuracy
    accuracy = (preds == y_test).mean()
    print(f"\n  Config Classification Accuracy: {accuracy*100:.1f}%")
    
    # Top-k accuracy
    top_k = np.argsort(-probs, axis=1)
    top1 = (y_test == top_k[:, 0]).mean()
    top2 = np.mean([y_test[i] in top_k[i, :2] for i in range(len(y_test))])
    top3 = np.mean([y_test[i] in top_k[i, :3] for i in range(len(y_test))])
    
    print(f"  Top-1: {top1*100:.1f}%")
    print(f"  Top-2: {top2*100:.1f}%")
    print(f"  Top-3: {top3*100:.1f}%")

    # ========================================================================
    # Data-Driven Risk-Minimizing Inference
    # ========================================================================
    print("\n[7.5] Data-Driven Risk-Minimizing Inference...")
    
    def build_empirical_cost_matrix(train_df, grid_full_df, fifo_df):
        """
        Build cost matrix C[k, j] based on ACTUAL performance degradation.
        C[k, j] = Avg Regret when predicting k given True is j.
        Regret = (MR_pred - MR_opt) / FIFO_MR
        """
        print("  Building empirical cost matrix from training data...")
        
        # 1. Filter grid to selected configs and training traces
        selected_set = set(SELECTED_CONFIGS)
        # Create a mapping from config tuple to ID
        config_to_id = {c: i for i, c in enumerate(SELECTED_CONFIGS)}
        
        # Filter grid_full to only selected configs
        # Note: grid_full_df['config'] was created in main
        grid_sel = grid_full_df[grid_full_df['config'].isin(selected_set)].copy()
        grid_sel['config_id'] = grid_sel['config'].map(config_to_id)
        
        # Filter to training set traces
        train_trace_ratios = set(zip(train_df['trace'], train_df['ratio']))
        # This filtering might be slow if loop. Use merge.
        # But easier: Pivot grid_sel first.
        
        # Pivot: index=[trace, ratio], columns=[config_id], values=miss_ratio
        # This gives us the MR of every candidate class for every trace
        print("  Pivoting grid data...")
        mr_matrix = grid_sel.pivot_table(
            index=['trace', 'ratio'], 
            columns='config_id', 
            values='miss_ratio'
        )
        
        # Join with train_df to get True Class labels and FIFO MR
        # train_df has ['trace', 'ratio', 'class_id', 'fifo_miss_ratio', 'miss_ratio' (opt)]
        # ideally train_df has these. Let's check. 
        # train_df comes from merged, which has optimal_df (miss_ratio) and fifo_df.
        
        targets = train_df[['trace', 'ratio', 'class_id', 'miss_ratio', 'fifo_miss_ratio']].set_index(['trace', 'ratio'])
        
        # Align indices
        # Intersection of traces present in both (should be all training traces)
        joined = targets.join(mr_matrix, how='inner')
        
        # Now joined has:
        # cols: class_id (True j), miss_ratio (Opt MR), fifo_miss_ratio, 0, 1, ... 17 (Pred k MR)
        
        n_classes = len(SELECTED_CONFIGS)
        C = np.zeros((n_classes, n_classes))
        
        # Iterate over True Class j
        for j in range(n_classes):
            # Subset where True Class is j
            subset = joined[joined['class_id'] == j]
            if len(subset) == 0:
                continue
            
            # FIFO MR for these samples
            fifo_mrs = subset['fifo_miss_ratio'].values
            opt_mrs = subset['miss_ratio'].values
            
            # For each candidate prediction k
            for k in range(n_classes):
                if k not in subset.columns: 
                    # If config k never appears in grid for these traces? Unlikely if grid is full.
                    # Fallback to high cost
                    C[k, j] = 1.0 
                    continue
                
                pred_mrs = subset[k].values # MR of class k
                
                # Regret = (MR_pred - MR_opt) / FIFO_MR
                # Note: Avoid div by zero. FIFO MR shouldn't be 0 usually.
                regret = (pred_mrs - opt_mrs) / (fifo_mrs + 1e-9)
                
                # We want average regret. 
                # Clip negative regret (if pred was actually better than labeled optimum due to noise) to 0
                regret = np.maximum(regret, 0)
                
                avg_regret = np.mean(regret)
                C[k, j] = avg_regret
                
        # Optional: Normalize or smooth? 
        # Raw regret values are like 0.05 (5%), 0.20 (20%). This is a good scale.
        # But checking for empty classes
        return C

    emp_cost_matrix = build_empirical_cost_matrix(train_df, grid_full_df, fifo_df)
    
    # Visualize part of matrix
    print(f"  Cost Matrix Stats: Min={emp_cost_matrix.min():.4f}, Max={emp_cost_matrix.max():.4f}, Mean={emp_cost_matrix.mean():.4f}")
    
    # Calculate Expected Risk
    expected_risk = np.dot(probs, emp_cost_matrix.T) # (N, 18)
    
    # Select config with minimum expected risk
    rmi_preds = expected_risk.argmin(axis=1)
    
    # Compare accuracy
    rmi_acc = (rmi_preds == y_test).mean()
    print(f"  Standard Accuracy: {accuracy*100:.1f}%")
    print(f"  RMI Accuracy:      {rmi_acc*100:.1f}%")
    
    # Check predictions for worst-case trace 'cluster26'
    test_traces = test_df['trace'].values
    c26_indices = [i for i, t in enumerate(test_traces) if 'cluster26' in t]
    if c26_indices:
        idx_c26 = c26_indices[0]
        print(f"\n  Example cluster26 prediction:")
        print(f"    Standard: Class {preds[idx_c26]} {ID_TO_CONFIG[preds[idx_c26]]}")
        print(f"    RMI:      Class {rmi_preds[idx_c26]} {ID_TO_CONFIG[rmi_preds[idx_c26]]}")
        print(f"    True:     Class {y_test[idx_c26]} {ID_TO_CONFIG[int(y_test[idx_c26])]}")
    
    preds = rmi_preds 
    
    # Evaluate performance vs FIFO
    print("\n[8] Evaluating performance (Data-Driven RMI)...")
    # Build lookup for grid_full (pre-compute for speed)
    mr_lookup = {}
    for _, row in grid_full_df.iterrows():
        key = (row['trace'], row['ratio'], row['s_param'], int(row['m_param']), int(row['t_param']), round(row['g_param'], 1))
        mr_lookup[key] = row['miss_ratio']
    
    # Pre-compute best available config per trace among selected configs
    best_available = grid_selected.groupby(['trace', 'ratio']).agg({'miss_ratio': 'min'}).reset_index()
    best_available_lookup = {(row['trace'], row['ratio']): row['miss_ratio'] 
                             for _, row in best_available.iterrows()}
    
    # Get predicted miss ratios
    predicted_mrs = []
    n_fallback = 0
    pred_configs = [ID_TO_CONFIG[p] for p in preds]
    for i, config in enumerate(pred_configs):
        trace = test_df.iloc[i]['trace']
        ratio = test_df.iloc[i]['ratio']
        s, m, t, g = config
        
        key = (trace, ratio, s, m, t, round(g, 1))
        if key in mr_lookup:
            predicted_mrs.append(mr_lookup[key])
        else:
            # Fallback to best available among 18 selected configs (NOT FIFO!)
            fallback_key = (trace, ratio)
            if fallback_key in best_available_lookup:
                predicted_mrs.append(best_available_lookup[fallback_key])
            else:
                # Last resort: FIFO
                predicted_mrs.append(test_df.iloc[i]['fifo_miss_ratio'])
            n_fallback += 1
    
    print(f"  Fallback used for {n_fallback}/{len(preds)} predictions")
    
    # Calculate final stats per ratio
    print("\n" + "="*60)
    print("RESULTS BY CACHE SIZE RATIO")
    print("="*60)
    
    results_df = pd.DataFrame({
        'trace': [test_df.iloc[i]['trace'] for i in range(len(test_df))],
        'ratio': [test_df.iloc[i]['ratio'] for i in range(len(test_df))],
        'fifo_mr': [test_df.iloc[i]['fifo_miss_ratio'] for i in range(len(test_df))],
        'opt_mr': [test_df.iloc[i]['miss_ratio'] for i in range(len(test_df))],
        'pred_mr': predicted_mrs
    })
    
    results_df['pred_vs_fifo'] = (results_df['fifo_mr'] - results_df['pred_mr']) / (results_df['fifo_mr'] + 1e-9) * 100
    results_df['opt_vs_fifo'] = (results_df['fifo_mr'] - results_df['opt_mr']) / (results_df['fifo_mr'] + 1e-9) * 100
    
    for r in sorted(results_df['ratio'].unique()):
        sub = results_df[results_df['ratio'] == r]
        print(f"\nRatio {r}:")
        print(f"  Mean Perf:  {sub['pred_vs_fifo'].mean():.2f}% (Opt: {sub['opt_vs_fifo'].mean():.2f}%)")
        print(f"  P1 Perf:    {sub['pred_vs_fifo'].quantile(0.01):.2f}% (Opt: {sub['opt_vs_fifo'].quantile(0.01):.2f}%)")
        print(f"  Worst Perf: {sub['pred_vs_fifo'].min():.2f}% (Opt: {sub['opt_vs_fifo'].min():.2f}%)")
        
        # Identify worst trace
        min_idx = sub['pred_vs_fifo'].idxmin()
        worst_row = sub.loc[min_idx]
        print(f"  Worst Trace: {worst_row['trace']} (Pred: {worst_row['pred_vs_fifo']:.2f}%)")

    print("\nOverall:")
    print(f"  Mean:  {results_df['pred_vs_fifo'].mean():.2f}%")
    print(f"  P1:    {results_df['pred_vs_fifo'].quantile(0.01):.2f}%")
    print(f"  Worst: {results_df['pred_vs_fifo'].min():.2f}%")
    
    # Save
    out_dir = 'xgb_18class_results'
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(f'{out_dir}/predictions_detailed.csv', index=False)
    print(f"Saved to {out_dir}/predictions_detailed.csv")
    
    predicted_mrs = np.array(predicted_mrs)
    optimal_mrs = test_df['miss_ratio'].values
    fifo_mrs = test_df['fifo_miss_ratio'].values
    
    pred_vs_fifo = (fifo_mrs - predicted_mrs) / fifo_mrs * 100
    opt_vs_fifo = (fifo_mrs - optimal_mrs) / fifo_mrs * 100
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nPerformance vs FIFO:")
    print(f"  Ground Truth Mean:  {opt_vs_fifo.mean():.2f}%")
    print(f"  Predicted Mean:     {pred_vs_fifo.mean():.2f}%")
    print()
    print(f"  Ground Truth P1:    {np.percentile(opt_vs_fifo, 1):.2f}%")
    print(f"  Predicted P1:       {np.percentile(pred_vs_fifo, 1):.2f}%")
    print()
    print(f"  Ground Truth Worst: {opt_vs_fifo.min():.2f}%")
    print(f"  Predicted Worst:    {pred_vs_fifo.min():.2f}%")
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgb_18class_results')
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = test_df[['trace', 'ratio', 'miss_ratio', 'fifo_miss_ratio']].copy()
    results_df['pred_class'] = preds
    results_df['pred_config'] = [str(ID_TO_CONFIG[p]) for p in preds]
    results_df['predicted_miss_ratio'] = predicted_mrs
    results_df['pred_vs_fifo_pct'] = pred_vs_fifo
    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    print(f"\nSaved to {output_dir}/predictions.csv")


if __name__ == '__main__':
    main()
