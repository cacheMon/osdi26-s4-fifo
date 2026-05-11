#!/usr/bin/env python3
"""
XGBoost Regression for predicting optimal S4FIFO parameters.

Uses features from warmup phase to predict optimal (s_param, m_param, t_param, g_param)
and evaluates using miss_ratio from grid_full.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

CACHE_RATIOS = [0.001, 0.01, 0.1]
WARMUP_PCTS = [5, 10, 20]
BASE_DIR = Path(__file__).resolve().parent
CLEANED_DIR = BASE_DIR / 'cleaned'

# Parameters to predict
PARAM_COLS = ['s_param', 'm_param', 't_param', 'g_param']

# ============================================================================
# Data Loading
# ============================================================================

def load_features(warmup_pct):
    """Load feature file for a given warmup percentage."""
    path = BASE_DIR / f'features_{warmup_pct}pct.csv'
    df = pd.read_csv(path)
    return df

def load_grid_data(ratio, data_type='pruned'):
    """Load grid_pruned or grid_full for a given ratio."""
    path = CLEANED_DIR / str(ratio) / f'grid_{data_type}.csv'
    df = pd.read_csv(path)
    return df

def load_baselines(ratio):
    """Load baseline algorithms for a given ratio."""
    path = CLEANED_DIR / str(ratio) / 'baselines.csv'
    df = pd.read_csv(path)
    return df

def get_fifo_miss_ratio(baselines_df, trace):
    """Get FIFO miss ratio for a trace."""
    fifo = baselines_df[(baselines_df['trace'] == trace) & 
                        (baselines_df['algorithm'] == 'FIFO')]
    if len(fifo) > 0:
        return fifo['miss_ratio'].values[0]
    return None

# ============================================================================
# Feature Engineering
# ============================================================================

def engineer_features(df):
    """Create additional features from the base features."""
    df = df.copy()
    
    # ==================== Basic Statistics ====================
    # Hit rates
    df['hit_rate'] = df['total_hits'] / df['total_reqs'].clip(lower=1)
    df['miss_rate'] = df['total_misses'] / df['total_reqs'].clip(lower=1)
    
    # Queue-specific hit proportions
    total_queue_hits = (df['hits_small'] + df['hits_main'] + df['hits_ghost']).clip(lower=1)
    df['small_hit_ratio'] = df['hits_small'] / total_queue_hits
    df['main_hit_ratio'] = df['hits_main'] / total_queue_hits
    df['ghost_hit_ratio'] = df['hits_ghost'] / total_queue_hits
    
    # ==================== Entropy-based Features ====================
    # Entropy ratios
    total_entropy = (df['H_s'] + df['H_m'] + df['H_g']).clip(lower=1e-6)
    df['H_s_ratio'] = df['H_s'] / total_entropy
    df['H_m_ratio'] = df['H_m'] / total_entropy
    df['H_g_ratio'] = df['H_g'] / total_entropy
    
    # Entropy products and differences
    df['H_sm_product'] = df['H_s'] * df['H_m']
    df['H_sg_product'] = df['H_s'] * df['H_g']
    df['H_mg_product'] = df['H_m'] * df['H_g']
    df['H_sm_diff'] = df['H_s'] - df['H_m']
    df['H_sg_diff'] = df['H_s'] - df['H_g']
    df['H_mg_diff'] = df['H_m'] - df['H_g']
    
    # Log entropy
    df['log_H_s'] = np.log1p(df['H_s'])
    df['log_H_m'] = np.log1p(df['H_m'])
    df['log_H_g'] = np.log1p(df['H_g'])
    
    # ==================== Workload Characteristics ====================
    # One-hit wonder ratio (important for cache sizing)
    df['onehit_ratio'] = df['rho_onehit']
    df['reuse_ratio'] = 1 - df['rho_onehit']
    
    # Uniqueness features
    df['unique_ratio'] = df['rho_unique']
    df['repeat_ratio'] = 1 - df['rho_unique']
    
    # Combined workload metrics
    df['workload_intensity'] = df['rho_unique'] * df['rho_onehit']
    df['cache_friendliness'] = (1 - df['rho_onehit']) * (1 - df['rho_unique'])
    
    # ==================== Histogram Statistics ====================
    # For each histogram, compute statistics
    for hist_name in ['hist_small', 'hist_main', 'hist_ghost']:
        cols = [f'{hist_name}_{i}' for i in range(20)]
        if all(c in df.columns for c in cols):
            hist_data = df[cols].values
            
            # Basic stats
            df[f'{hist_name}_mean'] = hist_data.mean(axis=1)
            df[f'{hist_name}_std'] = hist_data.std(axis=1)
            df[f'{hist_name}_max'] = hist_data.max(axis=1)
            df[f'{hist_name}_min'] = hist_data.min(axis=1)
            df[f'{hist_name}_range'] = df[f'{hist_name}_max'] - df[f'{hist_name}_min']
            
            # Skewness proxy (early vs late buckets)
            early_sum = hist_data[:, :5].sum(axis=1)
            late_sum = hist_data[:, -5:].sum(axis=1)
            df[f'{hist_name}_skew'] = (early_sum - late_sum) / (early_sum + late_sum + 1e-6)
            
            # Concentration in first bucket
            df[f'{hist_name}_first_bucket'] = hist_data[:, 0]
            
            # Tail weight
            df[f'{hist_name}_tail_weight'] = hist_data[:, -5:].sum(axis=1)
            
            # Entropy of histogram
            hist_normalized = hist_data / (hist_data.sum(axis=1, keepdims=True) + 1e-6)
            entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10), axis=1)
            df[f'{hist_name}_entropy'] = entropy
    
    # ==================== Cross-histogram Features ====================
    if all(f'hist_small_mean' in df.columns for _ in [1]):
        df['small_main_mean_ratio'] = df['hist_small_mean'] / (df['hist_main_mean'] + 1e-6)
        df['small_ghost_mean_ratio'] = df['hist_small_mean'] / (df['hist_ghost_mean'] + 1e-6)
        df['main_ghost_mean_ratio'] = df['hist_main_mean'] / (df['hist_ghost_mean'] + 1e-6)
    
    # ==================== Log Scale Features ====================
    df['log_total_reqs'] = np.log1p(df['total_reqs'])
    df['log_total_hits'] = np.log1p(df['total_hits'])
    df['log_total_misses'] = np.log1p(df['total_misses'])
    
    # ==================== Interaction Features ====================
    df['log_C_x_H_s'] = df['log_C'] * df['H_s']
    df['log_C_x_H_m'] = df['log_C'] * df['H_m']
    df['log_C_x_H_g'] = df['log_C'] * df['H_g']
    df['log_C_x_onehit'] = df['log_C'] * df['rho_onehit']
    df['log_C_x_unique'] = df['log_C'] * df['rho_unique']
    
    # Replace inf/nan with 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df

# ============================================================================
# Model Training & Evaluation
# ============================================================================

def prepare_training_data(features_df, grid_pruned_df, ratio):
    """Merge features with optimal parameters."""
    # Filter features for this ratio
    feat_ratio = features_df[features_df['ratio'] == ratio].copy()
    
    # Standardize trace names
    feat_ratio['trace_clean'] = feat_ratio['trace'].str.replace('.oracleGeneral', '', regex=False)
    grid_pruned_df['trace_clean'] = grid_pruned_df['trace'].str.replace('.oracleGeneral.zst', '', regex=False)
    
    # Merge
    merged = feat_ratio.merge(
        grid_pruned_df[['trace_clean'] + PARAM_COLS + ['miss_ratio']],
        on='trace_clean',
        how='inner',
        suffixes=('', '_optimal')
    )
    
    # Rename miss_ratio to optimal_miss_ratio
    merged = merged.rename(columns={'miss_ratio': 'optimal_miss_ratio'})
    
    return merged

def get_feature_columns(df):
    """Get list of feature columns (exclude metadata and targets)."""
    exclude = ['trace', 'trace_clean', 'ratio'] + PARAM_COLS + ['optimal_miss_ratio']
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

def train_xgb_model(X_train, y_train, param_name):
    """Train XGBoost model for a specific parameter."""
    # Use regression for all parameters - works better and avoids label issues
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_predictions(merged_df, grid_full_df, baselines_df, predictions):
    """Evaluate predictions using grid_full miss ratios."""
    results = []
    
    for idx, row in merged_df.iterrows():
        trace_clean = row['trace_clean']
        
        # Get predicted parameters
        pred_s = predictions['s_param'][idx]
        pred_m = predictions['m_param'][idx]
        pred_t = predictions['t_param'][idx]
        pred_g = predictions['g_param'][idx]
        
        # Round parameters to valid values
        pred_s = round_to_valid(pred_s, [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        pred_m = int(round(pred_m))
        pred_m = max(1, min(2, pred_m))
        pred_t = int(round(pred_t))
        pred_t = max(0, min(1, pred_t))
        pred_g = round_to_valid(pred_g, [0.9, 3.0, 6.0])
        
        # Look up miss ratio in grid_full
        grid_full_df['trace_clean'] = grid_full_df['trace'].str.replace('.oracleGeneral.zst', '', regex=False)
        match = grid_full_df[
            (grid_full_df['trace_clean'] == trace_clean) &
            (grid_full_df['s_param'] == pred_s) &
            (grid_full_df['m_param'] == pred_m) &
            (grid_full_df['t_param'] == pred_t) &
            (grid_full_df['g_param'] == pred_g)
        ]
        
        if len(match) > 0:
            pred_miss_ratio = match['miss_ratio'].values[0]
        else:
            pred_miss_ratio = None
        
        # Get optimal miss ratio
        optimal_miss_ratio = row['optimal_miss_ratio']
        
        # Get FIFO miss ratio
        trace_zst = trace_clean + '.oracleGeneral.zst'
        fifo_miss_ratio = get_fifo_miss_ratio(baselines_df, trace_zst)
        
        results.append({
            'trace': trace_clean,
            'pred_s': pred_s,
            'pred_m': pred_m,
            'pred_t': pred_t,
            'pred_g': pred_g,
            'pred_miss_ratio': pred_miss_ratio,
            'optimal_s': row['s_param'],
            'optimal_m': row['m_param'],
            'optimal_t': row['t_param'],
            'optimal_g': row['g_param'],
            'optimal_miss_ratio': optimal_miss_ratio,
            'fifo_miss_ratio': fifo_miss_ratio
        })
    
    return pd.DataFrame(results)

def round_to_valid(value, valid_values):
    """Round to nearest valid value."""
    return min(valid_values, key=lambda x: abs(x - value))

# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(warmup_pct, ratio):
    """Run the full pipeline for a given warmup percentage and ratio."""
    print(f"\n{'='*80}")
    print(f"Running pipeline: warmup={warmup_pct}%, ratio={ratio}")
    print(f"{'='*80}")
    
    # Load data
    features_df = load_features(warmup_pct)
    grid_pruned_df = load_grid_data(ratio, 'pruned')
    grid_full_df = load_grid_data(ratio, 'full')
    baselines_df = load_baselines(ratio)
    
    print(f"Features: {len(features_df)} rows")
    print(f"Grid Pruned: {len(grid_pruned_df)} rows")
    print(f"Grid Full: {len(grid_full_df)} rows")
    print(f"Baselines: {len(baselines_df)} rows")
    
    # Engineer features
    features_df = engineer_features(features_df)
    
    # Prepare training data
    merged_df = prepare_training_data(features_df, grid_pruned_df, ratio)
    print(f"Merged data: {len(merged_df)} rows")
    
    if len(merged_df) == 0:
        print("ERROR: No matching traces found!")
        return None
    
    # Get feature columns
    feature_cols = get_feature_columns(merged_df)
    print(f"Feature columns: {len(feature_cols)}")
    
    # Prepare X and y
    X = merged_df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    train_idx, test_idx = train_test_split(
        range(len(merged_df)), test_size=0.2, random_state=42
    )
    
    X_train = X_scaled[train_idx]
    X_test = X_scaled[test_idx]
    
    # Train models for each parameter
    models = {}
    predictions = {p: np.zeros(len(merged_df)) for p in PARAM_COLS}
    
    for param in PARAM_COLS:
        y = merged_df[param].values
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        print(f"\nTraining model for {param}...")
        model = train_xgb_model(X_train, y_train, param)
        models[param] = model
        
        # Predict on all data
        predictions[param] = model.predict(X_scaled)
        
        # Evaluate on test set
        y_pred_test = model.predict(X_test)
        
        if param in ['m_param', 't_param']:
            accuracy = (y_pred_test == y_test).mean()
            print(f"  Test Accuracy: {accuracy:.4f}")
        else:
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            print(f"  Test MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Feature importance (top 10)
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            top_idx = np.argsort(importance)[-10:][::-1]
            print(f"  Top features: {[feature_cols[i] for i in top_idx[:5]]}")
    
    # Evaluate using grid_full
    print("\nEvaluating predictions using grid_full miss ratios...")
    results_df = evaluate_predictions(merged_df, grid_full_df, baselines_df, predictions)
    
    # Filter out rows where prediction couldn't be found
    valid_results = results_df[results_df['pred_miss_ratio'].notna()].copy()
    print(f"Valid predictions: {len(valid_results)} / {len(results_df)}")
    
    if len(valid_results) > 0:
        # Calculate metrics
        valid_results['miss_ratio_diff'] = valid_results['pred_miss_ratio'] - valid_results['optimal_miss_ratio']
        valid_results['relative_gap'] = (valid_results['pred_miss_ratio'] - valid_results['optimal_miss_ratio']) / (valid_results['optimal_miss_ratio'] + 1e-6)
        
        # Overall statistics
        print(f"\nResults Summary:")
        print(f"  Mean predicted miss ratio: {valid_results['pred_miss_ratio'].mean():.6f}")
        print(f"  Mean optimal miss ratio: {valid_results['optimal_miss_ratio'].mean():.6f}")
        print(f"  Mean gap: {valid_results['miss_ratio_diff'].mean():.6f}")
        print(f"  Mean relative gap: {valid_results['relative_gap'].mean()*100:.2f}%")
        
        # Comparison with FIFO (tail cases)
        valid_with_fifo = valid_results[valid_results['fifo_miss_ratio'].notna()]
        if len(valid_with_fifo) > 0:
            pred_vs_fifo = (valid_with_fifo['pred_miss_ratio'] < valid_with_fifo['fifo_miss_ratio']).mean()
            opt_vs_fifo = (valid_with_fifo['optimal_miss_ratio'] < valid_with_fifo['fifo_miss_ratio']).mean()
            print(f"\n  % traces where predicted beats FIFO: {pred_vs_fifo*100:.2f}%")
            print(f"  % traces where optimal beats FIFO: {opt_vs_fifo*100:.2f}%")
            
            # Tail cases: where predicted is worse than FIFO
            tail_cases = valid_with_fifo[valid_with_fifo['pred_miss_ratio'] >= valid_with_fifo['fifo_miss_ratio']]
            print(f"  Tail cases (predicted >= FIFO): {len(tail_cases)} traces")
        
        # Parameter accuracy
        param_match = {
            's_param': (valid_results['pred_s'] == valid_results['optimal_s']).mean(),
            'm_param': (valid_results['pred_m'] == valid_results['optimal_m']).mean(),
            't_param': (valid_results['pred_t'] == valid_results['optimal_t']).mean(),
            'g_param': (valid_results['pred_g'] == valid_results['optimal_g']).mean()
        }
        print(f"\n  Parameter Match Accuracy:")
        for p, acc in param_match.items():
            print(f"    {p}: {acc*100:.2f}%")
        
        # Exact match (all params correct)
        all_match = (
            (valid_results['pred_s'] == valid_results['optimal_s']) &
            (valid_results['pred_m'] == valid_results['optimal_m']) &
            (valid_results['pred_t'] == valid_results['optimal_t']) &
            (valid_results['pred_g'] == valid_results['optimal_g'])
        ).mean()
        print(f"    All params match: {all_match*100:.2f}%")
    
    return {
        'warmup_pct': warmup_pct,
        'ratio': ratio,
        'results': valid_results,
        'models': models,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

def main():
    """Run pipeline for all combinations."""
    all_results = []
    
    for warmup_pct in WARMUP_PCTS:
        for ratio in CACHE_RATIOS:
            try:
                result = run_pipeline(warmup_pct, ratio)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"Error for warmup={warmup_pct}%, ratio={ratio}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary comparison across warmup percentages
    print("\n" + "="*80)
    print("SUMMARY: Comparison across warmup percentages and ratios")
    print("="*80)
    
    summary_data = []
    for result in all_results:
        if result and 'results' in result and len(result['results']) > 0:
            df = result['results']
            valid_with_fifo = df[df['fifo_miss_ratio'].notna()]
            
            summary_data.append({
                'warmup_pct': result['warmup_pct'],
                'ratio': result['ratio'],
                'n_traces': len(df),
                'mean_pred_mr': df['pred_miss_ratio'].mean(),
                'mean_opt_mr': df['optimal_miss_ratio'].mean(),
                'mean_gap': df['miss_ratio_diff'].mean(),
                'mean_rel_gap': df['relative_gap'].mean() * 100,
                'beats_fifo_pct': (valid_with_fifo['pred_miss_ratio'] < valid_with_fifo['fifo_miss_ratio']).mean() * 100 if len(valid_with_fifo) > 0 else 0
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(BASE_DIR / 'xgb_prediction_summary.csv', index=False)
        print(f"\nSummary saved to xgb_prediction_summary.csv")

if __name__ == '__main__':
    main()
