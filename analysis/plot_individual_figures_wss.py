#!/usr/bin/env python3
"""
Generate individual plots for WSS model comparison.
Uses predictions from xgb_18class_results_wss.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
MODEL_DIR = 'xgb_18class_results_wss'
PRED_FILE = f'{MODEL_DIR}/predictions.csv'
OUT_DIR = 'cleaned/individual_plots_wss'
os.makedirs(OUT_DIR, exist_ok=True)

RATIOS = [0.001, 0.01, 0.1]
ALGO_ORDER = ['Grid Optimal', 'Learned', 'Tuned', 'ThreeLCache', 'LRB', 'LHD', 
              'ARC', 'LeCaR', 'S3FIFO', 'TwoQ', 'FIFO', 'LIRS']

plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 16,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
})


def normalize_trace(t):
    t = str(t)
    if '.oracleGeneral' in t:
        t = t.split('.oracleGeneral')[0]
    for ext in ['.zst', '.csv', '.txt']:
        if t.endswith(ext):
            t = t[:-len(ext)]
    return t


def get_colors(algos):
    colors = []
    for algo in algos:
        if algo == 'Grid Optimal':
            colors.append('#FFD700')
        elif algo == 'Tuned':
            colors.append('#2ecc71')
        else:
            colors.append('#1f77b4')
    return colors


DISPLAY_NAMES = {'ThreeLCache': '3LCache', 'TwoQ': '2Q'}


def get_display_names(algos):
    return [DISPLAY_NAMES.get(a, a) for a in algos]


def load_data():
    print("Loading data...")
    
    # Load baselines
    baselines = []
    for ratio in ['0.001', '0.01', '0.1']:
        path = f'cleaned/{ratio}/baselines.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            baselines.append(df)
    baselines_df = pd.concat(baselines, ignore_index=True)
    baselines_df['trace'] = baselines_df['trace'].apply(normalize_trace)
    
    # Load grid optimal
    grid_optimal = []
    for ratio in ['0.001', '0.01', '0.1']:
        path = f'cleaned/{ratio}/grid_optimal.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['algorithm'] = 'Grid Optimal'
            grid_optimal.append(df)
    grid_optimal_df = pd.concat(grid_optimal, ignore_index=True)
    grid_optimal_df['trace'] = grid_optimal_df['trace'].apply(normalize_trace)
    
    # Load WSS Learned predictions
    pred_df = pd.read_csv(PRED_FILE)
    # Keep only needed columns and rename
    if 'predicted_miss_ratio' in pred_df.columns:
        pred_df = pred_df[['trace', 'ratio', 'predicted_miss_ratio']].copy()
        pred_df = pred_df.rename(columns={'predicted_miss_ratio': 'miss_ratio'})
    pred_df['algorithm'] = 'Learned'
    pred_df['trace'] = pred_df['trace'].apply(normalize_trace)
    # Remove duplicates
    pred_df = pred_df.drop_duplicates(subset=['trace', 'ratio'])
    
    # Load Tuned
    tuned_df = pd.read_csv('xgb_18class_results/tune.csv')
    tuned_df = tuned_df[['trace', 'ratio', 'tuned']].copy()
    tuned_df = tuned_df.rename(columns={'tuned': 'miss_ratio'})
    tuned_df['algorithm'] = 'Tuned'
    tuned_df['trace'] = tuned_df['trace'].apply(normalize_trace)
    
    # Filter to common traces
    tuned_keys = set(zip(tuned_df['trace'], tuned_df['ratio']))
    print(f"Common trace set size: {len(tuned_keys)}")
    
    def filter_to_common(df):
        return df[df.apply(lambda x: (x['trace'], x['ratio']) in tuned_keys, axis=1)]
    
    baselines_filtered = filter_to_common(baselines_df)
    grid_optimal_filtered = filter_to_common(grid_optimal_df)
    pred_filtered = filter_to_common(pred_df)
    
    combined = pd.concat([
        baselines_filtered,
        grid_optimal_filtered[['trace', 'ratio', 'algorithm', 'miss_ratio']],
        pred_filtered,
        tuned_df
    ], ignore_index=True)
    
    fifo_df = baselines_filtered[baselines_filtered['algorithm'] == 'FIFO'][['trace', 'ratio', 'miss_ratio']].copy()
    fifo_df = fifo_df.rename(columns={'miss_ratio': 'fifo_mr'})
    
    print(f"Total combined records: {len(combined)}")
    return combined, fifo_df


def plot_percentile_vs_fifo(combined, fifo_df, percentile, name):
    print(f"\nGenerating: {name} vs FIFO plots...")
    
    for ratio in RATIOS:
        results = []
        fifo_ratio = fifo_df[fifo_df['ratio'] == ratio][['trace', 'fifo_mr']]
        
        for algo in ALGO_ORDER:
            algo_data = combined[(combined['algorithm'] == algo) & (combined['ratio'] == ratio)][['trace', 'miss_ratio']].copy()
            algo_data = algo_data.rename(columns={'miss_ratio': 'algo_mr'})
            merged = pd.merge(algo_data, fifo_ratio, on='trace', how='inner')
            if len(merged) > 0:
                merged['rel'] = (merged['fifo_mr'] - merged['algo_mr']) / merged['fifo_mr'] * 100
                val = merged['rel'].quantile(percentile)
                results.append({'Algorithm': algo, 'Value': val})
        
        results_df = pd.DataFrame(results)
        fig, ax = plt.subplots(figsize=(14, 9))
        bars = ax.bar(range(len(results_df)), results_df['Value'].values,
                     color=get_colors(results_df['Algorithm'].values), alpha=0.8, edgecolor='black')
        ax.set_ylabel('Miss Ratio Reduction From FIFO (%)', fontsize=22, fontweight='bold')
        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels(get_display_names(results_df['Algorithm'].values), rotation=45, ha='right', fontsize=18, fontweight='bold')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, results_df['Value'].values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/{name}_vs_fifo_ratio_{ratio}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{OUT_DIR}/{name}_vs_fifo_ratio_{ratio}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {name}_vs_fifo_ratio_{ratio}.pdf")


def plot_worse_than_fifo_pct(combined, fifo_df):
    print("\nGenerating: Worse than FIFO percentage plots...")
    
    for ratio in RATIOS:
        results = []
        fifo_ratio = fifo_df[fifo_df['ratio'] == ratio]
        
        for algo in ALGO_ORDER:
            algo_data = combined[(combined['algorithm'] == algo) & (combined['ratio'] == ratio)][['trace', 'miss_ratio']].copy()
            merged = pd.merge(algo_data, fifo_ratio, on='trace', how='inner')
            if len(merged) > 0:
                worse_count = (merged['miss_ratio'] > merged['fifo_mr']).sum()
                pct = worse_count / len(merged) * 100
                results.append({'Algorithm': algo, 'Pct': pct})
        
        results_df = pd.DataFrame(results)
        fig, ax = plt.subplots(figsize=(14, 9))
        bars = ax.bar(range(len(results_df)), results_df['Pct'].values, 
                     color=get_colors(results_df['Algorithm'].values), alpha=0.8, edgecolor='black')
        ax.set_ylabel('Trace Fraction (%)', fontsize=22, fontweight='bold')
        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels(get_display_names(results_df['Algorithm'].values), rotation=45, ha='right', fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, pct in zip(bars, results_df['Pct'].values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/worse_than_fifo_pct_ratio_{ratio}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{OUT_DIR}/worse_than_fifo_pct_ratio_{ratio}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: worse_than_fifo_pct_ratio_{ratio}.pdf")


def plot_mean_miss_ratio_reduction(combined, fifo_df):
    """Plot mean miss ratio reduction vs FIFO - aligned to Tuned traces."""
    print("\nGenerating: Mean miss ratio reduction plots...")
    
    for ratio in RATIOS:
        results = []
        fifo_ratio = fifo_df[fifo_df['ratio'] == ratio][['trace', 'fifo_mr']]
        
        for algo in ALGO_ORDER:
            algo_data = combined[(combined['algorithm'] == algo) & (combined['ratio'] == ratio)][['trace', 'miss_ratio']].copy()
            algo_data = algo_data.rename(columns={'miss_ratio': 'algo_mr'})
            merged = pd.merge(algo_data, fifo_ratio, on='trace', how='inner')
            if len(merged) > 0:
                merged['reduction'] = (merged['fifo_mr'] - merged['algo_mr']) / merged['fifo_mr'] * 100
                mean_val = merged['reduction'].mean()
                results.append({'Algorithm': algo, 'Mean': mean_val})
        
        results_df = pd.DataFrame(results)
        fig, ax = plt.subplots(figsize=(14, 9))
        bars = ax.bar(range(len(results_df)), results_df['Mean'].values,
                     color=get_colors(results_df['Algorithm'].values), alpha=0.8, edgecolor='black')
        ax.set_ylabel('Mean Miss Ratio Reduction (%)', fontsize=22, fontweight='bold')
        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels(get_display_names(results_df['Algorithm'].values), rotation=45, ha='right', fontsize=18, fontweight='bold')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, results_df['Mean'].values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/mean_vs_fifo_ratio_{ratio}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{OUT_DIR}/mean_vs_fifo_ratio_{ratio}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: mean_vs_fifo_ratio_{ratio}.pdf")


def main():
    combined, fifo_df = load_data()
    
    # Generate all plots (trace set aligned to Tuned)
    plot_worse_than_fifo_pct(combined, fifo_df)
    plot_percentile_vs_fifo(combined, fifo_df, 0.00, 'worst_case')
    plot_percentile_vs_fifo(combined, fifo_df, 0.01, 'p1')
    plot_percentile_vs_fifo(combined, fifo_df, 0.10, 'p10')
    plot_percentile_vs_fifo(combined, fifo_df, 0.50, 'median')
    plot_percentile_vs_fifo(combined, fifo_df, 0.90, 'p90')
    plot_percentile_vs_fifo(combined, fifo_df, 0.99, 'p99')
    plot_percentile_vs_fifo(combined, fifo_df, 1.00, 'best_case')
    plot_mean_miss_ratio_reduction(combined, fifo_df)
    
    print(f"\n{'='*60}")
    print(f"All WSS model plots saved to: {OUT_DIR}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
