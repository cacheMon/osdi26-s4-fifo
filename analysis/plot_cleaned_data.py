#!/usr/bin/env python3
"""
Visualization script for cleaned data
Grid pruned results are shown as "Ground Truth" (optimal)
Baseline algorithms are compared against Ground Truth
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def normalize_trace(t):
    """Normalize trace names for consistent matching across data sources"""
    t = str(t)
    if '.oracleGeneral' in t:
        t = t.split('.oracleGeneral')[0]
    if t.endswith('.zst'): t = t[:-4]
    if t.endswith('.csv'): t = t[:-4]
    if t.endswith('.txt'): t = t[:-4]
    return t

def load_cleaned_data():
    """Load baselines and grid optimal from cleaned directories"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cleaned_dir = os.path.join(script_dir, 'cleaned')
    
    all_baselines = []
    all_ground_truth = []
    all_ground_truth_old = []
    
    # Load data for each ratio
    ratios = ['0.001', '0.01', '0.1']
    
    for ratio_str in ratios:
        ratio_dir = os.path.join(cleaned_dir, ratio_str)
        baseline_path = os.path.join(ratio_dir, 'baselines.csv')
        optimal_path = os.path.join(ratio_dir, 'grid_optimal.csv')
        optimal_old_path = os.path.join(ratio_dir, 'grid_optimal_old.csv')
        
        if os.path.exists(baseline_path):
            baseline_df = pd.read_csv(baseline_path)
            all_baselines.append(baseline_df)
            print(f"Loaded {len(baseline_df)} rows from baselines.csv (ratio={ratio_str})")
        else:
            print(f"Warning: {baseline_path} not found")
        
        if os.path.exists(optimal_path):
            optimal_df = pd.read_csv(optimal_path)
            # Add algorithm column to mark as Grid Optimal
            optimal_df['algorithm'] = 'Grid Optimal'
            all_ground_truth.append(optimal_df)
            print(f"Loaded {len(optimal_df)} rows from grid_optimal.csv (ratio={ratio_str})")
        else:
            print(f"Warning: {optimal_path} not found")
        
        if os.path.exists(optimal_old_path):
            optimal_old_df = pd.read_csv(optimal_old_path)
            # Add algorithm column to mark as Grid Optimal Old
            optimal_old_df['algorithm'] = 'Grid Optimal Old'
            all_ground_truth_old.append(optimal_old_df)
            print(f"Loaded {len(optimal_old_df)} rows from grid_optimal_old.csv (ratio={ratio_str})")
        else:
            print(f"Warning: {optimal_old_path} not found")
    
    # Combine all data
    baselines_df = pd.concat(all_baselines, ignore_index=True) if all_baselines else None
    ground_truth_df = pd.concat(all_ground_truth, ignore_index=True) if all_ground_truth else None
    ground_truth_old_df = pd.concat(all_ground_truth_old, ignore_index=True) if all_ground_truth_old else None
    
    # Normalize trace names
    if baselines_df is not None:
        baselines_df['trace'] = baselines_df['trace'].apply(normalize_trace)
    if ground_truth_df is not None:
        ground_truth_df['trace'] = ground_truth_df['trace'].apply(normalize_trace)
    if ground_truth_old_df is not None:
        ground_truth_old_df['trace'] = ground_truth_old_df['trace'].apply(normalize_trace)
    
    return baselines_df, ground_truth_df, ground_truth_old_df

def load_learned_data():
    """Load ML predictions if available"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # pred_path = os.path.join(script_dir, 'xgb_config_results', 'predictions.csv')
    pred_path = os.path.join(script_dir, 'xgb_18class_results', 'predictions.csv')
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        # Keep only necessary columns and rename
        pred_df = pred_df[['trace', 'ratio', 'predicted_miss_ratio']].copy()
        pred_df = pred_df.rename(columns={'predicted_miss_ratio': 'miss_ratio'})
        pred_df['algorithm'] = 'Learned'
        pred_df['trace'] = pred_df['trace'].apply(normalize_trace)
        print(f"Loaded {len(pred_df)} rows from predictions.csv (Learned)")
        return pred_df
    else:
        print("Warning: predictions.csv not found")
        return None

def load_tuned_data():
    """Load tuned predictions if available"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tuned_path = os.path.join(script_dir, 'xgb_18class_results', 'tune.csv')
    if os.path.exists(tuned_path):
        tuned_df = pd.read_csv(tuned_path)
        # Keep only necessary columns and rename
        tuned_df = tuned_df[['trace', 'ratio', 'tuned']].copy()
        tuned_df = tuned_df.rename(columns={'tuned': 'miss_ratio'})
        tuned_df['algorithm'] = 'Tuned'
        tuned_df['trace'] = tuned_df['trace'].apply(normalize_trace)
        print(f"Loaded {len(tuned_df)} rows from tune.csv (Tuned)")
        return tuned_df
    else:
        print("Warning: tune.csv not found")
        return None

def prepare_combined_data(baselines_df, ground_truth_df, ground_truth_old_df=None):
    """Combine baseline, ground truth, learned, and tuned data"""
    
    # Load learned data and tuned data
    learned_df = load_learned_data()
    tuned_df = load_tuned_data()
    
    # Ensure all have same columns
    ground_truth_clean = ground_truth_df[['trace', 'ratio', 'algorithm', 'miss_ratio']].copy()
    
    dfs_to_concat = [baselines_df, ground_truth_clean]
    
    # Add old optimal if available
    if ground_truth_old_df is not None:
        ground_truth_old_clean = ground_truth_old_df[['trace', 'ratio', 'algorithm', 'miss_ratio']].copy()
        dfs_to_concat.append(ground_truth_old_clean)
    
    # Filter for Test Set if Learned data is available
    if learned_df is not None:
        print("\nFiltering all data to match Test Set traces...")
        # Get set of (trace, ratio) in learned data
        test_keys = set(zip(learned_df['trace'], learned_df['ratio']))
        
        # Filter baselines
        initial_len = len(baselines_df)
        baselines_df = baselines_df[baselines_df.apply(lambda x: (x['trace'], x['ratio']) in test_keys, axis=1)]
        print(f"Baselines: {initial_len} -> {len(baselines_df)}")
        
        # Filter ground truth
        initial_len = len(ground_truth_clean)
        ground_truth_clean = ground_truth_clean[ground_truth_clean.apply(lambda x: (x['trace'], x['ratio']) in test_keys, axis=1)]
        print(f"Ground Truth: {initial_len} -> {len(ground_truth_clean)}")
        
        # Filter ground truth old
        if ground_truth_old_df is not None:
            initial_len = len(ground_truth_old_clean)
            ground_truth_old_clean = ground_truth_old_clean[ground_truth_old_clean.apply(lambda x: (x['trace'], x['ratio']) in test_keys, axis=1)]
            print(f"Ground Truth Old: {initial_len} -> {len(ground_truth_old_clean)}")
        
        # Filter tuned data if available
        if tuned_df is not None:
            initial_len = len(tuned_df)
            tuned_df = tuned_df[tuned_df.apply(lambda x: (x['trace'], x['ratio']) in test_keys, axis=1)]
            print(f"Tuned: {initial_len} -> {len(tuned_df)}")
        
        dfs_to_concat = [baselines_df, ground_truth_clean, learned_df]
        if ground_truth_old_df is not None:
            dfs_to_concat.append(ground_truth_old_clean)
        if tuned_df is not None:
            dfs_to_concat.append(tuned_df)
    
    # Combine
    combined_df = pd.concat(dfs_to_concat, ignore_index=True)
    
    print(f"\nCombined data: {len(combined_df)} total rows")
    print(f"Algorithms: {sorted(combined_df['algorithm'].unique())}")
    print(f"Ratios: {sorted(combined_df['ratio'].unique())}")
    
    # Find common traces for each ratio
    for ratio in sorted(combined_df['ratio'].unique()):
        ratio_data = combined_df[combined_df['ratio'] == ratio]
        print(f"\nRatio {ratio}:")
        for algo in sorted(ratio_data['algorithm'].unique()):
            algo_traces = ratio_data[ratio_data['algorithm'] == algo]['trace'].nunique()
            print(f"  {algo}: {algo_traces} traces")
    
    return combined_df

def calculate_percentiles(df):
    """Calculate percentiles for each algorithm and cache ratio"""
    percentiles_list = []
    
    for ratio in sorted(df['ratio'].unique()):
        for algo in sorted(df['algorithm'].unique()):
            algo_data = df[(df['ratio'] == ratio) & (df['algorithm'] == algo)]
            
            if len(algo_data) == 0:
                continue
            
            miss_ratios = algo_data['miss_ratio'].values
            
            percentiles_list.append({
                'ratio': ratio,
                'algorithm': algo,
                'min': np.min(miss_ratios),
                'p1': np.percentile(miss_ratios, 1),
                'p5': np.percentile(miss_ratios, 5),
                'p10': np.percentile(miss_ratios, 10),
                'p20': np.percentile(miss_ratios, 20),
                'p50': np.percentile(miss_ratios, 50),
                'mean': np.mean(miss_ratios),
                'p80': np.percentile(miss_ratios, 80),
                'p90': np.percentile(miss_ratios, 90),
                'p95': np.percentile(miss_ratios, 95),
                'p99': np.percentile(miss_ratios, 99),
                'max': np.max(miss_ratios),
                'count': len(miss_ratios)
            })
    
    return pd.DataFrame(percentiles_list)

def calculate_mean_miss_ratio_reduction(combined_df, baseline_algo='FIFO'):
    """Calculate mean miss ratio reduction for each algorithm compared to baseline"""
    algos = sorted(combined_df['algorithm'].unique())
    results = []
    
    # Get baseline data
    baseline_data = combined_df[combined_df['algorithm'] == baseline_algo].copy()
    baseline_grouped = baseline_data.groupby(['trace', 'ratio'])['miss_ratio'].mean()
    
    for algo in algos:
        if algo == baseline_algo:
            continue
        
        algo_data = combined_df[combined_df['algorithm'] == algo].copy()
        algo_grouped = algo_data.groupby(['trace', 'ratio'])['miss_ratio'].mean()
        
        # Calculate reduction for matching (trace, ratio) pairs
        reductions = []
        for (trace, ratio), baseline_mr in baseline_grouped.items():
            if (trace, ratio) in algo_grouped.index:
                algo_mr = algo_grouped[(trace, ratio)]
                reduction = (baseline_mr - algo_mr) / baseline_mr
                reductions.append(reduction)
        
        if len(reductions) > 0:
            results.append({
                'Algorithm': algo,
                'Mean_Miss_Ratio_Reduction': np.mean(reductions),
                'Std_Miss_Ratio_Reduction': np.std(reductions),
                'Count': len(reductions)
            })
    
    return pd.DataFrame(results)

def plot_mean_miss_ratio_reduction_bar(combined_df, baseline_algo='FIFO'):
    # Custom order: Grid Optimal first, then Grid Optimal Old, then Learned, then Tuned, then others
    algo_order = ['Grid Optimal', 'Grid Optimal Old', 'Learned', 'Tuned', 'S3FIFO', 'LeCaR', 'ThreeLCache', 'LRB', 'ARC', 'LRU', 'LIRS', 'TwoQ', 'LHD']
    reduction_df = calculate_mean_miss_ratio_reduction(combined_df, baseline_algo)
    
    # Filter out algorithms we don't want to show
    exclude_algos = ['FIFO', 'LFU', 'WTinyLFU', 'Sieve']
    reduction_df = reduction_df[~reduction_df['Algorithm'].isin(exclude_algos)].copy()
    
    # Custom order: Grid Optimal first, then Learned, then Tuned, then others
    algo_order = ['Grid Optimal', 'Learned', 'Tuned', 'S3FIFO', 'LeCaR', 'ThreeLCache', 'LRB', 'ARC', 'LRU', 'LIRS', 'TwoQ', 'LHD']
    
    # Filter to only include algorithms that exist in data
    available_algos = set(reduction_df['Algorithm'].unique())
    algo_order = [a for a in algo_order if a in available_algos]
    
    # Reorder the dataframe according to custom order
    reduction_df['Algorithm'] = pd.Categorical(reduction_df['Algorithm'], categories=algo_order, ordered=True)
    reduction_df = reduction_df.sort_values('Algorithm')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    algos = reduction_df['Algorithm'].values
    means = reduction_df['Mean_Miss_Ratio_Reduction'].values * 100  # Convert to percentage
    stds = reduction_df['Std_Miss_Ratio_Reduction'].values * 100
    
    # Create color map - Grid Optimal uses gold, Grid Optimal Old uses orange, Tuned uses green, others use blue
    colors = []
    for algo in algos:
        if algo == 'Grid Optimal':
            colors.append('#FFD700')  # Gold color
        elif algo == 'Grid Optimal Old':
            colors.append('#FFA500')  # Orange color
        elif algo == 'Tuned':
            colors.append('#2ecc71')  # Green color for Tuned
        else:
            colors.append('#1f77b4')  # Blue color
    
    bars = ax.bar(range(len(algos)), means, yerr=stds, capsize=5, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Algorithm', fontsize=20, fontweight='bold')
    ax.set_ylabel('Mean Miss Ratio Reduction (%)', fontsize=20, fontweight='bold')
    ax.set_title(f'Mean Miss Ratio Reduction vs {baseline_algo}', fontsize=22, fontweight='bold')
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, rotation=45, ha='right', fontsize=18, fontweight='bold')
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}%',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleaned')
    output_path = os.path.join(output_dir, 'figure_mean_miss_ratio_reduction.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()
    
    # Print the table
    print("\n" + "="*80)
    print(f"Mean Miss Ratio Reduction (compared to {baseline_algo})")
    print("="*80)
    print(reduction_df.to_string(index=False))
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'mean_miss_ratio_reduction.csv')
    reduction_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    
    return reduction_df

def plot_worst_and_p1_relative_to_fifo(df, percentiles_df):
    """Plot worst case, P99 worst, mean, and P1 best performance relative to FIFO"""
    cache_ratios = sorted(df['ratio'].unique())
    
    # Custom algorithm order (excluding FIFO)
    algorithm_order = ['Grid Optimal', 'Grid Optimal Old', 'Learned', 'Tuned', 'S3FIFO', 'LeCaR', 'ThreeLCache', 'LRB', 'ARC', 'LRU', 'LIRS', 'TwoQ', 'LHD', 'Sieve', 'WTinyLFU', 'LFU']
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleaned')
    
    # For each ratio, calculate relative performance
    for ratio in cache_ratios:
        ratio_data = df[df['ratio'] == ratio]
        
        # Get FIFO data as baseline
        fifo_data = ratio_data[ratio_data['algorithm'] == 'FIFO'][['trace', 'miss_ratio']].copy()
        fifo_data = fifo_data.rename(columns={'miss_ratio': 'fifo_mr'})
        
        worst_relative = []
        p99_worst_relative = []
        mean_relative = []
        p1_best_relative = []
        algo_labels = []
        
        for algo in algorithm_order:
            if algo not in ratio_data['algorithm'].unique():
                continue
                
            algo_data = ratio_data[ratio_data['algorithm'] == algo][['trace', 'miss_ratio']].copy()
            
            # Merge with FIFO on trace
            merged = pd.merge(algo_data, fifo_data, on='trace', how='inner')
            
            if len(merged) == 0:
                continue
            
            # Calculate relative performance: (algo_mr - fifo_mr) / fifo_mr * 100
            # Negative means better than FIFO, positive means worse
            merged['relative_mr'] = (merged['miss_ratio'] - merged['fifo_mr']) / merged['fifo_mr'] * 100
            
            # Get worst case (maximum relative miss ratio = most degraded)
            worst_case = merged['relative_mr'].max()
            
            # Get P99 worst (99th percentile = worst 1%)
            p99_worst = np.percentile(merged['relative_mr'].values, 99)
            
            # Get mean
            mean_case = merged['relative_mr'].mean()
            
            # Get P1 best (1st percentile = best 1%)
            p1_best = np.percentile(merged['relative_mr'].values, 1)
            
            worst_relative.append(worst_case)
            p99_worst_relative.append(p99_worst)
            mean_relative.append(mean_case)
            p1_best_relative.append(p1_best)
            algo_labels.append(algo)
        
        # Create plot with four subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 14))
        
        # Colors
        colors = []
        for a in algo_labels:
            if a == 'Grid Optimal':
                colors.append('#FFD700')  # Gold
            elif a == 'Grid Optimal Old':
                colors.append('#FFA500')  # Orange
            elif a == 'Tuned':
                colors.append('#2ecc71')  # Green for Tuned
            else:
                colors.append('#1f77b4')  # Blue
        
        # Plot 1: Worst Case
        bars1 = ax1.bar(range(len(algo_labels)), worst_relative, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Worst Case Miss Ratio vs FIFO (%)', fontsize=14, fontweight='bold')
        ax1.set_title(f'Worst Case (Ratio: {ratio})', fontsize=16, fontweight='bold')
        ax1.set_xticks(range(len(algo_labels)))
        ax1.set_xticklabels(algo_labels, rotation=45, ha='right', fontsize=12, fontweight='bold')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='FIFO baseline')
        ax1.set_yscale('symlog', linthresh=1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(fontsize=10)
        
        for i, (bar, val) in enumerate(zip(bars1, worst_relative)):
            height = bar.get_height()
            label_text = f'{val:.1f}%' if abs(val) < 100 else (f'{val:.0f}%' if abs(val) < 10000 else f'{val:.0e}')
            ax1.text(bar.get_x() + bar.get_width()/2., height, label_text,
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8, fontweight='bold', rotation=90 if abs(val) > 1000 else 0)
        
        # Plot 2: P99 worst (worst 1%)
        bars2 = ax2.bar(range(len(algo_labels)), p99_worst_relative, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
        ax2.set_ylabel('P99 Worst Miss Ratio vs FIFO (%)', fontsize=14, fontweight='bold')
        ax2.set_title(f'P99 Worst (Ratio: {ratio})', fontsize=16, fontweight='bold')
        ax2.set_xticks(range(len(algo_labels)))
        ax2.set_xticklabels(algo_labels, rotation=45, ha='right', fontsize=12, fontweight='bold')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='FIFO baseline')
        ax2.set_yscale('symlog', linthresh=1.0)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(fontsize=10)
        
        for i, (bar, val) in enumerate(zip(bars2, p99_worst_relative)):
            height = bar.get_height()
            label_text = f'{val:.1f}%' if abs(val) < 100 else (f'{val:.0f}%' if abs(val) < 10000 else f'{val:.0e}')
            ax2.text(bar.get_x() + bar.get_width()/2., height, label_text,
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8, fontweight='bold', rotation=90 if abs(val) > 1000 else 0)
        
        # Plot 3: Mean
        bars3 = ax3.bar(range(len(algo_labels)), mean_relative, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Mean Miss Ratio vs FIFO (%)', fontsize=14, fontweight='bold')
        ax3.set_title(f'Mean Performance (Ratio: {ratio})', fontsize=16, fontweight='bold')
        ax3.set_xticks(range(len(algo_labels)))
        ax3.set_xticklabels(algo_labels, rotation=45, ha='right', fontsize=12, fontweight='bold')
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='FIFO baseline')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend(fontsize=10)
        
        for i, (bar, val) in enumerate(zip(bars3, mean_relative)):
            height = bar.get_height()
            label_text = f'{val:.1f}%'
            ax3.text(bar.get_x() + bar.get_width()/2., height, label_text,
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8, fontweight='bold')
        
        # Plot 4: P1 Best (best 1%)
        bars4 = ax4.bar(range(len(algo_labels)), p1_best_relative, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
        ax4.set_ylabel('P1 Best Miss Ratio vs FIFO (%)', fontsize=14, fontweight='bold')
        ax4.set_title(f'P1 Best (Ratio: {ratio})', fontsize=16, fontweight='bold')
        ax4.set_xticks(range(len(algo_labels)))
        ax4.set_xticklabels(algo_labels, rotation=45, ha='right', fontsize=12, fontweight='bold')
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, label='FIFO baseline')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend(fontsize=10)
        
        for i, (bar, val) in enumerate(zip(bars4, p1_best_relative)):
            height = bar.get_height()
            label_text = f'{val:.1f}%'
            ax4.text(bar.get_x() + bar.get_width()/2., height, label_text,
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'figure_ratio_{ratio}_tail_performance_vs_fifo.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
        
        # Save data to CSV
        stats_df = pd.DataFrame({
            'algorithm': algo_labels,
            'worst_vs_fifo_pct': worst_relative,
            'p99_worst_vs_fifo_pct': p99_worst_relative,
            'mean_vs_fifo_pct': mean_relative,
            'p1_best_vs_fifo_pct': p1_best_relative
        })
        csv_path = os.path.join(output_dir, f'tail_stats_ratio_{ratio}_vs_fifo.csv')
        stats_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

def plot_by_cache_ratio(df, percentiles_df):
    """Create one plot per cache ratio with algorithms on x-axis"""
    # Just call the new function
    plot_worst_and_p1_relative_to_fifo(df, percentiles_df)

def create_statistics_table(percentiles_df):
    """Create a statistics summary table"""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleaned')
    
    for ratio in sorted(percentiles_df['ratio'].unique()):
        ratio_stats = percentiles_df[percentiles_df['ratio'] == ratio].copy()
        ratio_stats = ratio_stats.sort_values('mean')
        
        print(f"\n{'='*100}")
        print(f"Statistics for Cache Ratio: {ratio}")
        print('='*100)
        print(ratio_stats[['algorithm', 'min', 'p10', 'p50', 'mean', 'p90', 'p99', 'max', 'count']].to_string(index=False))
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'statistics.csv')
    percentiles_df.to_csv(csv_path, index=False)
    print(f"\n\nSaved statistics to: {csv_path}")

def analyze_grid_optimal_performance(combined_df):
    """Analyze how Grid Optimal compares to other algorithms"""
    print("\n" + "="*110)
    print("Grid Optimal Performance Analysis")
    print("="*110)
    
    cache_ratios = sorted(combined_df['ratio'].unique())
    
    for ratio in cache_ratios:
        ratio_data = combined_df[combined_df['ratio'] == ratio]
        
        # Get Grid Optimal data for this ratio
        opt_data = ratio_data[ratio_data['algorithm'] == 'Grid Optimal'][['trace', 'miss_ratio']].copy()
        opt_data = opt_data.rename(columns={'miss_ratio': 'opt_mr'})
        
        print(f"\n--- Cache Ratio: {ratio} ---")
        print(f"Grid Optimal traces: {len(opt_data)}")
        
        # Compare with each algorithm
        algos = [a for a in sorted(ratio_data['algorithm'].unique()) if a not in ['Grid Optimal', 'Grid Optimal Old']]
        
        for algo in algos:
            algo_data = ratio_data[ratio_data['algorithm'] == algo][['trace', 'miss_ratio']].copy()
            algo_data = algo_data.rename(columns={'miss_ratio': 'algo_mr'})
            
            # Merge on trace
            merged = pd.merge(opt_data, algo_data, on='trace', how='inner')
            
            if len(merged) == 0:
                continue
            
            # Calculate improvement
            merged['improvement'] = (merged['algo_mr'] - merged['opt_mr']) / merged['algo_mr'] * 100
            
            mean_improvement = merged['improvement'].mean()
            cases_opt_better = (merged['opt_mr'] < merged['algo_mr']).sum()
            total_cases = len(merged)
            
def main():
    print("Loading cleaned data...")
    baselines_df, ground_truth_df, ground_truth_old_df = load_cleaned_data()
    
    if baselines_df is None or ground_truth_df is None:
        print("Error: Failed to load data")
        return
    
    print("\nPreparing combined data...")
    combined_df = prepare_combined_data(baselines_df, ground_truth_df, ground_truth_old_df)
    
    if baselines_df is None or ground_truth_df is None:
        print("Error: Failed to load data")
        return
    
    print("\nPreparing combined data...")
    combined_df = prepare_combined_data(baselines_df, ground_truth_df)
    
    print("\nCalculating percentiles...")
    percentiles_df = calculate_percentiles(combined_df)
    print(f"Calculated percentiles for {len(percentiles_df)} algorithm-ratio combinations")
    
    print("\nCreating statistics table...")
    create_statistics_table(percentiles_df)
    
    print("\nCreating mean miss ratio reduction plot...")
    plot_mean_miss_ratio_reduction_bar(combined_df, baseline_algo='FIFO')
    
    print("\nCreating worst case and P99 plots (relative to FIFO)...")
    plot_by_cache_ratio(combined_df, percentiles_df)
    
    print("\nAnalyzing Grid Optimal performance...")
    analyze_grid_optimal_performance(combined_df)
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)

if __name__ == '__main__':
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    main()
