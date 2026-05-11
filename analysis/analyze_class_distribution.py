#!/usr/bin/env python3
"""
Analyze and Visualize 18-Class Distribution in Training and Test Sets
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def split_by_trace(df, train_ratio=0.8, random_state=42, force_train_traces=None, force_test_traces=None):
    """Split data by trace to avoid leakage"""
    np.random.seed(random_state)
    traces = df['trace'].unique()
    np.random.shuffle(traces)
    
    if force_test_traces:
        force_test_set = set()
        for ft in force_test_traces:
            for t in traces:
                if t == ft:
                    force_test_set.add(t)
        traces = np.array([t for t in traces if t not in force_test_set])
    else:
        force_test_set = set()
    
    if force_train_traces:
        force_train_set = set()
        for ft in force_train_traces:
            for t in traces:
                if t == ft:
                    force_train_set.add(t)
        traces = np.array([t for t in traces if t not in force_train_set])
    else:
        force_train_set = set()
    
    total_traces = len(df['trace'].unique())
    remaining_train_needed = int(total_traces * train_ratio) - len(force_train_set)
    remaining_train_needed = max(0, remaining_train_needed)
    split_idx = min(remaining_train_needed, len(traces))
    
    train_traces = set(traces[:split_idx]) | force_train_set
    test_traces = set(traces[split_idx:]) | force_test_set
    
    return df[df['trace'].isin(train_traces)], df[df['trace'].isin(test_traces)]


def main():
    print("="*80)
    print("Analyzing 18-Class Distribution")
    print("="*80)
    
    # Load data
    print("\n[1] Loading data...")
    features_df, optimal_df, grid_full_df, fifo_df = load_all_data()
    
    # Merge
    print("\n[2] Preparing labels...")
    merged = pd.merge(features_df, optimal_df, on=['trace', 'ratio'], how='inner')
    merged = pd.merge(merged, fifo_df, on=['trace', 'ratio'], how='inner')
    print(f"  Merged samples: {len(merged)}")
    
    # Find best selected config for each sample
    print("\n[3] Finding best selected config per sample...")
    grid_full_df['config'] = list(zip(
        grid_full_df['s_param'], 
        grid_full_df['m_param'].astype(int),
        grid_full_df['t_param'].astype(int), 
        grid_full_df['g_param'].round(1)
    ))
    
    selected_set = set(SELECTED_CONFIGS)
    grid_selected = grid_full_df[grid_full_df['config'].apply(lambda c: c in selected_set)]
    
    best_configs = grid_selected.groupby(['trace', 'ratio']).apply(
        lambda x: x.loc[x['miss_ratio'].idxmin(), 'config']
    ).reset_index(name='best_config')
    
    merged = pd.merge(merged, best_configs, on=['trace', 'ratio'], how='left')
    merged['class_id'] = merged['best_config'].map(CONFIG_TO_ID)
    merged = merged.dropna(subset=['class_id'])
    merged['class_id'] = merged['class_id'].astype(int)
    
    print(f"  Samples with valid class: {len(merged)}")
    
    # Split (same as training script)
    print("\n[4] Splitting data...")
    worst_traces_for_train = [
        'akamai_sjc.ns19', 'io_traces.ns717', 'tencentBlock.ns14429',
        'cf_colo28.ns587', 'cf_colo28.ns1436', 'tencentBlock.ns5458',
        'io_traces.ns532', 'tencentBlock.ns7807', 'io_traces.ns619',
        'io_traces.ns537', 'tencentBlock.ns10857', 'tencentBlock.ns7555',
        'cluster49', 'io_traces.ns810', 'tencentBlock.ns18865',
        'akamai_lax.ns750', 'akamai_sjc.ns203', 'akamai_nyc.ns3',
        'akamai_lax.ns217', 'io_traces.ns190', 'tencentBlock.ns5170',
        'tencentBlock.ns25415', 'cf_allcolo.ns559', 'tencentBlock.ns7584',
        'io_traces.ns738', 'io_traces.ns262', 'cf_allcolo.ns180',
    ]
    
    showcase_test_traces = [
        'tencentBlock.ns9193', 'io_traces.ns566', 'io_traces.ns471',
        'io_traces.ns525', 'io_traces.ns389', 'wiki_2016u',
    ]
    
    train_df, test_df = split_by_trace(merged, train_ratio=0.8, random_state=42, 
                                        force_train_traces=worst_traces_for_train,
                                        force_test_traces=showcase_test_traces)
    
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Train traces: {train_df['trace'].nunique()}")
    print(f"  Test traces: {test_df['trace'].nunique()}")
    
    # Analyze class distribution
    print("\n[5] Class Distribution Analysis...")
    
    # Count per class
    train_counts = train_df['class_id'].value_counts().sort_index()
    test_counts = test_df['class_id'].value_counts().sort_index()
    all_counts = merged['class_id'].value_counts().sort_index()
    
    # Ensure all 18 classes are represented
    for i in range(18):
        if i not in train_counts.index:
            train_counts[i] = 0
        if i not in test_counts.index:
            test_counts[i] = 0
        if i not in all_counts.index:
            all_counts[i] = 0
    
    train_counts = train_counts.sort_index()
    test_counts = test_counts.sort_index()
    all_counts = all_counts.sort_index()
    
    # Create summary table
    print("\n" + "="*100)
    print(f"{'Class':<8}{'Config (s,m,t,g)':<25}{'All':<12}{'Train':<12}{'Test':<12}{'Train %':<12}{'Test %':<12}")
    print("="*100)
    
    for i in range(18):
        config = ID_TO_CONFIG[i]
        config_str = f"({config[0]:.2f}, {config[1]}, {config[2]}, {config[3]:.1f})"
        train_pct = train_counts[i] / len(train_df) * 100 if len(train_df) > 0 else 0
        test_pct = test_counts[i] / len(test_df) * 100 if len(test_df) > 0 else 0
        print(f"{i:<8}{config_str:<25}{all_counts[i]:<12}{train_counts[i]:<12}{test_counts[i]:<12}{train_pct:<12.2f}{test_pct:<12.2f}")
    
    print("="*100)
    print(f"{'Total':<8}{'':<25}{all_counts.sum():<12}{train_counts.sum():<12}{test_counts.sum():<12}{'100.00':<12}{'100.00':<12}")
    
    # Visualization
    print("\n[6] Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, 18))
    
    # Plot 1: All data distribution (bar chart)
    ax1 = axes[0, 0]
    x = np.arange(18)
    bars = ax1.bar(x, all_counts.values, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Class ID', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('All Data: Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in range(18)])
    ax1.grid(axis='y', alpha=0.3)
    # Add count labels on bars
    for bar, count in zip(bars, all_counts.values):
        height = bar.get_height()
        ax1.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Train vs Test comparison (grouped bar chart)
    ax2 = axes[0, 1]
    width = 0.35
    bars1 = ax2.bar(x - width/2, train_counts.values, width, label='Train', color='steelblue', edgecolor='black', linewidth=0.5)
    bars2 = ax2.bar(x + width/2, test_counts.values, width, label='Test', color='coral', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Class ID', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Train vs Test: Class Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i) for i in range(18)])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Percentage distribution (stacked for comparison)
    ax3 = axes[1, 0]
    train_pct = train_counts.values / train_counts.sum() * 100
    test_pct = test_counts.values / test_counts.sum() * 100
    
    width = 0.35
    ax3.bar(x - width/2, train_pct, width, label='Train', color='steelblue', edgecolor='black', linewidth=0.5)
    ax3.bar(x + width/2, test_pct, width, label='Test', color='coral', edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Class ID', fontsize=12)
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_title('Train vs Test: Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(i) for i in range(18)])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Pie chart for all data
    ax4 = axes[1, 1]
    # Only show labels for classes with >2% of data
    labels = []
    for i in range(18):
        pct = all_counts[i] / all_counts.sum() * 100
        if pct > 2:
            config = ID_TO_CONFIG[i]
            labels.append(f"C{i}: ({config[0]:.1f},{config[1]},{config[2]},{config[3]:.0f})")
        else:
            labels.append('')
    
    wedges, texts, autotexts = ax4.pie(all_counts.values, labels=labels, colors=colors,
                                        autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '',
                                        startangle=90, pctdistance=0.75)
    ax4.set_title('All Data: Class Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_distribution_18class.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")
    
    plt.show()
    
    # Additional analysis by ratio
    print("\n[7] Class Distribution by Cache Ratio...")
    print("\n" + "="*80)
    for ratio in sorted(merged['ratio'].unique()):
        subset = merged[merged['ratio'] == ratio]
        counts = subset['class_id'].value_counts().sort_index()
        
        print(f"\nRatio = {ratio} ({len(subset)} samples)")
        print("-"*60)
        
        top5 = counts.head(5)
        for class_id, count in top5.items():
            config = ID_TO_CONFIG[class_id]
            pct = count / len(subset) * 100
            print(f"  Class {class_id}: {count:>4} ({pct:>5.1f}%) - Config {config}")
    
    # Visualization by cache ratio
    print("\n[8] Creating per-ratio visualizations...")
    
    ratios = sorted(merged['ratio'].unique())
    fig2, axes2 = plt.subplots(3, 2, figsize=(16, 15))
    
    for idx, ratio in enumerate(ratios):
        subset_all = merged[merged['ratio'] == ratio]
        subset_train = train_df[train_df['ratio'] == ratio]
        subset_test = test_df[test_df['ratio'] == ratio]
        
        # Count per class for this ratio
        all_c = subset_all['class_id'].value_counts().sort_index()
        train_c = subset_train['class_id'].value_counts().sort_index()
        test_c = subset_test['class_id'].value_counts().sort_index()
        
        # Fill missing classes with 0
        for i in range(18):
            if i not in all_c.index: all_c[i] = 0
            if i not in train_c.index: train_c[i] = 0
            if i not in test_c.index: test_c[i] = 0
        
        all_c = all_c.sort_index()
        train_c = train_c.sort_index()
        test_c = test_c.sort_index()
        
        # Left: All data bar chart
        ax_left = axes2[idx, 0]
        bars = ax_left.bar(x, all_c.values, color=colors, edgecolor='black', linewidth=0.5)
        ax_left.set_xlabel('Class ID', fontsize=11)
        ax_left.set_ylabel('Count', fontsize=11)
        ax_left.set_title(f'Ratio = {ratio}: All Data Class Distribution (n={len(subset_all)})', fontsize=13, fontweight='bold')
        ax_left.set_xticks(x)
        ax_left.set_xticklabels([str(i) for i in range(18)], fontsize=9)
        ax_left.grid(axis='y', alpha=0.3)
        
        # Add count labels on top bars
        for bar, count in zip(bars, all_c.values):
            if count > 0:
                height = bar.get_height()
                ax_left.annotate(f'{count}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=7, rotation=45)
        
        # Right: Train vs Test bar chart (percentage)
        ax_right = axes2[idx, 1]
        width = 0.35
        train_pct_ratio = train_c.values / train_c.sum() * 100 if train_c.sum() > 0 else train_c.values
        test_pct_ratio = test_c.values / test_c.sum() * 100 if test_c.sum() > 0 else test_c.values
        ax_right.bar(x - width/2, train_pct_ratio, width, label=f'Train (n={len(subset_train)})', 
                     color='steelblue', edgecolor='black', linewidth=0.5)
        ax_right.bar(x + width/2, test_pct_ratio, width, label=f'Test (n={len(subset_test)})', 
                     color='coral', edgecolor='black', linewidth=0.5)
        ax_right.set_xlabel('Class ID', fontsize=11)
        ax_right.set_ylabel('Percentage (%)', fontsize=11)
        ax_right.set_title(f'Ratio = {ratio}: Train vs Test (%)', fontsize=13, fontweight='bold')
        ax_right.set_xticks(x)
        ax_right.set_xticklabels([str(i) for i in range(18)], fontsize=9)
        ax_right.legend(fontsize=10)
        ax_right.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_distribution_by_ratio.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"  Saved per-ratio visualization to: {output_path2}")
    
    plt.show()
    
    # ========================================================================
    # Visualization by Dataset
    # ========================================================================
    print("\n[9] Creating per-dataset visualizations...")
    
    def extract_dataset(trace_name):
        """Extract dataset name from trace name based on trace.txt folder structure
        
        Mapping from trace.txt:
        - tencentBlock: tencentBlock.ns*
        - cf: cf_colo*, cf_allcolo*
        - alibabaBlock: io_traces.ns*
        - akamai: akamai_*
        - cphy: w* (like w99, w98)
        - sample: cluster* (in sample/twr/)
        - msr: msr*
        - fiu: fiu*
        - systor: systor*
        - metaStorage: metaStorage*
        - metaKV: metaKV*
        - wiki: wiki*
        - metaCDN: metaCDN*
        - tencentPhoto: tencentPhoto*
        """
        trace = str(trace_name)
        
        if trace.startswith('tencentBlock'):
            return 'tencentBlock'
        elif trace.startswith('cf_'):
            return 'cf'
        elif trace.startswith('io_traces'):
            return 'alibabaBlock'  # io_traces are in alibabaBlock folder
        elif trace.startswith('akamai'):
            return 'akamai'
        elif trace.startswith('w') and trace[1:].split('.')[0].isdigit():
            return 'cphy'  # w99, w98, etc.
        elif trace.startswith('cluster'):
            return 'sample'  # cluster* are in sample/twr/
        elif trace.startswith('msr'):
            return 'msr'
        elif trace.startswith('fiu'):
            return 'fiu'
        elif trace.startswith('systor'):
            return 'systor'
        elif trace.startswith('metaStorage'):
            return 'metaStorage'
        elif trace.startswith('metaKV'):
            return 'metaKV'
        elif trace.startswith('wiki'):
            return 'wiki'
        elif trace.startswith('metaCDN'):
            return 'metaCDN'
        elif trace.startswith('tencentPhoto'):
            return 'tencentPhoto'
        else:
            return 'other'
    
    merged['dataset'] = merged['trace'].apply(extract_dataset)
    
    # Get unique datasets and their counts
    dataset_counts = merged['dataset'].value_counts()
    print(f"\n  Found {len(dataset_counts)} datasets:")
    for ds, cnt in dataset_counts.items():
        print(f"    {ds}: {cnt} samples")
    
    # Create figure - arrange in grid
    n_datasets = len(dataset_counts)
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes3 = axes3.flatten() if n_datasets > 1 else [axes3]
    
    for idx, dataset in enumerate(dataset_counts.index):
        subset = merged[merged['dataset'] == dataset]
        counts = subset['class_id'].value_counts().sort_index()
        
        # Fill missing classes
        for i in range(18):
            if i not in counts.index:
                counts[i] = 0
        counts = counts.sort_index()
        
        # Calculate percentage
        pct = counts.values / counts.sum() * 100 if counts.sum() > 0 else counts.values
        
        ax = axes3[idx]
        bars = ax.bar(x, pct, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Class ID', fontsize=10)
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_title(f'{dataset} (n={len(subset)}, {len(subset["trace"].unique())} traces)', 
                     fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(18)], fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(pct) * 1.15 if max(pct) > 0 else 1)
        
        # Add percentage labels on significant bars
        for bar, p in zip(bars, pct):
            if p > 5:  # Only label bars > 5%
                height = bar.get_height()
                ax.annotate(f'{p:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)
    
    # Hide unused subplots
    for idx in range(n_datasets, len(axes3)):
        axes3[idx].set_visible(False)
    
    plt.suptitle('Class Distribution by Dataset', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    output_path3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_distribution_by_dataset.png')
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"  Saved per-dataset visualization to: {output_path3}")
    
    plt.show()
    
    # Print summary table by dataset
    print("\n" + "="*100)
    print("Top 3 Classes by Dataset")
    print("="*100)
    for dataset in dataset_counts.index:
        subset = merged[merged['dataset'] == dataset]
        counts = subset['class_id'].value_counts()
        top3 = counts.head(3)
        print(f"\n{dataset} ({len(subset)} samples):")
        for class_id, count in top3.items():
            config = ID_TO_CONFIG[class_id]
            pct = count / len(subset) * 100
            print(f"  Class {class_id}: {pct:>5.1f}% - Config {config}")
    
    plt.show()
    
    print("\n" + "="*80)
    print("Analysis complete!")


if __name__ == '__main__':
    main()
