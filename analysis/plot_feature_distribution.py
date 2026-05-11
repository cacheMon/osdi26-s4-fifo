#!/usr/bin/env python3
"""
Visualize feature distribution to demonstrate training data diversity.
Shows that features have wide range coverage for model generalization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
DATA_PATH = 'gpu/features_20pct.csv'
OUT_DIR = 'cleaned/feature_distribution'
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

# Exclude non-feature columns
exclude_cols = ['trace', 'ratio']
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"Feature columns: {len(feature_cols)}")

# Key features to highlight
KEY_FEATURES = ['H_g', 'H_m', 'H_s', 'rho_onehit', 'rho_unique', 'log_C']

# ============================================================================
# 1. Histogram grid for key features
# ============================================================================
print("\n1. Generating key feature histograms...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(KEY_FEATURES):
    ax = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel(col, fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {col}', fontweight='bold')
    
    # Add statistics
    ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.3f}')
    ax.axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.3f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/key_features_histogram.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/key_features_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: key_features_histogram.pdf")

# ============================================================================
# 2. Box plots for all histogram features (grouped)
# ============================================================================
print("\n2. Generating histogram feature box plots...")

# Small queue histograms
small_cols = [c for c in feature_cols if c.startswith('hist_small_')]
main_cols = [c for c in feature_cols if c.startswith('hist_main_')]
ghost_cols = [c for c in feature_cols if c.startswith('hist_ghost_')]

for name, cols in [('small', small_cols), ('main', main_cols), ('ghost', ghost_cols)]:
    if not cols:
        continue
    fig, ax = plt.subplots(figsize=(14, 6))
    box_data = [df[c].dropna() for c in cols]
    bp = ax.boxplot(box_data, labels=[c.split('_')[-1] for c in cols], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xlabel('Bucket Index', fontweight='bold')
    ax.set_ylabel('Hit Position Probability', fontweight='bold')
    ax.set_title(f'Hit Position Distribution in {name.capitalize()} Queue', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/hist_{name}_boxplot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/hist_{name}_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: hist_{name}_boxplot.pdf")

# ============================================================================
# 3. Feature range summary table
# ============================================================================
print("\n3. Generating feature range summary...")
summary_data = []
for col in feature_cols:
    data = df[col].dropna()
    summary_data.append({
        'Feature': col,
        'Min': data.min(),
        'P5': data.quantile(0.05),
        'P25': data.quantile(0.25),
        'Median': data.median(),
        'Mean': data.mean(),
        'P75': data.quantile(0.75),
        'P95': data.quantile(0.95),
        'Max': data.max(),
        'Std': data.std(),
        'Range': data.max() - data.min()
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f'{OUT_DIR}/feature_statistics.csv', index=False)
print(f"  Saved: feature_statistics.csv")

# ============================================================================
# 4. Ratio distribution (cache size ratios in training)
# ============================================================================
print("\n4. Generating ratio distribution...")
fig, ax = plt.subplots(figsize=(8, 6))
ratio_counts = df['ratio'].value_counts().sort_index()
bars = ax.bar(range(len(ratio_counts)), ratio_counts.values, color='steelblue', alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(ratio_counts)))
ax.set_xticklabels([f'{r:.3f}' for r in ratio_counts.index], rotation=45, ha='right')
ax.set_xlabel('Cache Size Ratio', fontweight='bold')
ax.set_ylabel('Number of Samples', fontweight='bold')
ax.set_title('Training Data Distribution by Cache Size Ratio', fontweight='bold')

for bar, val in zip(bars, ratio_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val}', ha='center', va='bottom', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/ratio_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/ratio_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: ratio_distribution.pdf")

# ============================================================================
# 5. Correlation heatmap of key features
# ============================================================================
print("\n5. Generating feature correlation heatmap...")
key_cols = KEY_FEATURES + ['ratio']
corr_matrix = df[key_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

ax.set_xticks(range(len(key_cols)))
ax.set_yticks(range(len(key_cols)))
ax.set_xticklabels(key_cols, rotation=45, ha='right')
ax.set_yticklabels(key_cols)

# Add correlation values
for i in range(len(key_cols)):
    for j in range(len(key_cols)):
        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', fontsize=10)

plt.colorbar(im, ax=ax, label='Correlation')
ax.set_title('Feature Correlation Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/feature_correlation.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/feature_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: feature_correlation.pdf")

# ============================================================================
# 6. 2D scatter plots showing feature space coverage
# ============================================================================
print("\n6. Generating 2D feature scatter plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
pairs = [
    ('H_m', 'H_s'),
    ('H_g', 'H_m'),
    ('rho_onehit', 'rho_unique'),
    ('H_m', 'rho_onehit'),
    ('log_C', 'H_m'),
    ('log_C', 'rho_unique')
]

for ax, (x_col, y_col) in zip(axes.flatten(), pairs):
    ax.scatter(df[x_col], df[y_col], alpha=0.3, s=5, c='steelblue')
    ax.set_xlabel(x_col, fontweight='bold')
    ax.set_ylabel(y_col, fontweight='bold')
    ax.set_title(f'{x_col} vs {y_col}', fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/feature_scatter_2d.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/feature_scatter_2d.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: feature_scatter_2d.pdf")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*60}")
print(f"Feature Distribution Analysis Complete!")
print(f"Output directory: {OUT_DIR}/")
print(f"{'='*60}")

print("\nKey observations:")
print(f"  - Total samples: {len(df)}")
print(f"  - Unique traces: {df['trace'].nunique()}")
print(f"  - Cache ratios: {sorted(df['ratio'].unique())}")
print(f"  - H_m range: [{df['H_m'].min():.3f}, {df['H_m'].max():.3f}]")
print(f"  - H_s range: [{df['H_s'].min():.3f}, {df['H_s'].max():.3f}]")
print(f"  - rho_unique range: [{df['rho_unique'].min():.3f}, {df['rho_unique'].max():.3f}]")
