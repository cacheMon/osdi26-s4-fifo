#!/usr/bin/env python3
"""
Plot feature importance from saved CSV.
Outputs PDF format for paper publication.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Font settings for paper
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Load data
df = pd.read_csv('xgb_18class_results/feature_importance.csv')

# Rename for display
display_names = {
    'hist_ghost': 'Ghost Histogram',
    'hist_main': 'Main Histogram',
    'hist_small': 'Small Histogram',
    'total_reqs': 'Total Requests',
    'thrashing_risk': 'Thrashing Risk',
    'ghost_pressure': 'Ghost Pressure',
    'rho_onehit': 'One-Hit Ratio',
    'rho_unique': 'Unique Ratio',
    'scan_intensity': 'Scan Intensity',
    'decay_rate_small': 'Decay Rate (Small)',
    'H_m': 'Entropy (Main)',
    'tail_heaviness': 'Tail Heaviness',
    'entropy_gap': 'Entropy Gap',
    'H_s': 'Entropy (Small)',
    'probation_efficiency': 'Probation Efficiency',
    'H_g': 'Entropy (Ghost)',
}

df['display_name'] = df['feature'].map(lambda x: display_names.get(x, x))

# Colors: histogram features in one color, others in another
colors = []
for feat in df['feature']:
    if 'hist_' in feat:
        colors.append('#e74c3c')  # Red for histograms
    else:
        colors.append('#3498db')  # Blue for others

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Horizontal bar chart (easier to read feature names)
y_pos = range(len(df))
bars = ax.barh(y_pos, df['importance_pct'].values, color=colors, alpha=0.8, edgecolor='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(df['display_name'].values)
ax.set_xlabel('Importance (%)')
ax.invert_yaxis()  # Highest on top
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, pct) in enumerate(zip(bars, df['importance_pct'].values)):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
           f'{pct:.1f}%', va='center', fontsize=14, fontweight='bold')

plt.tight_layout()

# Save
plt.savefig('xgb_18class_results/feature_importance.pdf', dpi=300, bbox_inches='tight')
print("Saved: xgb_18class_results/feature_importance.pdf")

plt.savefig('xgb_18class_results/feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: xgb_18class_results/feature_importance.png")

plt.close()
