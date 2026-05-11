#!/usr/bin/env python3
"""
Create grid_optimal.csv from grid_full.csv by selecting the best (minimum miss_ratio) 
configuration for each trace.
"""
import os
import pandas as pd

def create_optimal_from_full(ratio_dir):
    """Process one ratio directory"""
    full_path = os.path.join(ratio_dir, 'grid_full.csv')
    optimal_path = os.path.join(ratio_dir, 'grid_optimal.csv')
    
    if not os.path.exists(full_path):
        print(f"Warning: {full_path} not found")
        return
    
    # Load full grid data
    full_df = pd.read_csv(full_path)
    print(f"\nProcessing {ratio_dir}")
    print(f"  Loaded {len(full_df)} rows from grid_full.csv")
    print(f"  Unique traces: {full_df['trace'].nunique()}")
    
    # For each trace, find the row with minimum miss_ratio
    optimal_df = full_df.loc[full_df.groupby('trace')['miss_ratio'].idxmin()]
    
    print(f"  Created optimal with {len(optimal_df)} rows (one per trace)")
    print(f"  Mean optimal miss_ratio: {optimal_df['miss_ratio'].mean():.6f}")
    
    # Save to grid_optimal.csv
    optimal_df.to_csv(optimal_path, index=False)
    print(f"  Saved: {optimal_path}")
    
    return optimal_df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cleaned_dir = os.path.join(script_dir, 'cleaned')
    
    ratios = ['0.001', '0.01', '0.1']
    
    for ratio_str in ratios:
        ratio_dir = os.path.join(cleaned_dir, ratio_str)
        if os.path.exists(ratio_dir):
            create_optimal_from_full(ratio_dir)
        else:
            print(f"Warning: {ratio_dir} not found")
    
    print("\n" + "="*80)
    print("Grid optimal files created successfully!")
    print("="*80)

if __name__ == '__main__':
    main()
