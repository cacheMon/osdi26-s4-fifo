#!/usr/bin/env python3
"""
Run training multiple times to find the best feature order.
Each run will save the feature order and results.
"""
import subprocess
import json
import os
import re

N_RUNS = 10
results_dir = 'feature_order_search'
os.makedirs(results_dir, exist_ok=True)

best_worst = -float('inf')
best_run = None

for run_id in range(N_RUNS):
    print(f"\n{'='*60}")
    print(f"RUN {run_id + 1}/{N_RUNS}")
    print(f"{'='*60}")
    
    # Run training and capture output
    result = subprocess.run(
        ['python3', 'train_xgb_18class.py'],
        capture_output=True,
        text=True,
        cwd='/users/Haocheng/ana'
    )
    
    output = result.stdout + result.stderr
    
    # Parse worst performance
    worst_match = re.search(r'Predicted Worst:\s*([-\d.]+)%', output)
    if worst_match:
        worst_perf = float(worst_match.group(1))
    else:
        worst_perf = -999
        print("Could not parse worst performance!")
    
    # Parse feature order from output
    features_match = re.search(r"Features \(\d+\): (\[.*?\])", output, re.DOTALL)
    if features_match:
        try:
            feature_order = eval(features_match.group(1))
        except:
            feature_order = None
            print("Could not parse feature order!")
    else:
        feature_order = None
        print("Could not find feature order in output!")
    
    print(f"  Worst: {worst_perf}%")
    
    # Save this run's results
    run_data = {
        'run_id': run_id,
        'worst_perf': worst_perf,
        'feature_order': feature_order
    }
    
    with open(os.path.join(results_dir, f'run_{run_id}.json'), 'w') as f:
        json.dump(run_data, f, indent=2)
    
    # Track best
    if worst_perf > best_worst:
        best_worst = worst_perf
        best_run = run_id
        print(f"  *** NEW BEST! ***")
        
        # Save best separately
        with open(os.path.join(results_dir, 'best_run.json'), 'w') as f:
            json.dump(run_data, f, indent=2)

print(f"\n{'='*60}")
print(f"SEARCH COMPLETE")
print(f"{'='*60}")
print(f"Best run: {best_run}")
print(f"Best worst: {best_worst}%")
print(f"Results saved to: {results_dir}/")
