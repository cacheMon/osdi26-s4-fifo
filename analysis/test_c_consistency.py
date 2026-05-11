#!/usr/bin/env python3
"""
Test C code consistency with Python sorted model.
Compares predictions from:
1. Python sorted model (ensemble_models.pkl with sorted features)
2. C code (compiled ensemble)
"""
import os
import json
import numpy as np
import joblib
import subprocess
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load sorted feature columns
features_path = os.path.join(SCRIPT_DIR, 'xgb_18class_results', 'feature_columns.json')
with open(features_path) as f:
    FEATURE_COLS = json.load(f)

print(f"Feature columns: {len(FEATURE_COLS)}")
print(f"First 5: {FEATURE_COLS[:5]}")

# Load sorted model
models_path = os.path.join(SCRIPT_DIR, 'xgb_18class_results', 'ensemble_models.pkl')
models = joblib.load(models_path)
print(f"Loaded {len(models)} models")

# Create a test input (random but fixed for reproducibility)
np.random.seed(42)
test_input = np.random.randn(73).astype(np.float64)

# Python prediction
print("\n=== Python Prediction ===")
probs = np.zeros(18)
for model in models:
    probs += model.predict_proba([test_input])[0]
probs /= len(models)

py_pred = np.argmax(probs)
print(f"Predicted class: {py_pred}")
print(f"Top-3 probs: {np.argsort(-probs)[:3]} = {probs[np.argsort(-probs)[:3]]}")

# Check if C code exists
c_dir = os.path.join(SCRIPT_DIR, 'xgb_18class_results', 'code', 'c')
ensemble_path = os.path.join(c_dir, 'ensemble.c')
if not os.path.exists(ensemble_path):
    print("\nC ensemble.c not found! Export may still be running.")
    exit(1)

print(f"\n=== Compiling C Test ===")

# Create C test file
c_test_code = '''
#include <stdio.h>
#include <stdlib.h>
#include "ensemble.c"

int main() {
    // Same random input as Python (numpy seed=42)
    double input[73] = {
'''

# Add the test input values
for i, val in enumerate(test_input):
    c_test_code += f"        {val:.17g}"
    if i < 72:
        c_test_code += ","
    c_test_code += "\n"

c_test_code += '''    };
    
    double probs[18];
    ensemble_score(input, probs);
    
    int pred = ensemble_predict(input);
    
    printf("Predicted class: %d\\n", pred);
    printf("Probabilities:\\n");
    for (int i = 0; i < 18; i++) {
        printf("  Class %d: %.10f\\n", i, probs[i]);
    }
    
    return 0;
}
'''

# Write test file
test_c_path = os.path.join(c_dir, 'test_consistency.c')
with open(test_c_path, 'w') as f:
    f.write(c_test_code)
print(f"Created {test_c_path}")

# Compile
test_bin_path = os.path.join(c_dir, 'test_consistency')
compile_cmd = ['gcc', '-O2', '-o', test_bin_path, test_c_path, '-lm']
result = subprocess.run(compile_cmd, cwd=c_dir, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Compilation failed: {result.stderr}")
    exit(1)
print("Compiled successfully")

# Run
print("\n=== C Prediction ===")
result = subprocess.run([test_bin_path], capture_output=True, text=True)
print(result.stdout)

# Parse C output
lines = result.stdout.strip().split('\n')
c_pred = None
c_probs = []
for line in lines:
    if 'Predicted class:' in line:
        c_pred = int(line.split(':')[1].strip())
    if 'Class' in line and ':' in line:
        prob = float(line.split(':')[1].strip())
        c_probs.append(prob)

c_probs = np.array(c_probs)

# Compare
print("\n=== Comparison ===")
print(f"Python pred: {py_pred}, C pred: {c_pred}")
print(f"Match: {py_pred == c_pred}")

if len(c_probs) == 18:
    max_diff = np.max(np.abs(probs - c_probs))
    print(f"Max probability difference: {max_diff:.2e}")
    if max_diff < 1e-6:
        print("✓ Probabilities match within tolerance!")
    else:
        print("✗ Probabilities differ!")
        for i in range(18):
            if abs(probs[i] - c_probs[i]) > 1e-6:
                print(f"  Class {i}: Python={probs[i]:.10f}, C={c_probs[i]:.10f}")
