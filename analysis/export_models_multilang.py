#!/usr/bin/env python3
"""Export lightweight models to multiple languages: C, Ruby, Java, JavaScript, Go, Rust."""
import os
import sys
import re
import joblib
import m2cgen as m2c

sys.setrecursionlimit(50000)

# Supported languages and their file extensions
LANGUAGES = {
    'c': {'ext': 'c', 'exporter': m2c.export_to_c},
    'ruby': {'ext': 'rb', 'exporter': m2c.export_to_ruby},
    'java': {'ext': 'java', 'exporter': m2c.export_to_java},
    'javascript': {'ext': 'js', 'exporter': m2c.export_to_javascript},
    'go': {'ext': 'go', 'exporter': m2c.export_to_go},
    'rust': {'ext': 'rs', 'exporter': m2c.export_to_rust},
}

# Config entries for the 18 classes
CONFIGS = [
    (0.20, 1, 0, 3.0, 0.25),  # Class 0
    (0.05, 1, 0, 0.9, 0.25),  # Class 1
    (0.50, 1, 0, 0.9, 0.25),  # Class 2
    (0.20, 1, 0, 0.9, 0.25),  # Class 3
    (0.05, 2, 0, 6.0, 0.25),  # Class 4
    (0.10, 2, 1, 3.0, 0.25),  # Class 5
    (0.30, 2, 0, 3.0, 0.25),  # Class 6
    (0.05, 2, 0, 3.0, 0.25),  # Class 7
    (0.10, 2, 0, 0.9, 0.25),  # Class 8
    (0.70, 1, 1, 0.9, 0.25),  # Class 9
    (0.20, 1, 1, 0.9, 0.25),  # Class 10
    (0.05, 1, 1, 0.9, 0.25),  # Class 11
    (0.30, 1, 0, 6.0, 0.25),  # Class 12
    (0.20, 2, 0, 0.9, 0.25),  # Class 13
    (0.90, 2, 0, 3.0, 0.25),  # Class 14
    (0.10, 2, 0, 6.0, 0.25),  # Class 15
    (0.30, 2, 1, 3.0, 0.25),  # Class 16
    (0.05, 2, 0, 0.9, 0.25),  # Class 17
]

def rename_functions_c(code, model_idx):
    """Rename C functions to avoid conflicts."""
    code = code.replace('void softmax(', f'void softmax_{model_idx}(')
    code = code.replace('softmax(', f'softmax_{model_idx}(')
    code = code.replace('void score(', f'void score_{model_idx}(')
    return code

def rename_functions_java(code, model_idx):
    """Rename Java class and methods."""
    code = code.replace('public class Model', f'public class Model{model_idx}')
    code = code.replace('public static double[] score(', f'public static double[] score{model_idx}(')
    return code

def rename_functions_ruby(code, model_idx):
    """Rename Ruby methods."""
    code = code.replace('def score(', f'def score_{model_idx}(')
    code = code.replace('def softmax(', f'def softmax_{model_idx}(')
    return code

def rename_functions_js(code, model_idx):
    """Rename JavaScript functions."""
    code = code.replace('function score(', f'function score_{model_idx}(')
    code = code.replace('function softmax(', f'function softmax_{model_idx}(')
    code = code.replace('softmax(', f'softmax_{model_idx}(')
    return code

def rename_functions_go(code, model_idx):
    """Rename Go functions."""
    code = code.replace('func Score(', f'func Score{model_idx}(')
    code = code.replace('func softmax(', f'func softmax{model_idx}(')
    code = code.replace('softmax(', f'softmax{model_idx}(')
    return code

def rename_functions_rust(code, model_idx):
    """Rename Rust functions."""
    code = code.replace('fn score(', f'fn score_{model_idx}(')
    code = code.replace('fn softmax(', f'fn softmax_{model_idx}(')
    code = code.replace('softmax(', f'softmax_{model_idx}(')
    return code

RENAMERS = {
    'c': rename_functions_c,
    'ruby': rename_functions_ruby,
    'java': rename_functions_java,
    'javascript': rename_functions_js,
    'go': rename_functions_go,
    'rust': rename_functions_rust,
}

def generate_ensemble_c(n_models, output_dir):
    """Generate C ensemble wrapper."""
    code = '#include <math.h>\n// Ensemble aggregation wrapper\n'
    for i in range(n_models):
        code += f'#include "model_{i}.c"\n'
    
    code += f'''
void ensemble_score(double *input, double *result) {{
'''
    for i in range(n_models):
        code += f'    double probs_{i}[18];\n'
    
    code += '\n    // Call each model\n'
    for i in range(n_models):
        code += f'    score_{i}(input, probs_{i});\n'
    
    code += '''
    // Average probabilities
    for (int c = 0; c < 18; c++) result[c] = 0.0;
    for (int c = 0; c < 18; c++) {
'''
    for i in range(n_models):
        code += f'        result[c] += probs_{i}[c];\n'
    code += f'''    }}
    for (int c = 0; c < 18; c++) result[c] /= {n_models}.0;
}}

int ensemble_predict(double *input) {{
    double probs[18];
    ensemble_score(input, probs);
    int best = 0;
    for (int c = 1; c < 18; c++) {{
        if (probs[c] > probs[best]) best = c;
    }}
    return best;
}}
'''
    with open(os.path.join(output_dir, 'ensemble.c'), 'w') as f:
        f.write(code)

def generate_ensemble_js(n_models, output_dir):
    """Generate JavaScript ensemble wrapper."""
    code = '// Ensemble wrapper\n'
    for i in range(n_models):
        code += f'// Include model_{i}.js\n'
    
    code += f'''
function ensembleScore(input) {{
    let result = new Array(18).fill(0);
'''
    for i in range(n_models):
        code += f'    let probs_{i} = score_{i}(input);\n'
    
    code += '''
    // Average probabilities
    for (let c = 0; c < 18; c++) {
'''
    for i in range(n_models):
        code += f'        result[c] += probs_{i}[c];\n'
    code += f'''    }}
    for (let c = 0; c < 18; c++) result[c] /= {n_models};
    return result;
}}

function ensemblePredict(input) {{
    let probs = ensembleScore(input);
    let best = 0;
    for (let c = 1; c < 18; c++) {{
        if (probs[c] > probs[best]) best = c;
    }}
    return best;
}}

// Config lookup
const CONFIGS = [
'''
    for i, (s, m, t, g, k) in enumerate(CONFIGS):
        code += f'    {{ s: {s}, m: {m}, t: {t}, g: {g}, k: {k} }},  // Class {i}\n'
    code += '''];

function predictConfig(input) {
    let classId = ensemblePredict(input);
    return CONFIGS[classId];
}

module.exports = { ensembleScore, ensemblePredict, predictConfig, CONFIGS };
'''
    with open(os.path.join(output_dir, 'ensemble.js'), 'w') as f:
        f.write(code)

def generate_ensemble_ruby(n_models, output_dir):
    """Generate Ruby ensemble wrapper."""
    code = '# Ensemble wrapper\n'
    for i in range(n_models):
        code += f"require_relative 'model_{i}'\n"
    
    code += f'''
CONFIGS = [
'''
    for i, (s, m, t, g, k) in enumerate(CONFIGS):
        code += f'  {{ s: {s}, m: {m}, t: {t}, g: {g}, k: {k} }},  # Class {i}\n'
    code += ''']

def ensemble_score(input)
  result = Array.new(18, 0.0)
'''
    for i in range(n_models):
        code += f'  probs_{i} = score_{i}(input)\n'
    
    code += '''
  # Average probabilities
  18.times do |c|
'''
    for i in range(n_models):
        code += f'    result[c] += probs_{i}[c]\n'
    code += f'''  end
  result.map {{ |p| p / {n_models}.0 }}
end

def ensemble_predict(input)
  probs = ensemble_score(input)
  probs.each_with_index.max[1]
end

def predict_config(input)
  class_id = ensemble_predict(input)
  CONFIGS[class_id]
end
'''
    with open(os.path.join(output_dir, 'ensemble.rb'), 'w') as f:
        f.write(code)

def generate_ensemble_go(n_models, output_dir):
    """Generate Go ensemble wrapper."""
    code = '''package model

import "math"

// S3FIFOConfig represents the predicted configuration
type S3FIFOConfig struct {
    S float64
    M int
    T int
    G float64
    K float64
}

var configs = []S3FIFOConfig{
'''
    for i, (s, m, t, g, k) in enumerate(CONFIGS):
        code += f'    {{S: {s}, M: {m}, T: {t}, G: {g}, K: {k}}},  // Class {i}\n'
    
    code += f'''}}

func EnsembleScore(input []float64) []float64 {{
    result := make([]float64, 18)
'''
    for i in range(n_models):
        code += f'    probs{i} := Score{i}(input)\n'
    
    code += '''
    // Average probabilities
    for c := 0; c < 18; c++ {
'''
    for i in range(n_models):
        code += f'        result[c] += probs{i}[c]\n'
    code += f'''    }}
    for c := 0; c < 18; c++ {{
        result[c] /= {n_models}.0
    }}
    return result
}}

func EnsemblePredict(input []float64) int {{
    probs := EnsembleScore(input)
    best := 0
    for c := 1; c < 18; c++ {{
        if probs[c] > probs[best] {{
            best = c
        }}
    }}
    return best
}}

func PredictConfig(input []float64) S3FIFOConfig {{
    classID := EnsemblePredict(input)
    return configs[classID]
}}

// Suppress unused import warning
var _ = math.Exp
'''
    with open(os.path.join(output_dir, 'ensemble.go'), 'w') as f:
        f.write(code)

def generate_ensemble_rust(n_models, output_dir):
    """Generate Rust ensemble wrapper."""
    code = '''// Ensemble wrapper
'''
    for i in range(n_models):
        code += f'mod model_{i};\n'
    
    code += f'''
pub struct S3FIFOConfig {{
    pub s: f64,
    pub m: i32,
    pub t: i32,
    pub g: f64,
    pub k: f64,
}}

const CONFIGS: [S3FIFOConfig; 18] = [
'''
    for i, (s, m, t, g, k) in enumerate(CONFIGS):
        code += f'    S3FIFOConfig {{ s: {s}, m: {m}, t: {t}, g: {g}, k: {k} }},  // Class {i}\n'
    
    code += f'''];

pub fn ensemble_score(input: &[f64; 73]) -> [f64; 18] {{
    let mut result = [0.0; 18];
'''
    for i in range(n_models):
        code += f'    let probs_{i} = model_{i}::score_{i}(input);\n'
    
    code += '''
    // Average probabilities
    for c in 0..18 {
'''
    for i in range(n_models):
        code += f'        result[c] += probs_{i}[c];\n'
    code += f'''    }}
    for c in 0..18 {{
        result[c] /= {n_models}.0;
    }}
    result
}}

pub fn ensemble_predict(input: &[f64; 73]) -> usize {{
    let probs = ensemble_score(input);
    let mut best = 0;
    for c in 1..18 {{
        if probs[c] > probs[best] {{
            best = c;
        }}
    }}
    best
}}

pub fn predict_config(input: &[f64; 73]) -> &'static S3FIFOConfig {{
    let class_id = ensemble_predict(input);
    &CONFIGS[class_id]
}}
'''
    with open(os.path.join(output_dir, 'ensemble.rs'), 'w') as f:
        f.write(code)

def generate_ensemble_java(n_models, output_dir):
    """Generate Java ensemble wrapper."""
    code = '''// Ensemble wrapper
public class Ensemble {
    
    public static class S3FIFOConfig {
        public double s;
        public int m;
        public int t;
        public double g;
        public double k;
        
        public S3FIFOConfig(double s, int m, int t, double g, double k) {
            this.s = s; this.m = m; this.t = t; this.g = g; this.k = k;
        }
    }
    
    private static final S3FIFOConfig[] CONFIGS = {
'''
    for i, (s, m, t, g, k) in enumerate(CONFIGS):
        code += f'        new S3FIFOConfig({s}, {m}, {t}, {g}, {k}),  // Class {i}\n'
    
    code += f'''    }};
    
    public static double[] ensembleScore(double[] input) {{
        double[] result = new double[18];
'''
    for i in range(n_models):
        code += f'        double[] probs{i} = Model{i}.score{i}(input);\n'
    
    code += '''
        // Average probabilities
        for (int c = 0; c < 18; c++) {
'''
    for i in range(n_models):
        code += f'            result[c] += probs{i}[c];\n'
    code += f'''        }}
        for (int c = 0; c < 18; c++) {{
            result[c] /= {n_models}.0;
        }}
        return result;
    }}
    
    public static int ensemblePredict(double[] input) {{
        double[] probs = ensembleScore(input);
        int best = 0;
        for (int c = 1; c < 18; c++) {{
            if (probs[c] > probs[best]) best = c;
        }}
        return best;
    }}
    
    public static S3FIFOConfig predictConfig(double[] input) {{
        int classId = ensemblePredict(input);
        return CONFIGS[classId];
    }}
}}
'''
    with open(os.path.join(output_dir, 'Ensemble.java'), 'w') as f:
        f.write(code)

ENSEMBLE_GENERATORS = {
    'c': generate_ensemble_c,
    'javascript': generate_ensemble_js,
    'ruby': generate_ensemble_ruby,
    'go': generate_ensemble_go,
    'rust': generate_ensemble_rust,
    'java': generate_ensemble_java,
}

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(script_dir, 'xgb_18class_results_lite', 'ensemble_models.pkl')
    
    print("Loading models...")
    models = joblib.load(models_path)
    print(f"Loaded {len(models)} models")
    
    base_output_dir = os.path.join(script_dir, 'xgb_18class_results_lite', 'code')
    
    for lang_name, lang_info in LANGUAGES.items():
        print(f"\n{'='*60}")
        print(f"Exporting to {lang_name.upper()}")
        print('='*60)
        
        output_dir = os.path.join(base_output_dir, lang_name)
        os.makedirs(output_dir, exist_ok=True)
        
        exporter = lang_info['exporter']
        ext = lang_info['ext']
        renamer = RENAMERS.get(lang_name)
        
        for i, model in enumerate(models):
            print(f"  [{i+1}/{len(models)}] model_{i}.{ext}...", end=" ", flush=True)
            
            try:
                if lang_name == 'java':
                    code = exporter(model, class_name=f'Model{i}')
                else:
                    code = exporter(model)
                
                if renamer:
                    code = renamer(code, i)
                
                out_path = os.path.join(output_dir, f'model_{i}.{ext}')
                with open(out_path, 'w') as f:
                    f.write(code)
                
                size_mb = os.path.getsize(out_path) / (1024 * 1024)
                print(f"OK ({size_mb:.2f} MB)")
            except Exception as e:
                print(f"FAILED: {e}")
        
        # Generate ensemble wrapper
        if lang_name in ENSEMBLE_GENERATORS:
            print(f"  Generating ensemble.{ext}...")
            ENSEMBLE_GENERATORS[lang_name](len(models), output_dir)
        
        # Report total size
        total_size = sum(
            os.path.getsize(os.path.join(output_dir, f)) 
            for f in os.listdir(output_dir) 
            if f.endswith(f'.{ext}')
        )
        print(f"  Total size: {total_size / (1024*1024):.2f} MB")
    
    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)

if __name__ == '__main__':
    main()
