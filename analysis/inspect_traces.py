import pandas as pd
import os

script_dir = "/users/Haocheng/ana"
cleaned_dir = os.path.join(script_dir, 'cleaned')

features_df = pd.read_csv(os.path.join(script_dir, 'features_20pct.csv'))
print("Features traces (head):")
print(features_df['trace'].head(10).tolist())

opt = pd.read_csv(os.path.join(cleaned_dir, '0.001', 'grid_optimal.csv'))
print("\nOptimal traces (head):")
print(opt['trace'].head(10).tolist())

print("\nFeatures traces (sample of non-matching):")
common = set(opt['trace'])
diff = [t for t in features_df['trace'] if t not in common]
print(diff[:10])
print(f"Total non-matching in features: {len(diff)}")
