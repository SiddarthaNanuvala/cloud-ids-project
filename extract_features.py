#!/usr/bin/env python3
import joblib
import json

# Load model.pkl and extract features
m = joblib.load('model.pkl')
features = m['feature_names']

# Create threshold.json
threshold_config = {
    "threshold": 0.5,
    "feature_columns": features
}

# Save to app/model/threshold.json
with open('app/model/threshold.json', 'w') as f:
    json.dump(threshold_config, f, indent=2)

print(f"✓ Created threshold.json with {len(features)} features")
print(f"Features: {features}")
