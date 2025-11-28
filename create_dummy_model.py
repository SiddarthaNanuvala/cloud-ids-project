import torch
import json

# Create minimal dummy model for Docker build
# In production, this is replaced with trained model from train_autoencoder.py

dummy_ae = {"state": "placeholder"}
torch.save(dummy_ae, "model/ae.pth")

dummy_scaler = {"data": "placeholder"}
torch.save(dummy_scaler, "model/scaler.joblib")

threshold = {"threshold": 0.5}
with open("model/threshold.json", "w") as f:
    json.dump(threshold, f)

print("Created minimal model artifacts for Docker build")
