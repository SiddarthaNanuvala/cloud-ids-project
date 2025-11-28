# Model Artifacts

This directory should contain:
- `ae.pth` - Trained PyTorch autoencoder
- `scaler.joblib` - StandardScaler for feature normalization  
- `threshold.json` - Anomaly detection threshold

## To Generate Models

Run the training pipeline:
```bash
python scripts/preprocess_cicids2017.py --input "Raw Data" --out features
python scripts/train_autoencoder.py --data features --out model
```

This will create the three model artifacts needed for inference.
