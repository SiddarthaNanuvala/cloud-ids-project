# ML Pipeline Scripts

## Overview

4-step machine learning pipeline for anomaly detection.

## 1. Preprocessing

```bash
python scripts/preprocess_cicids2017.py --input "Raw Data" --out features
```

**Input:** 8 CIC-IDS2017 CSV files  
**Output:** 
- `features/train_normal.csv` - Training data (benign samples)
- `features/val.csv` - Validation data
- `features/test.csv` - Test data
- `features/feature_schema.json` - Feature metadata

**What it does:**
- Loads 8 CSV files (3.1M records)
- Cleans data (removes Infinity, NaN, duplicates)
- Selects 22 numeric features
- Splits into 70% train / 15% val / 15% test

## 2. Training

```bash
python scripts/train_autoencoder.py --data features --out model
```

**Input:** `features/` directory from preprocessing  
**Output:**
- `model/ae.pth` - Trained autoencoder weights
- `model/scaler.joblib` - StandardScaler for normalization
- `model/threshold.json` - Anomaly detection threshold

**What it does:**
- Builds PyTorch autoencoder (n→64→16→64→n)
- Trains on benign samples (60 epochs)
- Computes 99th percentile reconstruction error threshold

## 3. Evaluation

```bash
python scripts/evaluate_model.py --model model --data features
```

**Output:** `model/evaluation_results.json`

**Metrics:**
- Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Sensitivity, Specificity
- Confusion matrix

## 4. Load Testing

```bash
python -m locust -f scripts/locustfile.py
```

**Tests:** 500 concurrent users with realistic traffic patterns

## Training Data

- **Source:** CIC-IDS2017 network flow dataset
- **Records:** 3.1M raw → 2.83M cleaned
- **Features:** 85 total → 22 selected numeric
- **Split:** 70% train (benign) / 15% val / 15% test
- **Format:** CSV

## Model Details

**Architecture:**
```
Input (22 features)
  ↓
Dense(64, ReLU)
  ↓
Dense(16, ReLU)  ← Bottleneck
  ↓
Dense(64, ReLU)
  ↓
Output (22 features)
```

**Training:**
- Optimizer: Adam
- Loss: MSE (reconstruction error)
- Epochs: 60
- Batch Size: 32
- Learning Rate: 0.001

**Inference:**
- Reconstruction error > threshold = Anomaly
- Threshold: 99th percentile of validation benign errors

## Performance

- **Training Time:** ~5 minutes
- **Inference:** <1ms per sample
- **Accuracy:** ~90% anomaly detection
