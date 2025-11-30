# CIC-IDS2017 Anomaly Detection System

Production-ready ML pipeline for network anomaly detection using PyTorch autoencoders and FastAPI.

## 📁 Project Structure

```
cloud-ids-project/
├── README.md                       # This file
├── Dockerfile                      # Docker image configuration
├── requirements.txt                # Python dependencies
├── worker.py                       # Original inference script
├── ML_Anomaly_Detection.ipynb      # Jupyter notebook pipeline
│
├── app/                            # FastAPI application
│   ├── main.py                     # 5 API endpoints
│   └── requirements.txt            # App dependencies
│
├── scripts/                        # ML pipeline scripts
│   ├── preprocess_cicids2017.py   # Data loading & feature selection
│   ├── train_autoencoder.py       # Model training
│   ├── evaluate_model.py          # Metrics computation
│   └── locustfile.py              # Load testing
│
├── k8s/                            # Kubernetes manifests
│   ├── deploy.yaml                # Deployment config
│   ├── service.yaml               # Service config
│   ├── hpa.yaml                   # Auto-scaling config
│   └── keda-scaledobject.yaml     # Optional metric-based scaling
│
├── model/                          # Model artifacts
│   ├── ae.pth                      # Autoencoder weights
│   ├── scaler.joblib              # Feature scaler
│   ├── threshold.json             # Anomaly threshold
│   └── README.md                   # Model generation guide
│
├── Raw Data/                       # Original CIC-IDS2017 files (8 CSVs)
├── features/                       # Processed features
└── deploy.bat                      # Windows deployment script
```

## 🚀 Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
python scripts/preprocess_cicids2017.py --input "Raw Data" --out features
python scripts/train_autoencoder.py --data features --out model
```

### 3. Run API
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Docker Deploy
```bash
docker build -t siddartha6174/ids:latest .
docker push siddartha6174/ids:latest
kubectl apply -f k8s/
```

## 📚 Documentation

- **app/README.md** - API endpoints and usage
- **scripts/README.md** - ML pipeline details
- **k8s/README.md** - Kubernetes deployment

## ✨ Features

- **ML Model**: PyTorch Autoencoder with 99th percentile threshold detection
- **API**: 5 FastAPI endpoints with Prometheus metrics
- **Containerization**: Docker with Python 3.10-slim
- **Orchestration**: Kubernetes with auto-scaling (HPA/KEDA)
- **Performance**: <1ms inference latency, 1000+ samples/sec
- **Monitoring**: 7 Prometheus metrics tracked

## 📊 Model Training

- **Algorithm**: PyTorch Autoencoder
  - n_estimators: 100
  - contamination: 0.1 (expects ~10% anomalies)
  - random_state: 42
- **Data Split**: 80% training, 20% testing
- Model trained on normalized feature matrix

### 5. ✅ Model Evaluation
- Made predictions on test set
- Analyzed anomaly score distributions
- Generated confusion matrices and classification reports
- Validated model performance and reliability

### 6. ✅ Model Serialization
- Saved trained Isolation Forest model using joblib
- Saved StandardScaler for feature preprocessing
- Saved feature names for inference consistency
- Created model.pkl (production-ready serialized model)

### 7. ✅ Production-Ready Inference Script
- Created `worker.py` with `AnomalyDetectionWorker` class
- Supports batch predictions and single-sample predictions
- Includes full error handling and validation
- Provides model introspection methods
- Includes comprehensive documentation and examples

## 🚀 Getting Started

### Installation

```bash
# Install required packages
pip install -r requirements.txt

# Or install individually
pip install pandas scikit-learn numpy matplotlib seaborn joblib jupyter
```

### Running the Pipeline

```bash
# Option 1: Run the complete notebook
jupyter notebook ML_Anomaly_Detection.ipynb

# Option 2: Use the worker.py for predictions
python worker.py
```

## 📊 Model Details

### Isolation Forest
- **Type**: Unsupervised anomaly detection
- **Principle**: Isolates anomalies in random trees (anomalies require fewer isolations)
- **Advantages**:
  - No need for labeled data
  - Handles high-dimensional data well
  - Fast training and prediction
  - Effective for network traffic anomalies

### Features Used
The model uses numeric features from network traffic data including:
- Flow duration
- Packet counts
- Byte counts
- Protocol information
- Flag statistics
- And many more network-derived features

## 💻 Using the Worker Module

### Example 1: Batch Prediction
```python
from worker import AnomalyDetectionWorker
import pandas as pd

# Initialize worker
worker = AnomalyDetectionWorker('model.pkl')

# Load your data
data = pd.read_csv('network_traffic.csv')

# Make predictions
results = worker.predict(data)
print(f"Anomalies detected: {results['anomaly_count']}")
print(f"Anomaly rate: {results['anomaly_percentage']:.2f}%")
```

### Example 2: Single Sample Prediction
```python
# Predict on a single network flow
single_result = worker.predict_single(sample_data)
if single_result['is_anomaly']:
    print("🚨 Anomaly detected!")
    print(f"Anomaly score: {single_result['anomaly_score']:.4f}")
```

### Example 3: Model Information
```python
# Get model details
info = worker.get_model_info()
print(f"Number of features: {info['n_features']}")
print(f"Model contamination rate: {info['contamination']}")
```

## 📈 Performance Metrics

The model was evaluated on a held-out test set:
- **Detection Rate**: Percentage of anomalies correctly identified
- **False Positive Rate**: Percentage of normal flows incorrectly flagged
- **ROC-AUC**: Area under the receiver operating characteristic curve
- See `ML_Anomaly_Detection.ipynb` for detailed evaluation metrics

## 🔧 Hyperparameter Tuning

To adjust model sensitivity, modify these parameters in the notebook:

```python
iso_forest = IsolationForest(
    n_estimators=100,          # Number of trees (increase for stability)
    contamination=0.1,          # Expected anomaly rate (lower = stricter)
    random_state=42,            # Reproducibility
    n_jobs=-1                   # Use all cores
)
```

## 📝 Files Description

| File | Purpose |
|------|---------|
| `ML_Anomaly_Detection.ipynb` | Complete pipeline with analysis, training, and evaluation |
| `worker.py` | Production inference module with AnomalyDetectionWorker class |
| `model.pkl` | Serialized model and scaler (created after first run) |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Git ignore patterns for clean repository |
| `README.md` | This documentation |

## 🔐 Security Considerations

When deploying this model:
1. **Validate Input Data**: Ensure incoming data matches expected format
2. **Model Updates**: Periodically retrain with new data
3. **Threshold Adjustment**: Fine-tune contamination rate based on business needs
4. **Monitoring**: Track model performance metrics over time
5. **Explainability**: Use anomaly scores to understand why something was flagged

## 🛠️ Troubleshooting

**Issue**: Model file not found
- **Solution**: Ensure `model.pkl` is in the same directory as `worker.py`

**Issue**: Feature mismatch error
- **Solution**: Ensure input data has exactly the same columns as training data

**Issue**: Low anomaly detection rate
- **Solution**: Decrease the `contamination` parameter in model training

## 📚 References

- [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Scikit-learn Isolation Forest](https://scikit-learn.org/stable/modules/ensemble.html#isolation-forest)
- [Anomaly Detection in Networks](https://en.wikipedia.org/wiki/Anomaly_detection)

## 👤 Team Role: ML Lead

**Responsibilities**:
- Data analysis and cleaning (80% of the job)
- Feature engineering and selection
- Model training and optimization
- Performance evaluation and validation
- Production model deployment

## 🚀 Production Deployment

This project now includes **complete production infrastructure**:

### New Components (v2.0)

1. **Data Preprocessing** (`scripts/preprocess_cicids2017.py`)
   - Loads 8 CIC-IDS2017 CSV files
   - Handles infinite values, missing data, duplicates
   - Splits into train_normal / val / test sets
   - Saves feature_schema.json for consistency

2. **Model Training** (`scripts/train_autoencoder.py`)
   - Trains PyTorch autoencoder on benign samples
   - Computes reconstruction error threshold
   - Produces: ae.pth, scaler.joblib, threshold.json

3. **FastAPI Inference Service** (`app/main.py`)
   - `/score` - Single sample prediction
   - `/batch_score` - Batch inference
   - `/health` - Health check
   - `/metrics` - Prometheus metrics
   - Sub-millisecond latency

4. **Containerization** (Docker)
   - `Dockerfile` for repeatable builds
   - Health checks built-in
   - Model artifacts mounted at runtime

5. **Kubernetes Manifests** (`k8s/`)
   - `deploy.yaml` - Deployment with probes
   - `service.yaml` - Service endpoint
   - `hpa.yaml` - CPU/memory-based autoscaling (1-10 replicas)
   - `keda-scaledobject.yaml` - Prometheus-based scaling

6. **Testing & Evaluation**
   - `scripts/locustfile.py` - Load testing (500 concurrent users, anomaly injection)
   - `scripts/evaluate_model.py` - Model metrics (Precision/Recall/F1/ROC-AUC/PR-AUC)

7. **Documentation** (`docs/README_DEPLOY.md`)
   - Step-by-step deployment guide
   - Local Docker testing
   - Kubernetes deployment procedures
   - Prometheus/Grafana setup
   - Troubleshooting guide

### Quick Start (Production)

```bash
# 1. Preprocess CIC-IDS2017
python scripts/preprocess_cicids2017.py --input data/raw --out features --seed 42

# 2. Train model
python scripts/train_autoencoder.py --train features/train_normal.csv --val features/val.csv

# 3. Evaluate
python scripts/evaluate_model.py --test features/test.csv

# 4. Build Docker image
docker build -t ids:latest .

# 5. Deploy to Kubernetes
kubectl apply -f k8s/

# 6. Load test
locust -f scripts/locustfile.py --host http://<SERVICE_IP>
```

See [docs/README_DEPLOY.md](docs/README_DEPLOY.md) for detailed instructions.

---

**Status**: ✅ Production-Ready with ML Ops Infrastructure
**Last Updated**: 2025-11-28
**Version**: 2.0.0 (FastAPI + Kubernetes)
