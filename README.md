# CIC-IDS2017 Anomaly Detection System

## Project Overview
This is a comprehensive machine learning pipeline for detecting network anomalies in the CIC-IDS2017 dataset using an Isolation Forest algorithm. The system is designed to identify potential security threats and attack patterns in network traffic.

## 📋 Project Structure

```
Security Use Cases/
├── ML_Anomaly_Detection.ipynb      # Main Jupyter notebook with full pipeline
├── worker.py                        # Production-ready inference script
├── model.pkl                        # Serialized trained model (joblib format)
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
└── Raw Data/                        # Original CIC-IDS2017 CSV files
    ├── Monday-WorkingHours.pcap_ISCX.csv
    ├── Tuesday-WorkingHours.pcap_ISCX.csv
    ├── Wednesday-workingHours.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
    ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

## 🎯 Core Tasks Completed

### 1. ✅ Data Analysis & Exploration
- Loaded all 8 CIC-IDS2017 CSV files
- Combined datasets into a single unified dataframe
- Analyzed data structure, dimensions, and data types
- Identified missing values and infinite values
- Examined class distributions and dataset statistics

### 2. ✅ Data Cleaning & Preprocessing
- Removed duplicate records
- Handled missing values by removing incomplete rows
- Replaced infinite values with NaN and removed them
- Standardized data types for numeric columns
- Applied StandardScaler normalization to all features

### 3. ✅ Feature Engineering
- Selected numeric features only (non-text)
- Removed zero-variance features
- Scaled all features to mean=0, std=1 for better model performance
- Prepared clean feature matrix ready for training

### 4. ✅ Model Training
- **Algorithm**: Isolation Forest (unsupervised anomaly detection)
- **Hyperparameters**:
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
