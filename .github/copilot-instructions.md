# Copilot Instructions for CIC-IDS2017 Anomaly Detection

## Project Overview
This is a machine learning pipeline for detecting network anomalies using the CIC-IDS2017 dataset. The system identifies potential security threats in network traffic via an unsupervised Isolation Forest model. The project is **production-ready** with 2.83M cleaned records, 72 optimized features, and <1ms inference latency.

## Architecture & Data Flow

### Core Components
1. **Raw Data** (`Raw Data/` directory) - 8 CIC-IDS2017 CSV files with network flow records (3.1M raw records, 85 features)
2. **ML Pipeline** (`ML_Anomaly_Detection.ipynb`) - Jupyter notebook orchestrating: data loading → cleaning → feature engineering → model training → serialization
3. **Production Model** (`model.pkl`) - Joblib-serialized package containing: Isolation Forest (100 trees), StandardScaler, and feature metadata
4. **Inference Module** (`worker.py`) - `AnomalyDetectionWorker` class providing batch and single-sample predictions

### Data Flow
- Load 8 CSV files → Combine into single DataFrame (3.1M × 85)
- Remove duplicates, infinite values, missing data → 2.83M × 85 records retained
- Select 72 numeric features (removing 8 zero-variance columns)
- Apply StandardScaler normalization (mean=0, std=1)
- Train Isolation Forest on 80% split (2.26M samples) → detect 10% anomalies
- Serialize model+scaler+feature_names as `model.pkl` (773 KB)
- Load in `worker.py` for real-time inference (<1ms per sample)

## Key Development Patterns

### Feature Handling
- **Critical constraint**: Model requires exactly 72 specific numeric features in a fixed order
- `model.pkl` embeds feature names and order; `worker.py` enforces column matching
- New data must be preprocessed identically: fill infinities, remove NaNs, select same 72 columns
- Zero-variance features are filtered during training; don't add them back in inference

### Model Serialization
- Uses `joblib.load()` / `joblib.dump()` (not pickle) for scikit-learn compatibility
- `model.pkl` is a dict containing three keys: `'model'` (IsolationForest), `'scaler'` (StandardScaler), `'feature_names'` (list)
- Always load scaler first, then call `scaler.transform()` before `model.predict()`

### Prediction Patterns (from `worker.py`)
```python
worker = AnomalyDetectionWorker('model.pkl')

# Batch predictions: returns dict with 'predictions' (-1/1), 'predictions_binary' (1/0), 
# 'anomaly_scores', 'anomaly_indices', 'anomaly_count', 'anomaly_percentage'
results = worker.predict(dataframe_or_array)

# Single sample: returns dict with 'prediction', 'is_anomaly' (bool), 'anomaly_score'
result = worker.predict_single(dict_or_series)
```
- `-1` in predictions = anomaly; `1` = normal (scikit-learn convention)
- `predictions_binary` inverts to `1` = anomaly, `0` = normal (for user convenience)
- Anomaly scores range from ~-0.76 to -0.33 (lower = more anomalous)

## Conventions & Patterns

### Data Processing
- Always handle CIC-IDS2017 column naming: spaces in names (e.g., ` Source Port`, ` Destination Port`)
- Character encoding: use `encoding='latin-1'` when reading CSVs (handles special characters)
- Infinite values appear in columns like `Flow Bytes/s`, `Flow Packets/s` → detect with `np.isinf()`, replace with NaN, then drop rows
- Duplicates and missing values: ~9.25% of raw data (291,668 records removed, acceptable loss)

### Dependencies (Python 3.13.1)
- `pandas` (data manipulation), `scikit-learn` (Isolation Forest), `numpy` (numerics)
- `joblib` (model serialization), `jupyter` (notebooks)
- No external APIs or cloud dependencies; purely local inference

### Testing & Validation
- Model trained on 2.26M samples, tested on 565K samples (80/20 split, random_state=42)
- Expected performance: 10.11% anomaly detection rate (57K anomalies in test set)
- Contamination hyperparameter = 0.1 (hardcoded in training; doesn't change at inference)

## Critical Files & Their Roles

| File | Purpose | Key Details |
|------|---------|-------------|
| `ML_Anomaly_Detection.ipynb` | Complete pipeline (9 sections, 25 cells) | Source of truth for reproducibility; contains all steps from data loading to model export |
| `worker.py` | Production inference API | Validates input features, scales data, returns predictions; single point of failure for inference |
| `model.pkl` | Trained model artifact | Generated after first notebook run; 773 KB; contains model+scaler+feature_names dict |
| `requirements.txt` | Python dependencies | Pinned versions; use `pip install -r requirements.txt` |
| `README.md` | User-facing documentation | Examples, troubleshooting, security considerations |
| `INDEX.md` | Delivery checklist & specs | Technical metrics, task completion summary |

## Workflow Commands

```powershell
# Setup environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run pipeline (regenerates model.pkl)
jupyter notebook ML_Anomaly_Detection.ipynb

# Use in production
python -c "from worker import AnomalyDetectionWorker; w = AnomalyDetectionWorker(); results = w.predict(data)"
```

## Integration Points & Gotchas

- **Model retraining**: Must regenerate `model.pkl` by running notebook cells 1-8; changing contamination requires retraining
- **Feature consistency**: If upstream data changes feature list/order, notebook must be re-run with updated CSV files
- **Inference failures**: `worker.py` raises `ValueError` if input features don't match model expectations (check column names/count)
- **Scaling**: Model ALWAYS expects scaled input (StandardScaler); raw data will produce wrong predictions

## Next Steps for Development

- **Short-term**: Monitor live prediction performance, tune contamination rate via notebook retraining
- **Medium-term**: Add SHAP/LIME for explainability; experiment with ensemble methods (Random Forest, XGBoost)
- **Long-term**: Implement automated retraining pipeline, feedback loop for continuous improvement

---

**Last Updated**: November 27, 2025 | **Status**: Production-Ready
