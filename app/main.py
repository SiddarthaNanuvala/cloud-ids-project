#!/usr/bin/env python3
"""
FastAPI inference service for network anomaly detection using Isolation Forest model.
This service loads a pre-trained Isolation Forest model and provides real-time predictions.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import joblib
from typing import List
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse
import time

# ============================================================================
# Prometheus metrics
# ============================================================================
REQS = Counter("http_requests_total", "Total HTTP requests", ["endpoint"])
LAT = Histogram("inference_latency_seconds", "Inference latency seconds", ["endpoint"])
ANOMALIES = Counter("anomalies_detected_total", "Total anomalies detected")
NORMAL = Counter("normal_samples_total", "Total normal samples")

# ============================================================================
# Configuration
# ============================================================================
# Find model.pkl in project root (first) or /app (Docker)
model_search_paths = [
    "../../model.pkl",  # From app/ running locally
    "../model.pkl",  # Alternative from app/
    "./model.pkl",  # From project root
    "/app/model.pkl",  # Docker path
    "/models/model.pkl",  # Docker volume path
]

MODEL_PATH = None
for path in model_search_paths:
    abs_path = os.path.abspath(path) if not path.startswith("/") else path
    if os.path.exists(abs_path):
        MODEL_PATH = abs_path
        print(f"[+] Found model at: {MODEL_PATH}")
        break

if MODEL_PATH is None:
    print(f"[!] Warning: Could not find model.pkl in any search paths")

# ============================================================================
# Load model
# ============================================================================
worker = None
try:
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        model_package = joblib.load(MODEL_PATH)
        iso_forest = model_package['model']
        scaler = model_package['scaler']
        feature_names = model_package['feature_names']
        
        print(f"[+] Loaded Isolation Forest model successfully")
        print(f"    - Features: {len(feature_names)}")
        print(f"    - Model type: {type(iso_forest).__name__}")
        
        # Create a simple worker object
        class ModelWorker:
            def __init__(self, model, scaler, features):
                self.model = model
                self.scaler = scaler
                self.features = features
            
            def predict(self, X):
                """Predict on batch"""
                if X.shape[1] != len(self.features):
                    raise ValueError(f"Expected {len(self.features)} features, got {X.shape[1]}")
                X_scaled = self.scaler.transform(X)
                predictions = self.model.predict(X_scaled)  # -1 = anomaly, 1 = normal
                scores = self.model.score_samples(X_scaled)
                return predictions, scores
            
            def predict_single(self, x):
                """Predict on single sample"""
                if len(x) != len(self.features):
                    raise ValueError(f"Expected {len(self.features)} features, got {len(x)}")
                x_array = np.array(x).reshape(1, -1)
                preds, scores = self.predict(x_array)
                return preds[0], scores[0]
        
        worker = ModelWorker(iso_forest, scaler, feature_names)
    else:
        print(f"[!] Model file not found at: {MODEL_PATH}")
except Exception as e:
    print(f"[!] Error loading model: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# FastAPI app
# ============================================================================
app = FastAPI(
    title="Cloud-IDS Anomaly Detection Service",
    description="Real-time network anomaly detection using Isolation Forest",
    version="1.0.0"
)

# ============================================================================
# Request/Response schemas
# ============================================================================
class SampleRequest(BaseModel):
    """Single sample with feature vector"""
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {"features": [100.0, 5.2, 1.3, 8.9, 2.1] + [0.0] * 67}
        }


class BatchRequest(BaseModel):
    """Batch of samples"""
    features: List[List[float]]
    
    class Config:
        json_schema_extra = {
            "example": {"features": [[100.0, 5.2, 1.3, 8.9, 2.1] + [0.0] * 67] * 3}
        }


# ============================================================================
# Endpoints
# ============================================================================
@app.get("/health")
def health():
    """Health check endpoint"""
    REQS.labels(endpoint="/health").inc()
    return {"status": "ok", "model_ready": worker is not None}


@app.get("/model_info")
def model_info():
    """Get model information and feature names"""
    REQS.labels(endpoint="/model_info").inc()
    
    if worker is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    return {
        "model_type": "Isolation Forest",
        "n_features": len(worker.features),
        "features": worker.features,
        "version": "1.0.0"
    }


@app.post("/score")
def score(request: SampleRequest):
    """
    Score a single sample for anomaly.
    Returns:
        - prediction: -1 for anomaly, 1 for normal
        - is_anomaly: boolean flag
        - anomaly_score: raw anomaly score (lower = more anomalous)
    """
    REQS.labels(endpoint="/score").inc()
    
    start = time.time()
    try:
        if worker is None:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        # Predict
        prediction, score_value = worker.predict_single(request.features)
        is_anomaly = prediction == -1
        
        # Update metrics
        if is_anomaly:
            ANOMALIES.inc()
        else:
            NORMAL.inc()
        
        elapsed = time.time() - start
        LAT.labels(endpoint="/score").observe(elapsed)
        
        return {
            "prediction": int(prediction),  # -1 = anomaly, 1 = normal
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(score_value),
            "latency_seconds": float(elapsed)
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_score")
def batch_score(request: BatchRequest):
    """
    Score a batch of samples.
    Returns predictions and statistics for all samples.
    """
    REQS.labels(endpoint="/batch_score").inc()
    
    start = time.time()
    try:
        if worker is None:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        X = np.array(request.features, dtype=np.float64)
        
        # Predict
        predictions, scores = worker.predict(X)
        
        # Build results
        is_anomalies = predictions == -1
        results = []
        for pred, score_val in zip(predictions, scores):
            results.append({
                "prediction": int(pred),  # -1 = anomaly, 1 = normal
                "is_anomaly": bool(pred == -1),
                "anomaly_score": float(score_val)
            })
        
        # Count anomalies
        n_anomalies = np.sum(is_anomalies)
        ANOMALIES.inc(n_anomalies)
        NORMAL.inc(len(predictions) - n_anomalies)
        
        elapsed = time.time() - start
        LAT.labels(endpoint="/batch_score").observe(elapsed)
        
        return {
            "predictions": predictions.tolist(),  # List of -1 or 1
            "predictions_binary": (predictions == -1).astype(int).tolist(),  # List of 0 or 1
            "anomaly_scores": scores.tolist(),
            "results": results,
            "anomaly_count": int(n_anomalies),
            "anomaly_percentage": float(100.0 * n_anomalies / len(predictions)),
            "total_samples": len(predictions),
            "latency_seconds": float(elapsed)
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics_endpoint():
    """Prometheus metrics endpoint"""
    REQS.labels(endpoint="/metrics").inc()
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# Root endpoint
# ============================================================================
@app.get("/")
def root():
    """Root endpoint with service info"""
    return {
        "service": "Cloud-IDS Anomaly Detection",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "GET /model_info": "Model information and features",
            "POST /score": "Score single sample",
            "POST /batch_score": "Score batch of samples",
            "GET /metrics": "Prometheus metrics",
            "GET /docs": "Interactive API documentation (Swagger UI)"
        },
        "model_ready": worker is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
