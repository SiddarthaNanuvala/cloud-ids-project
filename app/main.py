#!/usr/bin/env python3
"""
FastAPI inference service for anomaly detection.
Expects model artifacts:
  - /app/model/ae.pth
  - /app/model/scaler.joblib
  - /app/model/threshold.json
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
import os
import json
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
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
print(f"[*] Using MODEL_DIR={MODEL_DIR}")

# Load artifacts
th_path = os.path.join(MODEL_DIR, "threshold.json")
scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
model_path = os.path.join(MODEL_DIR, "ae.pth")

TH = {"threshold": None, "feature_columns": None}
THRESHOLD = None
FEATURE_COLUMNS = None

try:
    with open(th_path, "r") as f:
        TH = json.load(f)
    THRESHOLD = TH.get("threshold", None)
    FEATURE_COLUMNS = TH.get("feature_columns", None)
    print(f"[+] Loaded threshold config: {len(FEATURE_COLUMNS)} features, threshold={THRESHOLD:.6f}")
except Exception as e:
    print(f"[!] Error loading threshold: {e}")

scaler = None
try:
    scaler = joblib.load(scaler_path)
    print(f"[+] Loaded scaler")
except Exception as e:
    print(f"[!] Error loading scaler: {e}")

# ============================================================================
# Model definition
# ============================================================================
class AE(nn.Module):
    """Autoencoder matching training architecture"""
    def __init__(self, n):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n, 64), 
            nn.ReLU(), 
            nn.Linear(64, 16), 
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(16, 64), 
            nn.ReLU(), 
            nn.Linear(64, n)
        )
    
    def forward(self, x):
        return self.dec(self.enc(x))


# ============================================================================
# Load model
# ============================================================================
device = "cpu"
model = None
try:
    if os.path.exists(model_path) and FEATURE_COLUMNS is not None:
        model = AE(len(FEATURE_COLUMNS)).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"[+] Loaded model with {len(FEATURE_COLUMNS)} features")
    else:
        print(f"[!] Model or feature columns not found")
except Exception as e:
    print(f"[!] Error loading model: {e}")

# ============================================================================
# FastAPI app
# ============================================================================
app = FastAPI(
    title="IDS Scoring Service",
    description="Real-time network anomaly detection via autoencoder",
    version="1.0.0"
)

# ============================================================================
# Request/Response schemas
# ============================================================================
class Sample(BaseModel):
    """Single sample with feature vector"""
    features: List[float]
    
    class Config:
        schema_extra = {
            "example": {"features": [0.1, 0.2, 0.3] * 7 + [0.1]}  # 22 features
        }


class BatchSamples(BaseModel):
    """Batch of samples"""
    features: List[List[float]]
    
    class Config:
        schema_extra = {
            "example": {"features": [[0.1, 0.2, 0.3] * 7 + [0.1]] * 10}
        }


# ============================================================================
# Endpoints
# ============================================================================
@app.get("/health")
def health():
    """Health check endpoint"""
    REQS.labels(endpoint="/health").inc()
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    REQS.labels(endpoint="/metrics").inc()
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/score")
def score(sample: Sample):
    """Score a single sample for anomaly."""
    REQS.labels(endpoint="/score").inc()
    
    start = time.time()
    try:
        if model is None or scaler is None or THRESHOLD is None:
            raise HTTPException(status_code=500, detail="Model not ready")
        
        x = np.array(sample.features).reshape(1, -1)
        
        # Validate feature count
        if FEATURE_COLUMNS and x.shape[1] != len(FEATURE_COLUMNS):
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {len(FEATURE_COLUMNS)} features, got {x.shape[1]}"
            )
        
        # Preprocess
        xs = scaler.transform(x)
        
        # Inference
        with torch.no_grad():
            inp = torch.from_numpy(xs.astype(np.float32)).to(device)
            recon = model(inp).cpu().numpy()
        
        # Compute reconstruction error
        recon_err = float(((xs - recon) ** 2).mean())
        is_anom = recon_err > THRESHOLD
        
        # Update metrics
        if is_anom:
            ANOMALIES.inc()
        else:
            NORMAL.inc()
        
        elapsed = time.time() - start
        LAT.labels(endpoint="/score").observe(elapsed)
        
        return {
            "reconstruction_error": recon_err,
            "anomaly": bool(is_anom),
            "threshold": THRESHOLD,
            "latency_seconds": elapsed
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_score")
def batch_score(batch: BatchSamples):
    """Score a batch of samples."""
    REQS.labels(endpoint="/batch_score").inc()
    
    start = time.time()
    try:
        if model is None or scaler is None or THRESHOLD is None:
            raise HTTPException(status_code=500, detail="Model not ready")
        
        X = np.array(batch.features)
        
        # Validate feature count
        if FEATURE_COLUMNS and X.shape[1] != len(FEATURE_COLUMNS):
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {len(FEATURE_COLUMNS)} features per sample, got {X.shape[1]}"
            )
        
        # Preprocess
        Xs = scaler.transform(X)
        
        # Inference
        with torch.no_grad():
            inp = torch.from_numpy(Xs.astype(np.float32)).to(device)
            recon = model(inp).cpu().numpy()
        
        # Compute reconstruction errors
        errs = ((Xs - recon) ** 2).mean(axis=1)
        
        # Build results
        results = []
        anom_count = 0
        for e in errs:
            is_anom = e > THRESHOLD
            results.append({
                "reconstruction_error": float(e),
                "anomaly": bool(is_anom)
            })
            if is_anom:
                anom_count += 1
                ANOMALIES.inc()
            else:
                NORMAL.inc()
        
        elapsed = time.time() - start
        LAT.labels(endpoint="/batch_score").observe(elapsed)
        
        return {
            "results": results,
            "anomaly_count": anom_count,
            "total_samples": len(X),
            "anomaly_rate": float(anom_count / len(X)),
            "latency_seconds": elapsed
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
def model_info():
    """Get model metadata."""
    REQS.labels(endpoint="/model_info").inc()
    return {
        "feature_count": len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 0,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": THRESHOLD,
        "device": device
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
