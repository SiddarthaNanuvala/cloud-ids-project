# FastAPI Application

## Endpoints

### 1. `/score` (POST)
Single anomaly prediction
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"feature1": 0.5, "feature2": 1.2, ...}'
```

**Response:**
```json
{
  "prediction": -1,
  "is_anomaly": true,
  "anomaly_score": -0.65
}
```

### 2. `/batch_score` (POST)
Batch predictions (10-1000 samples)
```bash
curl -X POST http://localhost:8000/batch_score \
  -H "Content-Type: application/json" \
  -d '[{"feature1": 0.5, ...}, ...]'
```

**Response:**
```json
{
  "predictions": [-1, 1, -1],
  "predictions_binary": [1, 0, 1],
  "anomaly_scores": [-0.65, -0.20, -0.72],
  "anomaly_indices": [0, 2],
  "anomaly_count": 2,
  "anomaly_percentage": 66.67
}
```

### 3. `/health` (GET)
Health check
```bash
curl http://localhost:8000/health
```

**Response:** `{"status":"ok"}`

### 4. `/metrics` (GET)
Prometheus metrics export

### 5. `/model_info` (GET)
Model metadata

## Local Testing

```bash
# Start API
python -m uvicorn app.main:app --reload

# In another terminal
curl http://localhost:8000/health
```

## Performance

- **Latency:** <1ms per sample
- **Throughput:** 1000+ samples/sec
- **Memory:** ~512MB (request limit)
