# Deployment & Production Setup Guide

This guide walks you through preprocessing data, training the model, containerizing, and deploying to Kubernetes.

## Prerequisites

- Python 3.10+
- Docker & Docker Compose (for local testing)
- Kubernetes cluster (kind, minikube, or managed service)
- kubectl configured
- (Optional) Prometheus & Grafana for monitoring

## Step 1: Preprocess CIC-IDS2017 Data

Place your CIC-IDS2017 CSV files in the `data/raw/` directory.

```bash
# Create directories
mkdir -p data/raw features model

# Run preprocessing
python scripts/preprocess_cicids2017.py \
  --input data/raw \
  --out features \
  --seed 42
```

**Outputs:**
- `features/train_normal.csv` - Benign samples for training (60%)
- `features/val.csv` - Mixed validation set (20% benign + sample of attacks)
- `features/test.csv` - Mixed test set (20% benign + all attacks)
- `features/feature_schema.json` - Feature column names

## Step 2: Train Autoencoder Model

```bash
# Train on normalized benign data, validate on mixed set
python scripts/train_autoencoder.py \
  --train features/train_normal.csv \
  --val features/val.csv \
  --outdir model \
  --epochs 60 \
  --device cpu
```

**Outputs:**
- `model/ae.pth` - Trained autoencoder weights
- `model/scaler.joblib` - StandardScaler for feature normalization
- `model/threshold.json` - Reconstruction error threshold (99th percentile on validation benign)

## Step 3: Evaluate Model (Optional)

```bash
python scripts/evaluate_model.py \
  --test features/test.csv \
  --model_dir model
```

This generates detailed metrics:
- Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion matrix
- Saves results to `model/evaluation_results.json`

## Step 4: Build Docker Image

```bash
# Build locally
docker build -t ids:latest .

# Or with a tag for your registry
docker build -t <YOUR_REGISTRY>/ids:v1.0 .
docker push <YOUR_REGISTRY>/ids:v1.0
```

### Local Testing (Docker)

```bash
# Run container with mounted model
docker run -it \
  -v $(pwd)/model:/app/model \
  -p 8000:8000 \
  ids:latest

# In another terminal, test the service
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2]}'

# Check health
curl http://localhost:8000/health

# Get metrics (Prometheus format)
curl http://localhost:8000/metrics
```

## Step 5: Deploy to Kubernetes

### Single deployment (HPA-based auto-scaling)

```bash
# Create deployment
kubectl apply -f k8s/deploy.yaml

# Create service
kubectl apply -f k8s/service.yaml

# Enable autoscaling (requires metrics-server)
kubectl apply -f k8s/hpa.yaml

# Check deployment status
kubectl get deployment ids-service
kubectl get pods -l app=ids
kubectl get svc ids-svc
```

### Update image in deployment (after build/push)

```bash
kubectl set image deployment/ids-service \
  ids=<YOUR_REGISTRY>/ids:v1.0 \
  --record
```

### Port-forward for local testing

```bash
kubectl port-forward svc/ids-svc 8080:80

# Test
curl -X POST http://localhost:8080/score \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ...]}'
```

## Step 6: Install Prometheus & Grafana (Optional)

```bash
# Add Helm repos
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --create-namespace

# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set adminPassword=admin

# Access Grafana
kubectl port-forward -n monitoring svc/grafana 3000:80
# Visit http://localhost:3000 (admin/admin)

# Add Prometheus as datasource:
# URL: http://prometheus-server.monitoring.svc.cluster.local
```

## Step 7: Load Testing with Locust

Generate sample features first (optional):

```bash
# Extract a sample from val.csv
head -2 features/val.csv | tail -1 > features/sample_feature.json
# Convert to JSON array manually or use pandas
```

Run load test:

```bash
# Headless mode: 500 users, 50 spawn rate, 5 min duration
locust -f scripts/locustfile.py \
  --host http://<SERVICE_IP_OR_INGRESS> \
  --headless \
  -u 500 \
  -r 50 \
  --run-time 5m

# Or with web UI (default: http://localhost:8089)
locust -f scripts/locustfile.py \
  --host http://<SERVICE_IP_OR_INGRESS>
```

The test includes:
- 80% single-sample scoring requests
- 20% batch scoring (10-100 samples per batch)
- 5% health checks
- Anomaly injection (5% of samples perturbed)

## Step 8: Monitor & Scale

Watch HPA scaling in real-time:

```bash
# Watch metrics and replica count
kubectl get hpa ids-hpa --watch

# Check pod metrics
kubectl top pods -l app=ids

# Check resource usage
kubectl describe hpa ids-hpa
```

### KEDA-based Scaling (Advanced)

Install KEDA:

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda --namespace keda --create-namespace
```

Deploy KEDA ScaledObject:

```bash
kubectl apply -f k8s/keda-scaledobject.yaml
```

This scales on Prometheus metrics:
- HTTP request rate (requests/sec)
- Anomaly detection rate (anomalies/sec)

## Troubleshooting

### Model artifacts not loading

```bash
# Check if model files exist in pod
kubectl exec -it <pod-name> -- ls -la /app/model

# Check logs for errors
kubectl logs <pod-name>
```

### Service not responding

```bash
# Check endpoint
kubectl get endpoints ids-svc

# Test connectivity
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://ids-svc/health
```

### Metrics not appearing in Prometheus

```bash
# Port-forward to Prometheus
kubectl port-forward -n monitoring svc/prometheus-server 9090:80

# Visit http://localhost:9090
# Query: http_requests_total
# Check Targets: http://localhost:9090/targets
```

### Out of memory

Increase resource limits in `k8s/deploy.yaml`:

```yaml
resources:
  requests:
    memory: "1Gi"
  limits:
    memory: "2Gi"
```

## Production Checklist

- [ ] Data preprocessing complete and validated
- [ ] Model trained and evaluation metrics acceptable
- [ ] Docker image built and pushed to registry
- [ ] Kubernetes deployment verified (pods running)
- [ ] Service endpoint accessible
- [ ] Health check responding
- [ ] Metrics endpoint accessible
- [ ] Load test passing
- [ ] Prometheus scraping metrics
- [ ] Grafana dashboards visible
- [ ] Autoscaling configured and tested
- [ ] Resource requests/limits set appropriately
- [ ] Monitoring alerts configured
- [ ] Backup/restore procedures documented
- [ ] Log aggregation configured (if needed)

## Next Steps

1. **Monitor** production traffic and adjust threshold if needed
2. **Tune** HPA thresholds based on actual load patterns
3. **Implement** feedback loop: collect mispredictions, retrain model
4. **Scale** to multiple regions/clusters as needed
5. **Add** explainability: SHAP/LIME for model interpretability

---

For support or issues, check logs with:
```bash
kubectl logs -f deployment/ids-service --tail=100
```
