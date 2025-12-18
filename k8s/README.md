# Kubernetes Deployment

## Quick Deploy

```bash
kubectl apply -f k8s/deploy.yaml k8s/service.yaml k8s/hpa.yaml
```

## Resources

### 1. Deployment (`deploy.yaml`)
- **Image:** `siddartha6174/ids:latest`
- **Replicas:** 1 (managed by HPA)
- **Ports:** 8000 (container), 80 (service)
- **Resources:**
  - Request: CPU 250m, Memory 512Mi
  - Limit: CPU 1000m, Memory 1Gi

### 2. Service (`service.yaml`)
- **Type:** ClusterIP
- **Port:** 80 → 8000
- **Selector:** app=ids

### 3. HPA (`hpa.yaml`)
Auto-scaling: 1-10 pods based on metrics
- CPU Target: 60%
- Memory Target: 70%

### 4. KEDA (Optional)
`keda-scaledobject.yaml` - Advanced scaling based on Prometheus metrics
- Min: 1 pod, Max: 20 pods
- Triggers: Request rate, anomaly rate

## Verify Deployment

```bash
# Check pods
kubectl get pods -l app=ids

# Check service
kubectl get svc ids-svc

# Check auto-scaling
kubectl get hpa ids-hpa

# View logs
kubectl logs -l app=ids --tail=50

# Port forward for testing
kubectl port-forward svc/ids-svc 8000:80

# Test API
curl http://localhost:8000/health
```

## Prometheus Metrics

Metrics available at `/metrics`:
- `http_requests_total` - Total requests
- `inference_latency_seconds` - Prediction latency
- `anomalies_detected_total` - Anomalies found
- `normal_samples_total` - Normal samples
- `batch_prediction_sizes` - Batch sizes
- `model_load_duration` - Model load time
- `prediction_errors_total` - Errors

## Troubleshooting

**Pods not starting?**
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

**Image pull error?**
```bash
# Verify image exists locally
docker images siddartha6174/ids

# If missing, build and push
docker build -t siddartha6174/ids:latest .
docker push siddartha6174/ids:latest
```

**Service not accessible?**
```bash
kubectl get svc
kubectl port-forward svc/ids-svc 8000:80
```
