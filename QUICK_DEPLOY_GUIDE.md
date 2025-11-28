# 🚀 Quick Deployment Guide - For You

**Docker Hub ID:** `siddartha6174`  
**Image:** `siddartha6174/ids:latest`  
**Date:** November 28, 2025

---

## 📋 Pre-Deployment Checklist

Before running deployment, ensure you have:

- ✅ CIC-IDS2017 CSVs in `data/raw/` (8 files)
- ✅ Docker Desktop installed and running
- ✅ kubectl installed (if using Kubernetes)
- ✅ Docker Hub account (siddartha6174)
- ✅ Python 3.10+ with dependencies installed

---

## 🚀 Three Ways to Deploy

### Option 1: PowerShell (Recommended for Windows)

```powershell
# Build Docker image
.\deploy.bat build

# Login and push to Docker Hub
.\deploy.bat push

# Deploy to Kubernetes
.\deploy.bat deploy

# Or run everything at once
.\deploy.bat all
```

### Option 2: Python Script (Cross-platform)

```bash
# Build Docker image
python deploy.py build

# Login and push to Docker Hub
python deploy.py push

# Deploy to Kubernetes
python deploy.py deploy

# Or run everything at once
python deploy.py all
```

### Option 3: Manual Commands

```bash
# 1. Preprocess data
python scripts/preprocess_cicids2017.py --input data/raw --out features

# 2. Train model
python scripts/train_autoencoder.py --train features/train_normal.csv --val features/val.csv --outdir model

# 3. Evaluate model
python scripts/evaluate_model.py --test features/test.csv --model_dir model

# 4. Build Docker image
docker build -t siddartha6174/ids:latest .

# 5. Login to Docker Hub
docker login -u siddartha6174

# 6. Push to Docker Hub
docker push siddartha6174/ids:latest

# 7. Deploy to Kubernetes
kubectl apply -f k8s/deploy.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

---

## 🐳 Docker Desktop - Quick Start

### Step 1: Verify Docker is Running
```powershell
docker --version
docker ps
```

### Step 2: Login to Docker Hub
```powershell
docker login -u siddartha6174
# Enter your Docker Hub personal access token or password when prompted
```

### Step 3: Build & Push

**Option A: Build Locally (No K8s)**
```powershell
# Build
docker build -t siddartha6174/ids:latest .

# Run locally
docker run -it -v ${PWD}/model:/app/model -p 8000:8000 siddartha6174/ids:latest

# In another terminal, test
curl -X POST http://localhost:8000/score `
  -H "Content-Type: application/json" `
  -d '{"features": [0.1, 0.2, 0.3, ...]}'
```

**Option B: Push to Docker Hub**
```powershell
# Build
docker build -t siddartha6174/ids:latest .

# Push
docker push siddartha6174/ids:latest

# Check on Docker Hub: https://hub.docker.com/r/siddartha6174/ids
```

---

## ☸️ Kubernetes Deployment (After Push)

### Step 1: Verify Kubectl
```bash
kubectl cluster-info
kubectl get nodes
```

### Step 2: Deploy
```bash
# All three manifests
kubectl apply -f k8s/deploy.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

### Step 3: Verify Deployment
```bash
# Check pods
kubectl get pods -l app=ids

# Check service
kubectl get svc ids-svc

# Check HPA
kubectl get hpa ids-hpa

# View logs
kubectl logs -f deployment/ids-service
```

### Step 4: Test Service
```bash
# Port-forward
kubectl port-forward svc/ids-svc 8080:80

# In another terminal, test
curl -X POST http://localhost:8080/score \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, ...]}'
```

---

## 🔍 Troubleshooting

### Docker Login Failed
```powershell
# Clear Docker credentials and retry
docker logout
docker login -u siddartha6174
```

### Image Won't Build
```powershell
# Check if model files exist
dir model/

# If missing, train first:
python scripts/train_autoencoder.py --train features/train_normal.csv --val features/val.csv

# Try building again
docker build -t siddartha6174/ids:latest .
```

### Push Fails
```powershell
# Check image exists
docker images | grep siddartha6174/ids

# Try pushing with explicit tag
docker push siddartha6174/ids:latest

# If still fails, rebuild
docker build --no-cache -t siddartha6174/ids:latest .
docker push siddartha6174/ids:latest
```

### Kubernetes Pods Not Starting
```bash
# Check pod events
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>

# Verify image exists
docker images | grep siddartha6174/ids

# Check if image pull policy needs changing
kubectl set image deployment/ids-service ids=siddartha6174/ids:latest
```

---

## ✅ Success Indicators

### Docker Build Success ✓
```
Successfully built abc123def456
Successfully tagged siddartha6174/ids:latest
```

### Docker Push Success ✓
```
The push refers to repository [docker.io/siddartha6174/ids]
latest: digest: sha256:abc123... size: 1234
```

### Kubernetes Deployment Success ✓
```bash
$ kubectl get pods -l app=ids
NAME                           READY   STATUS    RESTARTS   AGE
ids-service-abc123def456-xyz   1/1     Running   0          10s
```

### Service Test Success ✓
```bash
$ curl -X POST http://localhost:8080/score ...
{
  "reconstruction_error": 0.0234,
  "anomaly": false,
  "threshold": 0.0456,
  "latency_seconds": 0.0008
}
```

---

## 📊 Next Steps After Deployment

1. **Monitor:** `kubectl logs -f deployment/ids-service`
2. **Scale:** `kubectl scale deployment ids-service --replicas=3`
3. **Load Test:** `python -m locust -f scripts/locustfile.py --host http://localhost:8080`
4. **View Metrics:** `curl http://localhost:8000/metrics`
5. **Update:** `kubectl set image deployment/ids-service ids=siddartha6174/ids:v2.0`

---

## 🎯 Your Docker Hub Repository

Once deployed, your image will be at:
- **URL:** https://hub.docker.com/r/siddartha6174/ids
- **Pull Command:** `docker pull siddartha6174/ids:latest`
- **Full Image:** `siddartha6174/ids:latest`

---

## 📞 Support

- **Docker Hub Issues:** Check https://docs.docker.com/docker-hub/
- **Kubernetes Issues:** See `docs/README_DEPLOY.md`
- **Model Issues:** See `.github/copilot-instructions.md`

---

**Created:** November 28, 2025  
**Status:** Ready to Deploy  
**Next:** Run `deploy.bat build` or `python deploy.py build`
