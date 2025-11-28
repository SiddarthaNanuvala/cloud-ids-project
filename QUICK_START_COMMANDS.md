# QUICK START - Copy & Paste Commands

## ⚠️ FIRST: Restart Docker Desktop

1. Search **Docker Desktop** in Windows Start Menu
2. **Quit** it (right-click system tray icon → Quit)
3. Wait 10 seconds
4. **Reopen** Docker Desktop
5. Wait ~2 minutes for startup

Verify: `docker ps` (should work without errors)

---

## Option 1: Automated Deployment (Recommended)

```powershell
cd C:\Users\sidhu\Projects\cloud-ids-project

# Windows batch script
.\deploy.bat all
```

This runs everything automatically:
- ✅ Docker login
- ✅ Docker build
- ✅ Docker push
- ✅ Kubernetes deploy

**Estimated time:** 10-15 minutes

---

## Option 2: Step-by-Step Manual Commands

### 1. Login to Docker Hub
```powershell
docker login -u siddartha6174
# Paste your Docker Hub password/token when prompted
```

### 2. Navigate to project
```powershell
cd C:\Users\sidhu\Projects\cloud-ids-project
```

### 3. Build Docker image
```powershell
docker build -t siddartha6174/ids:latest .
```
⏱️ First time takes 7-10 minutes (downloading base image + pip packages)

### 4. Push to Docker Hub
```powershell
docker push siddartha6174/ids:latest
```

### 5. Deploy to Kubernetes
```powershell
kubectl apply -f k8s/deploy.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

---

## Option 3: Python Script (Cross-Platform)

```powershell
cd C:\Users\sidhu\Projects\cloud-ids-project
python deploy.py all
```

---

## Verify Deployment

```powershell
# Check pods running
kubectl get pods

# Check service
kubectl get svc ids-svc

# View logs
kubectl logs -l app=ids --tail=20

# Test API locally
kubectl port-forward svc/ids-svc 8000:80
# In another terminal:
curl http://localhost:8000/health
```

---

## Troubleshooting

**Q: "Docker Desktop is unable to start"**  
A: 
```powershell
# Fully kill Docker
wsl --list
wsl --unregister docker_data
# Then restart Docker Desktop from Start Menu
```

**Q: "unauthorized" when pushing**  
A: 
```powershell
docker logout
docker login -u siddartha6174
# Re-enter credentials
```

**Q: "No such file or directory" for Dockerfile**  
A: Make sure you're in the project root
```powershell
pwd  # Should show: C:\Users\sidhu\Projects\cloud-ids-project
ls Dockerfile  # Should exist
```

**Q: Kubernetes pods not starting**  
A:
```powershell
kubectl describe pod <pod-name>  # Shows why pod failed
kubectl get events  # Shows all cluster events
```

---

## What Gets Deployed

| Component | Details |
|-----------|---------|
| **Image** | `siddartha6174/ids:latest` on Docker Hub |
| **Pods** | 1-10 replicas (auto-scales by CPU/Memory) |
| **Service** | Port 80 → Container 8000 |
| **Endpoints** | /score, /batch_score, /health, /metrics, /model_info |
| **Metrics** | Prometheus format on /metrics endpoint |

---

## Next: Load Testing & Monitoring

After deployment is running:

```powershell
# Test with Locust (500 concurrent users)
python -m locust -f scripts/locustfile.py --host http://localhost:8000

# View Kubernetes metrics
kubectl top nodes
kubectl top pods
```

---

**Choose an option above and copy the commands into PowerShell!**

Need help? See `DOCKER_DEPLOYMENT_GUIDE.md` for detailed explanations.
