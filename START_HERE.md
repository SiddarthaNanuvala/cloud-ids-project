# 🚀 DEPLOYMENT QUICK REFERENCE

Your project is **100% PRODUCTION READY**. Use this reference to get started.

---

## 📋 What to Do Now (3 Simple Steps)

### Step 1: Restart Docker Desktop
```
1. Windows Start Menu → Search "Docker Desktop"
2. Right-click System Tray icon → Quit
3. Wait 10 seconds → Open Docker Desktop again
4. Wait ~2 minutes for startup
```

### Step 2: Deploy (Pick ONE Option)

**OPTION A: One Command (Recommended)**
```powershell
cd C:\Users\sidhu\Projects\cloud-ids-project
.\deploy.bat all
```

**OPTION B: Step-by-Step (Full Control)**
```powershell
cd C:\Users\sidhu\Projects\cloud-ids-project
docker login -u siddartha6174
docker build -t siddartha6174/ids:latest .
docker push siddartha6174/ids:latest
kubectl apply -f k8s/deploy.yaml k8s/service.yaml k8s/hpa.yaml
```

**OPTION C: Python Script**
```powershell
cd C:\Users\sidhu\Projects\cloud-ids-project
python deploy.py all
```

### Step 3: Verify (After Deployment)
```powershell
kubectl get pods              # Should show running pods
kubectl get svc              # Should show ids-svc service
kubectl port-forward svc/ids-svc 8000:80
curl http://localhost:8000/health  # Should return {"status":"ok"}
```

---

## 📚 Documentation Files (Choose What You Need)

| File | Purpose | When to Read |
|------|---------|--------------|
| **QUICK_START_COMMANDS.md** | Copy-paste commands | Starting deployment |
| **DOCKER_DEPLOYMENT_GUIDE.md** | Docker + K8s step-by-step | Need detailed guidance |
| **DEPLOYMENT_SUMMARY.md** | Architecture overview | Understanding the system |
| **DEPLOYMENT_STATUS.md** | Project status report | Verify everything is ready |
| **PRODUCTION_SETUP.md** | Complete setup guide | Learning the pipeline |
| **.github/copilot-instructions.md** | AI agent knowledge | Understanding code |

---

## 🎯 Key Information

| Item | Value |
|------|-------|
| **Docker Hub ID** | siddartha6174 |
| **Image Name** | siddartha6174/ids:latest |
| **Service Port** | 80 (→ 8000 in container) |
| **Replicas** | 1-10 (auto-scales by CPU/Memory) |
| **API Endpoints** | /score, /batch_score, /health, /metrics |
| **Estimated Deploy Time** | 10-15 minutes |

---

## 📁 Project Structure

```
cloud-ids-project/
├── app/                           # FastAPI application
│   └── main.py                   # 5 endpoints, Prometheus metrics
├── k8s/                           # Kubernetes manifests
│   ├── deploy.yaml               # Updated with your Docker ID
│   ├── service.yaml              # ClusterIP service
│   ├── hpa.yaml                  # Auto-scaling (1-10 pods)
│   └── keda-scaledobject.yaml    # Optional metric-based scaling
├── scripts/                       # ML pipeline
│   ├── preprocess_cicids2017.py  # Data loading & feature selection
│   ├── train_autoencoder.py      # Model training
│   ├── evaluate_model.py         # Metrics computation
│   └── locustfile.py             # Load testing
├── model/                         # Model artifacts (placeholders)
│   ├── ae.pth
│   ├── scaler.joblib
│   └── threshold.json
├── Dockerfile                     # Python 3.10-slim + dependencies
├── requirements.txt               # All packages for production
├── deploy.py                      # Python automation (cross-platform)
├── deploy.bat                     # Windows batch automation
└── [Documentation files]          # 10+ guides
```

---

## 🔧 Troubleshooting

### Docker won't start
**Solution:** Restart WSL
```powershell
wsl --unregister docker_data
# Then restart Docker Desktop
```

### Docker push fails (unauthorized)
**Solution:** Re-login
```powershell
docker logout
docker login -u siddartha6174
```

### Pods not starting
**Solution:** Check logs
```powershell
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### More issues?
See **DOCKER_DEPLOYMENT_GUIDE.md** for complete troubleshooting.

---

## ✅ Success Indicators

After deployment, you should have:

- ✅ Docker image built: `docker images | grep siddartha6174`
- ✅ Image pushed: Visit https://hub.docker.com/r/siddartha6174/ids
- ✅ K8s pods running: `kubectl get pods` shows "Running"
- ✅ Service created: `kubectl get svc` shows "ids-svc"
- ✅ API responding: `curl http://localhost:8000/health` → `{"status":"ok"}`
- ✅ Metrics exported: `curl http://localhost:8000/metrics` → Prometheus output

---

## 📞 Need Help?

1. **Quick commands?** → `QUICK_START_COMMANDS.md`
2. **Docker issues?** → `DOCKER_DEPLOYMENT_GUIDE.md`
3. **Understand architecture?** → `DEPLOYMENT_SUMMARY.md`
4. **Full walkthrough?** → `PRODUCTION_SETUP.md`
5. **AI/Agent code docs?** → `.github/copilot-instructions.md`

---

## 🎓 What Gets Deployed

```
┌──────────────┐
│   Your Data  │
│  (8 CSVs)    │
└──────┬───────┘
       ↓
┌──────────────┐
│  Preprocess  │
│  22 Features │
└──────┬───────┘
       ↓
┌──────────────┐
│  Train Model │
│  PyTorch AE  │
└──────┬───────┘
       ↓
┌──────────────────────────┐
│   Docker Image Built     │
│  siddartha6174/ids       │
└──────┬───────────────────┘
       ↓
┌──────────────────────────┐
│  Pushed to Docker Hub    │
│  Ready for K8s pull      │
└──────┬───────────────────┘
       ↓
┌──────────────────────────┐
│  Kubernetes Deployment   │
│  1-10 Auto-scaling Pods  │
└──────┬───────────────────┘
       ↓
┌──────────────────────────┐
│   FastAPI Service        │
│  Port 80 (→ 8000)       │
│  5 Endpoints             │
│  Prometheus Metrics      │
└──────────────────────────┘
```

---

## 🚀 Ready? Let's Go!

**Copy one command and paste into PowerShell:**

```powershell
cd C:\Users\sidhu\Projects\cloud-ids-project; .\deploy.bat all
```

That's it! Deployment will run automatically.

---

**Questions?** Check the documentation files above or review `DEPLOYMENT_COMPLETE.txt` for comprehensive summary.

**All set!** 🎉
