# Complete Docker & Kubernetes Deployment Guide

## Current Status
✅ Code: All production code ready  
✅ Configuration: k8s/deploy.yaml updated with `siddartha6174/ids:latest`  
✅ Deployment Scripts: deploy.py and deploy.bat created  
❌ Docker Desktop: Needs restart (unmounting error)  
⏳ Next: Docker login → Build → Push → Deploy

---

## Step 1: Restart Docker Desktop

**On Windows:**
1. Open **Docker Desktop** app (search in Start Menu)
2. If it's running, click the whale icon in system tray → **Quit Docker Desktop**
3. Wait 10 seconds
4. Restart Docker Desktop app
5. Wait for it to fully start (~1-2 minutes) - you'll see "Docker Desktop is running" in system tray

**Verify it's working:**
```powershell
docker ps
# Should return: CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS   PORTS     NAMES
# (empty list is OK if no containers running)
```

---

## Step 2: Login to Docker Hub

```powershell
docker login -u siddartha6174
```

**Prompts:**
- **Username:** siddartha6174 (pre-filled)
- **Password/Token:** Enter your Docker Hub password or Personal Access Token
  - If using PAT: Visit https://app.docker.com/settings/personal-access-tokens
  - Generate token with read/write permissions, copy it

**Success response:**
```
Login Succeeded
```

---

## Step 3: Build Docker Image

Navigate to project root and build:

```powershell
cd C:\Users\sidhu\Projects\cloud-ids-project
docker build -t siddartha6174/ids:latest .
```

**Expected output:**
```
[+] Building 450s (10/11)
 => [1/7] FROM docker.io/library/python:3.10-slim
 => [2/7] WORKDIR /app
 => [3/7] RUN apt-get update && apt-get install build-essential
 => [4/7] COPY app/ ./app
 => [5/7] COPY model/ ./model/
 => [6/7] COPY requirements.txt .
 => [7/7] RUN pip install --no-cache-dir -r requirements.txt
```

**Build time:** ~7-10 minutes (first time includes all pip packages)

---

## Step 4: Push to Docker Hub

```powershell
docker push siddartha6174/ids:latest
```

**Expected output:**
```
The push refers to repository [docker.io/siddartha6174/ids]
latest: digest: sha256:abc123def456... size: 2.5GB
```

**On Docker Hub:** Visit https://hub.docker.com/r/siddartha6174/ids
- You'll see `latest` tag with image details
- K8s will now be able to pull from your private registry

---

## Step 5: Verify Image Locally

```powershell
# List your images
docker images siddartha6174/ids

# Test run the image
docker run -it -p 8000:8000 siddartha6174/ids:latest

# In another terminal, test the API
curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

---

## Step 6: Deploy to Kubernetes

**Option A: Using kubectl directly**
```powershell
kubectl apply -f k8s/deploy.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Check deployment status
kubectl get pods
kubectl get svc
kubectl describe pod <pod-name>
```

**Option B: Using PowerShell deployment script**
```powershell
.\deploy.bat deploy
```

**Option C: Using Python deployment script**
```powershell
python deploy.py deploy
```

---

## Troubleshooting

### Docker Desktop won't start
- Check WSL status: `wsl --list --verbose`
- Reinstall WSL2: https://docs.microsoft.com/windows/wsl/install
- Update Windows: Settings → Windows Update → Check for updates

### Build fails with "Input/output error"
- Restart Docker Desktop
- Free up disk space (build needs ~3GB)
- Check Docker logs: Docker Desktop → Preferences → Troubleshoot

### Push fails with "unauthorized"
- Verify login: `docker logout; docker login -u siddartha6174`
- Check Docker Hub password is correct
- If using PAT, ensure token has `read` and `write` permissions

### Kubernetes pods won't start
- Check image exists: `docker images siddartha6174/ids`
- Check pull permissions: `kubectl describe pod <pod-name>` (look for ImagePullBackOff)
- Verify service is running: `kubectl get svc ids-svc`

---

## Complete Automated Option

Once Docker Desktop is running, execute the full pipeline:

```powershell
# Using batch script (Windows)
cd C:\Users\sidhu\Projects\cloud-ids-project
.\deploy.bat all

# OR using Python
python deploy.py all
```

This will:
1. Build Docker image locally
2. Push to Docker Hub
3. Deploy to Kubernetes

---

## Quick Health Check

```powershell
# Confirm deployment
kubectl get pods,svc,hpa

# Check logs
kubectl logs -l app=ids --tail=50

# Port forward to test locally
kubectl port-forward svc/ids-svc 8000:80

# In another terminal
curl http://localhost:8000/metrics
```

---

**Next Step:** Restart Docker Desktop, then follow the steps above!
