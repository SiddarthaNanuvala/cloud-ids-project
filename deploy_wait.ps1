#!/usr/bin/env pwsh
# Deploy script - waits for Docker build to complete, then deploys to Kubernetes

Write-Host "⏳ Waiting for Docker build to complete..." -ForegroundColor Yellow

# Check every 10 seconds until image is built
$maxWait = 1800  # 30 minutes max
$elapsed = 0
$checkInterval = 10

while ($elapsed -lt $maxWait) {
    $image = docker images --format "{{.Repository}}:{{.Tag}}" | Select-String "siddartha6174/ids:latest"
    
    if ($image) {
        Write-Host "✅ Docker image built successfully!" -ForegroundColor Green
        break
    }
    
    Write-Host "⏳ Still building... ($elapsed/$maxWait seconds)" -ForegroundColor Yellow
    Start-Sleep -Seconds $checkInterval
    $elapsed += $checkInterval
}

if ($elapsed -ge $maxWait) {
    Write-Host "❌ Build timeout after 30 minutes" -ForegroundColor Red
    exit 1
}

# Image is built, proceed with deployment
Write-Host "`n📦 Pushing to Docker Hub..." -ForegroundColor Cyan
docker push siddartha6174/ids:latest
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Push failed" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Pushed to Docker Hub" -ForegroundColor Green

Write-Host "`n🚀 Deploying to Kubernetes..." -ForegroundColor Cyan
kubectl apply -f k8s/deploy.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

Write-Host "`n📊 Checking deployment status..." -ForegroundColor Cyan
kubectl get pods -l app=ids
kubectl get svc ids-svc

Write-Host "`n✅ Deployment complete!" -ForegroundColor Green
Write-Host "To access the service, run:" -ForegroundColor Yellow
Write-Host "  kubectl port-forward svc/ids-svc 8000:80" -ForegroundColor Cyan
Write-Host "`nThen visit: http://localhost:8000/health" -ForegroundColor Cyan
