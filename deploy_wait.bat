@echo off
setlocal enabledelayedexpansion

REM Wait for Docker image to complete building (max 30 minutes)
echo.
echo [*] Waiting for Docker build to complete...
echo [*] This may take 5-15 minutes on first build (PyTorch is 900+ MB)
echo.

for /L %%i in (1,1,180) do (
    docker images siddartha6174/ids:latest >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo [+] Docker image built successfully!
        goto push_image
    )
    if %%i equ 30 echo [*] Still building... (30 seconds elapsed)
    if %%i equ 60 echo [*] Still building... (60 seconds elapsed)
    if %%i equ 120 echo [*] Still building... (120 seconds elapsed)
    timeout /t 1 /nobreak >nul
)

echo [-] Build timeout after 180 seconds
exit /b 1

:push_image
echo.
echo [*] Pushing to Docker Hub (siddartha6174/ids:latest)...
docker push siddartha6174/ids:latest
if !ERRORLEVEL! neq 0 (
    echo [-] Docker push failed
    exit /b 1
)
echo [+] Successfully pushed to Docker Hub

echo.
echo [*] Deploying to Kubernetes...
kubectl apply -f k8s\deploy.yaml
kubectl apply -f k8s\service.yaml
kubectl apply -f k8s\hpa.yaml

echo.
echo [*] Checking deployment status...
kubectl get pods -l app=ids
kubectl get svc ids-svc

echo.
echo [+] Deployment complete!
echo.
echo [*] To access the service locally:
echo    kubectl port-forward svc/ids-svc 8000:80
echo    Then visit: http://localhost:8000/health
echo.
