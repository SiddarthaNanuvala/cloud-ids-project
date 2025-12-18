@echo off
REM Deploy IDS Service - Windows Batch Script
REM Usage: deploy.bat [build|push|deploy|all]
REM
REM Set your Docker Hub ID here
set DOCKER_HUB_ID=siddartha6174
set IMAGE_NAME=ids
set IMAGE_TAG=latest
set FULL_IMAGE=%DOCKER_HUB_ID%/%IMAGE_NAME%:%IMAGE_TAG%

echo.
echo ==============================================================================
echo CIC-IDS2017 Anomaly Detection - Deployment Script
echo Docker Hub ID: %DOCKER_HUB_ID%
echo Image: %FULL_IMAGE%
echo ==============================================================================
echo.

if "%1"=="" (
    echo Usage: deploy.bat [command]
    echo.
    echo Available commands:
    echo   preprocess  - Run data preprocessing
    echo   train       - Train autoencoder model
    echo   evaluate    - Evaluate model performance
    echo   build       - Build Docker image locally
    echo   push        - Push image to Docker Hub
    echo   deploy      - Deploy to Kubernetes
    echo   all         - Run all steps
    echo.
    echo Examples:
    echo   deploy.bat build
    echo   deploy.bat push
    echo   deploy.bat all
    goto :eof
)

if /i "%1"=="preprocess" (
    echo [*] Running Data Preprocessing...
    python scripts/preprocess_cicids2017.py --input data/raw --out features --seed 42
    if errorlevel 1 (
        echo [!] Preprocessing failed
        exit /b 1
    )
    echo [+] Preprocessing complete
    goto :eof
)

if /i "%1"=="train" (
    echo [*] Training Model...
    python scripts/train_autoencoder.py --train features/train_normal.csv --val features/val.csv --outdir model --epochs 60
    if errorlevel 1 (
        echo [!] Training failed
        exit /b 1
    )
    echo [+] Training complete
    goto :eof
)

if /i "%1"=="evaluate" (
    echo [*] Evaluating Model...
    python scripts/evaluate_model.py --test features/test.csv --model_dir model
    if errorlevel 1 (
        echo [!] Evaluation failed
        exit /b 1
    )
    echo [+] Evaluation complete
    goto :eof
)

if /i "%1"=="build" (
    echo [*] Building Docker Image: %FULL_IMAGE%
    docker build -t %FULL_IMAGE% .
    if errorlevel 1 (
        echo [!] Docker build failed
        exit /b 1
    )
    echo [+] Docker build complete
    goto :eof
)

if /i "%1"=="push" (
    echo [*] Logging in to Docker Hub...
    docker login -u %DOCKER_HUB_ID%
    if errorlevel 1 (
        echo [!] Docker login failed
        exit /b 1
    )
    
    echo [*] Pushing %FULL_IMAGE% to Docker Hub...
    docker push %FULL_IMAGE%
    if errorlevel 1 (
        echo [!] Docker push failed
        exit /b 1
    )
    echo [+] Docker push complete
    goto :eof
)

if /i "%1"=="deploy" (
    echo [*] Deploying to Kubernetes...
    echo.
    echo [*] Applying Deployment...
    kubectl apply -f k8s/deploy.yaml
    if errorlevel 1 (
        echo [!] Deployment failed
        exit /b 1
    )
    
    echo [*] Applying Service...
    kubectl apply -f k8s/service.yaml
    if errorlevel 1 (
        echo [!] Service deployment failed
        exit /b 1
    )
    
    echo [*] Applying HPA...
    kubectl apply -f k8s/hpa.yaml
    if errorlevel 1 (
        echo [!] HPA deployment failed
        exit /b 1
    )
    
    echo.
    echo [+] Deployment complete! Checking status...
    echo.
    kubectl get deployments
    echo.
    kubectl get pods -l app=ids
    echo.
    kubectl get svc ids-svc
    goto :eof
)

if /i "%1"=="all" (
    echo [*] Running full pipeline...
    echo.
    
    echo Step 1: Preprocessing
    python scripts/preprocess_cicids2017.py --input data/raw --out features --seed 42
    if errorlevel 1 (
        echo [!] Preprocessing failed
        exit /b 1
    )
    
    echo.
    echo Step 2: Training
    python scripts/train_autoencoder.py --train features/train_normal.csv --val features/val.csv --outdir model --epochs 60
    if errorlevel 1 (
        echo [!] Training failed
        exit /b 1
    )
    
    echo.
    echo Step 3: Evaluating
    python scripts/evaluate_model.py --test features/test.csv --model_dir model
    if errorlevel 1 (
        echo [!] Evaluation failed
        exit /b 1
    )
    
    echo.
    echo Step 4: Building Docker Image
    docker build -t %FULL_IMAGE% .
    if errorlevel 1 (
        echo [!] Docker build failed
        exit /b 1
    )
    
    echo.
    echo Step 5: Pushing to Docker Hub
    docker login -u %DOCKER_HUB_ID%
    if errorlevel 1 (
        echo [!] Docker login failed
        exit /b 1
    )
    
    docker push %FULL_IMAGE%
    if errorlevel 1 (
        echo [!] Docker push failed
        exit /b 1
    )
    
    echo.
    echo Step 6: Deploying to Kubernetes
    kubectl apply -f k8s/deploy.yaml
    if errorlevel 1 (
        echo [!] Deployment failed
        exit /b 1
    )
    
    kubectl apply -f k8s/service.yaml
    if errorlevel 1 (
        echo [!] Service deployment failed
        exit /b 1
    )
    
    kubectl apply -f k8s/hpa.yaml
    if errorlevel 1 (
        echo [!] HPA deployment failed
        exit /b 1
    )
    
    echo.
    echo ==============================================================================
    echo [+] DEPLOYMENT COMPLETE!
    echo ==============================================================================
    echo.
    echo Access your service:
    echo   kubectl port-forward svc/ids-svc 8080:80
    echo   curl -X POST http://localhost:8080/score ...
    echo.
    echo View logs:
    echo   kubectl logs -f deployment/ids-service
    echo.
    echo Check metrics:
    echo   curl http://localhost:8080/metrics
    echo.
    goto :eof
)

echo [!] Unknown command: %1
echo Use 'deploy.bat' with no args for help.
