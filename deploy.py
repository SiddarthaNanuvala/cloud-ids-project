#!/usr/bin/env python3
"""
Deploy IDS Service - Automated build, push, and deployment script
Usage: python deploy.py [preprocess|train|build|push|deploy|all]
"""
import os
import subprocess
import sys
from pathlib import Path

DOCKER_HUB_ID = "siddartha6174"
IMAGE_NAME = "ids"
IMAGE_TAG = "latest"
FULL_IMAGE = f"{DOCKER_HUB_ID}/{IMAGE_NAME}:{IMAGE_TAG}"

def run_command(cmd, description=""):
    """Run shell command and handle errors."""
    if description:
        print(f"\n{'='*70}")
        print(f"▶ {description}")
        print(f"{'='*70}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        if isinstance(cmd, list):
            result = subprocess.run(cmd, check=True)
        else:
            result = subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description if description else 'Command'} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description if description else 'Command'} failed with error code {e.returncode}\n")
        return False

def preprocess():
    """Run data preprocessing."""
    return run_command([
        sys.executable, "scripts/preprocess_cicids2017.py",
        "--input", "data/raw",
        "--out", "features",
        "--seed", "42"
    ], "Data Preprocessing")

def train():
    """Train the autoencoder model."""
    return run_command([
        sys.executable, "scripts/train_autoencoder.py",
        "--train", "features/train_normal.csv",
        "--val", "features/val.csv",
        "--outdir", "model",
        "--epochs", "60"
    ], "Model Training")

def evaluate():
    """Evaluate the model."""
    return run_command([
        sys.executable, "scripts/evaluate_model.py",
        "--test", "features/test.csv",
        "--model_dir", "model"
    ], "Model Evaluation")

def build():
    """Build Docker image."""
    return run_command(f"docker build -t {FULL_IMAGE} .", "Building Docker Image")

def push():
    """Push Docker image to Docker Hub."""
    print(f"\n{'='*70}")
    print("🔐 Docker Hub Login Required")
    print(f"{'='*70}")
    print(f"Username: {DOCKER_HUB_ID}")
    print("You will be prompted to enter your Docker Hub token/password.\n")
    
    if not run_command("docker login", "Docker Hub Login"):
        print("❌ Docker login failed. Please check your credentials.")
        return False
    
    return run_command(f"docker push {FULL_IMAGE}", f"Pushing {FULL_IMAGE} to Docker Hub")

def deploy():
    """Deploy to Kubernetes."""
    print(f"\n{'='*70}")
    print("🚀 Kubernetes Deployment")
    print(f"{'='*70}\n")
    
    commands = [
        ("kubectl apply -f k8s/deploy.yaml", "Deploying Deployment"),
        ("kubectl apply -f k8s/service.yaml", "Deploying Service"),
        ("kubectl apply -f k8s/hpa.yaml", "Deploying HPA"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    
    # Verify deployment
    print(f"\n{'='*70}")
    print("✅ Deployment Complete! Verifying status...\n")
    print(f"{'='*70}\n")
    
    run_command("kubectl get deployments", "Checking Deployments")
    run_command("kubectl get pods -l app=ids", "Checking Pods")
    run_command("kubectl get svc ids-svc", "Checking Service")
    
    return True

def main():
    """Main entry point."""
    print(f"\n{'='*70}")
    print(f"CIC-IDS2017 Anomaly Detection - Deployment Script")
    print(f"Docker Hub ID: {DOCKER_HUB_ID}")
    print(f"Image: {FULL_IMAGE}")
    print(f"{'='*70}\n")
    
    if len(sys.argv) < 2:
        print("Usage: python deploy.py [command]")
        print("\nAvailable commands:")
        print("  preprocess  - Run data preprocessing")
        print("  train       - Train autoencoder model")
        print("  evaluate    - Evaluate model performance")
        print("  build       - Build Docker image locally")
        print("  push        - Push image to Docker Hub")
        print("  deploy      - Deploy to Kubernetes")
        print("  all         - Run all steps (preprocess -> train -> build -> push -> deploy)")
        print("\nExamples:")
        print("  python deploy.py build")
        print("  python deploy.py push")
        print("  python deploy.py all")
        return
    
    command = sys.argv[1].lower()
    
    if command == "preprocess":
        preprocess()
    elif command == "train":
        train()
    elif command == "evaluate":
        evaluate()
    elif command == "build":
        build()
    elif command == "push":
        push()
    elif command == "deploy":
        deploy()
    elif command == "all":
        print("Running full pipeline (preprocess → train → evaluate → build → push → deploy)\n")
        if preprocess() and train() and evaluate() and build() and push() and deploy():
            print(f"\n{'='*70}")
            print("🎉 DEPLOYMENT COMPLETE!")
            print(f"{'='*70}")
            print(f"\nAccess your service:")
            print("  kubectl port-forward svc/ids-svc 8080:80")
            print("  curl -X POST http://localhost:8080/score ...")
            print(f"\nView logs:")
            print("  kubectl logs -f deployment/ids-service")
            print(f"\nCheck metrics:")
            print("  curl http://localhost:8080/metrics")
            return
        else:
            print(f"\n{'='*70}")
            print("❌ Pipeline failed!")
            print(f"{'='*70}\n")
            return
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'python deploy.py' with no args for help.")

if __name__ == "__main__":
    main()
