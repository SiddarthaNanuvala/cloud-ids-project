#!/usr/bin/env python3
"""
Evaluate model performance on test set.
Usage:
  python scripts/evaluate_model.py --test features/test.csv --model_dir model
Computes precision, recall, F1, ROC-AUC, PR-AUC.
"""
import argparse
import os
import joblib
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report
)


class AE(nn.Module):
    """Autoencoder matching training architecture."""
    def __init__(self, n):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, n)
        )
    
    def forward(self, x):
        return self.dec(self.enc(x))


def main():
    p = argparse.ArgumentParser(description="Evaluate anomaly detection model")
    p.add_argument("--test", required=True, help="Path to test.csv")
    p.add_argument("--model_dir", default="model", help="Model artifacts directory")
    p.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = p.parse_args()
    
    print(f"Loading model from {args.model_dir}...")
    
    # Load scaler
    scaler = joblib.load(os.path.join(args.model_dir, "scaler.joblib"))
    print("✓ Loaded scaler")
    
    # Load threshold and feature columns
    with open(os.path.join(args.model_dir, "threshold.json"), "r") as f:
        th = json.load(f)
    feat_cols = th["feature_columns"]
    thresh = th["threshold"]
    print(f"✓ Loaded threshold config: {len(feat_cols)} features, threshold={thresh:.6f}")
    
    # Load model
    device = args.device
    model = AE(len(feat_cols)).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(args.model_dir, "ae.pth"),
            map_location=device
        )
    )
    model.eval()
    print("✓ Loaded model")
    
    # Load test data
    print(f"\nLoading test data from {args.test}...")
    df = pd.read_csv(args.test)
    if "Label_bin" not in df.columns:
        raise ValueError("Test set must have Label_bin column")
    
    X = df[feat_cols].values
    y = df["Label_bin"].values
    print(f"Test set: {len(X)} samples, {len(feat_cols)} features")
    print(f"  - Normal: {(y==0).sum()}")
    print(f"  - Anomaly: {(y==1).sum()}")
    
    # Preprocess
    Xs = scaler.transform(X)
    
    # Inference
    print("\nRunning inference...")
    with torch.no_grad():
        recon = model(torch.from_numpy(Xs.astype(np.float32)).to(device)).cpu().numpy()
    
    # Compute reconstruction errors
    errs = ((Xs - recon) ** 2).mean(axis=1)
    y_pred = (errs > thresh).astype(int)
    
    print(f"✓ Predictions: {len(y_pred)} samples")
    print(f"  - Normal: {(y_pred==0).sum()}")
    print(f"  - Anomaly: {(y_pred==1).sum()}")
    
    # Metrics
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:6d}  FP={fp:6d}")
    print(f"  FN={fn:6d}  TP={tp:6d}")
    
    # Precision, Recall, F1
    p_score, r_score, f_score, _ = precision_recall_fscore_support(
        y, y_pred, average='binary', zero_division=0
    )
    print(f"\nClassification Metrics:")
    print(f"  Precision: {p_score:.4f}")
    print(f"  Recall:    {r_score:.4f}")
    print(f"  F1-Score:  {f_score:.4f}")
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y, errs)
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    except Exception as e:
        print(f"  ROC-AUC:   N/A ({e})")
        roc_auc = None
    
    # PR-AUC
    try:
        precision_arr, recall_arr, _ = precision_recall_curve(y, errs)
        pr_auc = auc(recall_arr, precision_arr)
        print(f"  PR-AUC:    {pr_auc:.4f}")
    except Exception as e:
        print(f"  PR-AUC:    N/A ({e})")
        pr_auc = None
    
    # Specificity, Sensitivity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\nAdditional Metrics:")
    print(f"  Sensitivity (TPR): {sensitivity:.4f}")
    print(f"  Specificity (TNR): {specificity:.4f}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y, y_pred, target_names=["Normal", "Anomaly"], zero_division=0))
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Threshold: {thresh:.6f}")
    print(f"Error statistics on test set:")
    print(f"  Min:  {errs.min():.6f}")
    print(f"  Mean: {errs.mean():.6f}")
    print(f"  Std:  {errs.std():.6f}")
    print(f"  Max:  {errs.max():.6f}")
    
    # Save results
    results = {
        "threshold": thresh,
        "features": len(feat_cols),
        "test_samples": len(y),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(p_score),
        "recall": float(r_score),
        "f1_score": float(f_score),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "error_statistics": {
            "min": float(errs.min()),
            "mean": float(errs.mean()),
            "std": float(errs.std()),
            "max": float(errs.max())
        }
    }
    
    results_path = os.path.join(args.model_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")


if __name__ == "__main__":
    main()
