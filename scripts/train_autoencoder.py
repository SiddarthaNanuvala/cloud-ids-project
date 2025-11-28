#!/usr/bin/env python3
"""
Train a simple dense Autoencoder on train_normal.csv
Usage:
  python scripts/train_autoencoder.py --train features/train_normal.csv --val features/val.csv --outdir model --epochs 60
Produces:
 - model/ae.pth
 - model/scaler.joblib
 - model/threshold.json (99th percentile of reconstruction error on val benign)
"""
import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


class AE(nn.Module):
    """Simple dense autoencoder: input -> 64 -> 16 -> 64 -> input"""
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


def fit_ae(X, epochs=50, batch=128, lr=1e-3, device='cpu'):
    """Train autoencoder on data X."""
    n = X.shape[1]
    model = AE(n).to(device)
    ds = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32))), 
        batch_size=batch, 
        shuffle=True
    )
    opt = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch_x,) in ds:
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            loss = ((recon - batch_x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(ds)
            print(f"  Epoch {epoch+1:3d}/{epochs} loss={avg_loss:.6f}")
    
    return model


def main():
    p = argparse.ArgumentParser(description="Train autoencoder anomaly detector")
    p.add_argument("--train", required=True, help="Path to train_normal.csv")
    p.add_argument("--val", required=True, help="Path to val.csv")
    p.add_argument("--outdir", default="model", help="Output directory")
    p.add_argument("--epochs", type=int, default=60, help="Training epochs")
    p.add_argument("--batch", type=int, default=128, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = p.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"Loading training data from {args.train}...")
    df_train = pd.read_csv(args.train)
    print(f"Loading validation data from {args.val}...")
    df_val = pd.read_csv(args.val)
    
    # Get feature columns (exclude labels)
    feat_cols = [c for c in df_train.columns 
                 if c.lower() not in ['label', 'label_bin']]
    print(f"Using {len(feat_cols)} features")
    
    X_train = df_train[feat_cols].values
    X_val = df_val[feat_cols].values
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # Fit scaler and transform
    print("Fitting StandardScaler...")
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    Xv = scaler.transform(X_val)
    
    # Save scaler
    scaler_path = os.path.join(args.outdir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"✓ Saved scaler to {scaler_path}")
    
    # Train model
    print(f"Training autoencoder on {len(Xs)} samples ({args.epochs} epochs)...")
    device = args.device
    model = fit_ae(Xs, epochs=args.epochs, batch=args.batch, lr=args.lr, device=device)
    
    # Save model
    model_path = os.path.join(args.outdir, "ae.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved model to {model_path}")
    
    # Compute reconstruction error threshold on validation benign data
    print("Computing reconstruction error threshold...")
    thresh = None
    
    if "Label_bin" in df_val.columns:
        benign_v = df_val[df_val["Label_bin"] == 0]
        if len(benign_v) > 0:
            print(f"  Using {len(benign_v)} benign validation samples")
            Xbv = scaler.transform(benign_v[feat_cols].values)
            with torch.no_grad():
                mp = AE(len(feat_cols))
                mp.load_state_dict(model.state_dict())
                mp.eval()
                recon = mp(torch.from_numpy(Xbv.astype(np.float32))).numpy()
            errs = ((Xbv - recon) ** 2).mean(axis=1)
            thresh = float(np.percentile(errs, 99.0))
    
    if thresh is None:
        # Fallback: use 99th percentile of full validation set
        print(f"  Using full validation set ({len(Xv)} samples)")
        with torch.no_grad():
            mp = AE(len(feat_cols))
            mp.load_state_dict(model.state_dict())
            mp.eval()
            recon = mp(torch.from_numpy(Xv.astype(np.float32))).numpy()
        errs = ((Xv - recon) ** 2).mean(axis=1)
        thresh = float(np.percentile(errs, 99.0))
    
    # Save threshold
    th_path = os.path.join(args.outdir, "threshold.json")
    with open(th_path, "w") as f:
        json.dump(
            {"threshold": thresh, "feature_columns": feat_cols}, 
            f, 
            indent=2
        )
    print(f"✓ Saved threshold to {th_path}")
    print(f"  Threshold value: {thresh:.6f}")
    print(f"\n✓ Training complete! Artifacts saved to {args.outdir}/")


if __name__ == "__main__":
    main()
