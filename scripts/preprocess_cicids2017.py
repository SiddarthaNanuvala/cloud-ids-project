#!/usr/bin/env python3
"""
scripts/preprocess_cicids2017.py
Usage:
  python scripts/preprocess_cicids2017.py --input data/raw --out features --seed 42
Produces:
 - features/train_normal.csv
 - features/val.csv
 - features/test.csv
 - features/feature_schema.json
"""
import argparse
import json
import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

FEATURES = [
    "Flow Duration","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Total Fwd Packets","Total Backward Packets",
    "Total Length of Fwd Packets","Total Length of Bwd Packets",
    "Fwd Packet Length Mean","Bwd Packet Length Mean",
    "Max Packet Length","Min Packet Length","Packet Length Mean",
    "Fwd IAT Mean","Bwd IAT Mean","Flow Bytes/s","Flow Packets/s",
    "PSH Flag Count","SYN Flag Count","FIN Flag Count",
    "Protocol"
]

def load_all_csvs(input_dir):
    """Load and concatenate all CSV files from input directory."""
    files = sorted(glob(os.path.join(input_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs in {input_dir}")
    dfs = []
    for f in files:
        print(f"Loading {f}")
        df = pd.read_csv(f, encoding='latin-1')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def basic_clean(df):
    """Clean data: remove invalid values, constant columns, fill missing values."""
    # Replace 'Infinity' strings with NaN, coerce numeric columns
    df = df.replace(['Infinity', 'infinity', 'Inf', 'inf', 'NaN', 'nan'], np.nan)
    
    # Keep only columns we care about (if present)
    cols = [c for c in FEATURES if c in df.columns]
    df = df[cols + (["Label"] if "Label" in df.columns else [])]
    
    # Drop cols with >40% missing
    thresh = int(0.6*len(df))
    df = df.dropna(axis=1, thresh=thresh)
    
    # Fill remaining numeric NaNs with median
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            med = df[c].median()
            df[c] = df[c].fillna(med)
    
    # Remove constant columns
    nunique = df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f"Removing constant columns: {const_cols}")
        df = df.drop(columns=const_cols)
    
    return df

def label_to_binary(df):
    """Convert label to binary: 0=benign, 1=attack."""
    if "Label" not in df.columns:
        df["Label_bin"] = 0
        return df
    df["Label_bin"] = df["Label"].apply(lambda x: 0 if str(x).strip().upper()=="BENIGN" else 1)
    return df

def split_and_save(df, outdir, seed=42):
    """Split data into train/val/test and save to CSV."""
    os.makedirs(outdir, exist_ok=True)
    
    # Keep only feature columns existing in df
    feature_cols = [c for c in FEATURES if c in df.columns]
    df = label_to_binary(df)
    
    # Train: only benign samples
    benign = df[df["Label_bin"]==0].sample(frac=1.0, random_state=seed)
    
    # Use ~60% of benign for train_normal, 20% val, 20% holdout benign
    b_train, b_rest = train_test_split(benign, test_size=0.4, random_state=seed)
    b_val, b_hold = train_test_split(b_rest, test_size=0.5, random_state=seed)
    
    # For val and test include attacks as well
    attacks = df[df["Label_bin"]==1]
    
    # Assemble files
    train_df = b_train[feature_cols]
    val_df = pd.concat([b_val, attacks.sample(frac=0.1, random_state=seed)], ignore_index=True)
    test_df = pd.concat([b_hold, attacks], ignore_index=True)
    
    # Shuffle
    val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # Save CSVs
    train_df.to_csv(os.path.join(outdir, "train_normal.csv"), index=False)
    val_df.to_csv(os.path.join(outdir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(outdir, "test.csv"), index=False)
    
    # Save schema
    schema = {"features": feature_cols}
    with open(os.path.join(outdir, "feature_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)
    
    print(f"✓ Saved train_normal.csv ({len(train_df)} rows)")
    print(f"✓ Saved val.csv ({len(val_df)} rows)")
    print(f"✓ Saved test.csv ({len(test_df)} rows)")
    print(f"✓ Saved feature_schema.json ({len(feature_cols)} features)")
    print(f"Outputs: {outdir}/")

def main():
    p = argparse.ArgumentParser(description="Preprocess CIC-IDS2017 dataset")
    p.add_argument("--input", required=True, help="Input directory containing CSV files")
    p.add_argument("--out", required=True, help="Output directory for processed files")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()
    
    print(f"Loading CIC-IDS2017 data from {args.input}...")
    df = load_all_csvs(args.input)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    print("Cleaning data...")
    df = basic_clean(df)
    print(f"After cleaning: {len(df)} records with {len(df.columns)} columns")
    
    print(f"Splitting and saving to {args.out}...")
    split_and_save(df, args.out, seed=args.seed)

if __name__ == "__main__":
    main()
