#!/usr/bin/env python
"""Train-only preprocessing scaffold; refuses to fit on the global test set."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from care_ips.data.feature_engineering import drop_constant_and_redundant, fit_scaler, transform_with_scaler

def main():
    source = Path("data/interim/CIC-IDS2017/sampled_train.csv")
    if not source.exists():
        raise SystemExit("Missing sampled training data. Run run_sampling.py after adding train CSVs.")
    df = pd.read_csv(source)
    train, val = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Label"])
    # Feature filtering must be learned on train only; this compact scaffold
    # applies constant/correlation filtering to train and aligns other splits.
    train = drop_constant_and_redundant(train)
    features = [c for c in train.select_dtypes(include="number").columns if c != "Label"]
    val = val[train.columns]
    scaler = fit_scaler(train, features)
    train = transform_with_scaler(train, scaler, features)
    val = transform_with_scaler(val, scaler, features)
    out = Path("data/processed/CIC-IDS2017/splits")
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "validation.csv", index=False)
    print("Saved train/validation. Persist the fitted preprocessing object before test transformation.")
if __name__ == "__main__":
    main()
