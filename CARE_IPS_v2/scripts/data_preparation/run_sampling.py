#!/usr/bin/env python
"""Legacy-compatible stratified sampler for raw training CSVs only."""
from pathlib import Path
import pandas as pd
from care_ips.data.loader import load_raw_csv
from care_ips.data.sampler import stratified_sample
from care_ips.data.contracts import CIC_IDS2017_SCHEMA

def main():
    raw_dir = Path("data/raw/CIC-IDS2017/train")
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No training CSVs found in {raw_dir}; Global test is intentionally excluded.")
    frames = []
    for file in files:
        frame = load_raw_csv(file, schema=CIC_IDS2017_SCHEMA)
        frames.append(stratified_sample(frame, min(20000, len(frame)), seed=42))
    out = Path("data/interim/CIC-IDS2017")
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out / "sampled_train.csv", index=False)
if __name__ == "__main__":
    main()
