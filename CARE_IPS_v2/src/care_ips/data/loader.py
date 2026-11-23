from pathlib import Path
import logging
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

def load_raw_csv(path: Path, drop_invalid: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    if "Label" not in df.columns:
        raise ValueError(f"Missing Label column in {path}")
    if drop_invalid:
        df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return df

def inspect_csv(path: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    label_counts = df["Label"].value_counts(dropna=False).to_dict() if "Label" in df else {}
    numeric = df.select_dtypes(include="number")
    return {
        "path": str(path), "rows": int(len(df)), "columns": int(len(df.columns)),
        "labels": {str(k): int(v) for k, v in label_counts.items()},
        "missing_cells": int(df.isna().sum().sum()),
        "infinite_cells": int(np.isinf(numeric.to_numpy()).sum()),
        "column_names": list(df.columns),
    }
