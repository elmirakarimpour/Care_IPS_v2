import pandas as pd

def stratified_sample(df: pd.DataFrame, n: int, label_col: str = "Label", seed: int = 42):
    if label_col not in df:
        raise ValueError(f"Missing label column: {label_col}")
    if n <= 0:
        raise ValueError("n must be positive")
    if len(df) == 0:
        return df.copy()
    n = min(n, len(df))
    # Allocate approximately proportional counts; GroupBy.sample handles each class.
    frac = n / len(df)
    out = df.groupby(label_col, group_keys=False).sample(frac=frac, random_state=seed)
    return out.reset_index(drop=True)
