#!/usr/bin/env python
from pathlib import Path
import pandas as pd
from care_ips.config import GLOBAL_TEST_PATH
from care_ips.data.loader import load_raw_csv

def main():
    test = load_raw_csv(GLOBAL_TEST_PATH)
    raise SystemExit(
        f"Loaded {len(test)} valid global-test rows. No trained model artifact found; "
        "evaluation cannot run until a model is trained on a separate training set."
    )
if __name__ == "__main__":
    main()
