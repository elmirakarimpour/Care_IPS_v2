#!/usr/bin/env python
import json
from pathlib import Path
from care_ips.config import GLOBAL_TEST_PATH, RESULTS_DIR
from care_ips.data.loader import inspect_csv

def main():
    report = inspect_csv(GLOBAL_TEST_PATH)
    out = RESULTS_DIR / "summary"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "global_test_quality.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "column_names"}, indent=2))
    print(f"Report: {path}")

if __name__ == "__main__":
    main()
