
"""Audit raw CIC-IDS2017 CSV files without modifying the source data."""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_COLUMN = "Label"


def audit_csv(
    file_path: Path,
    chunksize: int = 100_000,
) -> dict:
    """Inspect one CSV file incrementally and report its data quality."""

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")

    header = pd.read_csv(file_path, nrows=0).columns
    columns = [str(column).strip() for column in header]

    if len(columns) != len(set(columns)):
        raise ValueError(
            f"Duplicate column names after normalization: {file_path.name}"
        )

    if LABEL_COLUMN not in columns:
        raise ValueError(
            f"Missing '{LABEL_COLUMN}' column in {file_path.name}"
        )

    missing_counts = Counter()
    infinite_counts = Counter()
    label_counts = Counter()
    observed_dtypes = {}

    total_rows = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk.columns = chunk.columns.astype(str).str.strip()

        total_rows += len(chunk)

        for column in chunk.columns:
            missing_counts[column] += int(chunk[column].isna().sum())

            dtype = str(chunk[column].dtype)
            observed_dtypes.setdefault(column, set()).add(dtype)

        numeric = chunk.select_dtypes(include="number")

        if not numeric.empty:
            infinite_mask = np.isinf(numeric.to_numpy())
            counts = infinite_mask.sum(axis=0)

            for column, count in zip(numeric.columns, counts):
                infinite_counts[column] += int(count)

        labels = (
            chunk[LABEL_COLUMN]
            .astype("string")
            .str.strip()
            .fillna("<MISSING>")
            .value_counts(dropna=False)
        )

        label_counts.update({
            str(label): int(count)
            for label, count in labels.items()
        })

    return {
        "file_name": file_path.name,
        "row_count": total_rows,
        "column_count": len(columns),
        "columns": columns,
        "data_types": {
            column: sorted(dtypes)
            for column, dtypes in observed_dtypes.items()
        },
        "missing_values": dict(missing_counts),
        "infinite_values": dict(infinite_counts),
        "total_missing": sum(missing_counts.values()),
        "total_infinite": sum(infinite_counts.values()),
        "class_distribution": dict(label_counts),
    }


def audit_raw_directory(
    raw_dir: Path,
    chunksize: int = 100_000,
) -> dict:
    """Audit all raw CIC-IDS2017 CSV files, excluding Global Test."""

    raw_dir = Path(raw_dir)

    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw data directory not found: {raw_dir}")

    csv_files = sorted(
        file
        for file in raw_dir.glob("*.csv")
        if file.name.casefold() != "global test set.csv"
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No raw CSV files found in {raw_dir}"
        )

    file_reports = [
        audit_csv(file, chunksize=chunksize)
        for file in csv_files
    ]

    reference_columns = set(file_reports[0]["columns"])

    schema_differences = {}

    for report in file_reports:
        current_columns = set(report["columns"])

        schema_differences[report["file_name"]] = {
            "missing_columns": sorted(
                reference_columns - current_columns
            ),
            "extra_columns": sorted(
                current_columns - reference_columns
            ),
        }

    aggregate_labels = Counter()

    for report in file_reports:
        aggregate_labels.update(report["class_distribution"])

    return {
        "dataset": "CIC-IDS2017",
        "files_audited": len(file_reports),
        "total_rows": sum(
            report["row_count"] for report in file_reports
        ),
        "reference_columns": file_reports[0]["columns"],
        "schema_consistent": all(
            not differences["missing_columns"]
            and not differences["extra_columns"]
            for differences in schema_differences.values()
        ),
        "schema_differences": schema_differences,
        "overall_class_distribution": dict(aggregate_labels),
        "files": file_reports,
    }
