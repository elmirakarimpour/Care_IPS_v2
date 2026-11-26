
"""Load raw CIC-IDS2017 CSV files while preserving data and provenance."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import (
    LABEL_COLUMN,
    LOADER_METADATA_COLUMNS,
    RAW_LABEL_COLUMN,
    SAMPLE_ID_COLUMN,
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_NUMBER_COLUMN,
    DatasetSchema,
)
from .validation import validate_loaded_dataframe


CSV_ENCODING = "utf-8-sig"


def _read_csv_headers(path: Path) -> list[str]:
    """Read original CSV headers while preserving duplicate names."""

    with path.open(
        mode="r",
        encoding=CSV_ENCODING,
        newline="",
    ) as csv_file:
        raw_headers = next(csv.reader(csv_file), None)

    if not raw_headers:
        raise ValueError(f"CSV file has no header: {path}")

    headers = [str(header).strip() for header in raw_headers]

    if any(not header for header in headers):
        raise ValueError(f"Empty column names found in {path}")

    if headers.count(LABEL_COLUMN) != 1:
        raise ValueError(
            f"Expected exactly one '{LABEL_COLUMN}' column in {path}."
        )

    reserved_columns = sorted(
        set(headers).intersection(LOADER_METADATA_COLUMNS)
    )

    if reserved_columns:
        raise ValueError(
            f"Raw CSV contains reserved metadata columns in {path}: "
            f"{reserved_columns}"
        )

    return headers


def _read_csv_data(
    path: Path,
    headers: list[str],
) -> pd.DataFrame:
    """Read CSV records without allowing pandas to rename duplicate headers."""

    try:
        df = pd.read_csv(
            path,
            header=None,
            skiprows=1,
            encoding=CSV_ENCODING,
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=range(len(headers)))

    if len(df.columns) != len(headers):
        raise ValueError(
            f"Header/data column mismatch in {path}: "
            f"{len(headers)} headers and {len(df.columns)} data columns."
        )

    df.columns = pd.Index(headers, dtype="object")

    return df


def _add_traceability_columns(
    df: pd.DataFrame,
    source_file: str,
) -> pd.DataFrame:
    """Add stable sample identifiers and source provenance."""

    result = df.copy()

    row_numbers = np.arange(
        start=1,
        stop=len(result) + 1,
        dtype=np.int64,
    )

    sample_ids = [
        f"{source_file}::row-{row_number:09d}"
        for row_number in row_numbers
    ]

    result.insert(
        0,
        SOURCE_ROW_NUMBER_COLUMN,
        row_numbers,
    )

    result.insert(
        0,
        SOURCE_FILE_COLUMN,
        pd.Series(source_file, index=result.index, dtype="string"),
    )

    result.insert(
        0,
        SAMPLE_ID_COLUMN,
        pd.Series(sample_ids, index=result.index, dtype="string"),
    )

    result[RAW_LABEL_COLUMN] = result[LABEL_COLUMN].copy()

    return result


def load_raw_csv(
    path: str | Path,
    schema: DatasetSchema | None = None,
) -> pd.DataFrame:
    """
    Load a raw CSV without cleaning or dropping records.

    Args:
        path: Source CSV file path.
        schema: Optional dataset-specific schema for strict validation.

    Returns:
        Original data with traceability metadata and preserved labels.

    Raises:
        FileNotFoundError: If the source file does not exist.
        ValueError: If the CSV structure violates the data contract.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    headers = _read_csv_headers(path)
    df = _read_csv_data(path, headers)

    result = _add_traceability_columns(
        df=df,
        source_file=path.name,
    )

    # Validate the loaded data before returning it.
    validate_loaded_dataframe(
        result,
        schema=schema,
    )

    return result


def inspect_csv(path: str | Path) -> dict:
    """
    Inspect source data quality without counting loader metadata.

    Statistics describe the original CSV columns, not the added
    traceability fields.
    """

    df = load_raw_csv(path)

    original_column_positions = [
        position
        for position, column_name in enumerate(df.columns)
        if column_name not in LOADER_METADATA_COLUMNS
    ]

    original_data = df.iloc[:, original_column_positions]

    label_position = next(
        position
        for position, column_name in enumerate(df.columns)
        if column_name == LABEL_COLUMN
    )

    label_values = df.iloc[:, label_position]

    label_counts = label_values.value_counts(
        dropna=False
    ).to_dict()

    numeric_data = original_data.select_dtypes(
        include=["number"]
    )

    numeric_values = numeric_data.to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    return {
        "path": str(path),
        "rows": int(len(df)),
        "columns": int(len(original_data.columns)),
        "labels": {
            str(label): int(count)
            for label, count in label_counts.items()
        },
        "missing_cells": int(original_data.isna().sum().sum()),
        "infinite_cells": int(np.isinf(numeric_values).sum()),
        "column_names": [
            str(column)
            for column in original_data.columns
        ],
    }
