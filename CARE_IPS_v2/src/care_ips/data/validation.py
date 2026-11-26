
"""Validation rules for CARE-IPS data contracts."""

from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .contracts import (
    LABEL_COLUMN,
    LOADER_METADATA_COLUMNS,
    NON_FEATURE_COLUMNS,
    RAW_LABEL_COLUMN,
    SAMPLE_ID_COLUMN,
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_NUMBER_COLUMN,
    TRACEABILITY_COLUMNS,
    DatasetSchema,
)


class DataContractError(ValueError):
    """Raised when a DataFrame violates the CARE-IPS data contract."""


def _require_single_column(
    df: pd.DataFrame,
    column: str,
) -> None:
    """Ensure that a required column exists exactly once."""

    count = list(df.columns).count(column)

    if count != 1:
        raise DataContractError(
            f"Expected exactly one '{column}' column; found {count}."
        )


def validate_raw_headers(
    headers: Iterable[str],
    schema: DatasetSchema | None = None,
) -> None:
    """
    Validate original CSV headers.

    Duplicate feature names are allowed when they match
    the supplied dataset schema.
    """

    header_list = list(headers)

    if not header_list:
        raise DataContractError("CSV headers cannot be empty.")

    if any(
        not isinstance(name, str) or not name.strip()
        for name in header_list
    ):
        raise DataContractError(
            "CSV headers must contain non-empty strings."
        )

    if header_list.count(LABEL_COLUMN) != 1:
        raise DataContractError(
            f"Expected exactly one '{LABEL_COLUMN}' column."
        )

    reserved_columns = sorted(
        set(header_list).intersection(LOADER_METADATA_COLUMNS)
    )

    if reserved_columns:
        raise DataContractError(
            f"Raw data contains reserved metadata columns: "
            f"{reserved_columns}"
        )

    duplicate_names = tuple(
        name
        for name, count in Counter(header_list).items()
        if count > 1
    )

    if schema is None:
        return

    if len(header_list) != schema.raw_column_count:
        raise DataContractError(
            f"Expected {schema.raw_column_count} raw columns, "
            f"found {len(header_list)}."
        )

    unique_count = len(set(header_list))

    if unique_count != schema.unique_source_column_count:
        raise DataContractError(
            f"Expected {schema.unique_source_column_count} "
            f"unique source columns, found {unique_count}."
        )

    if set(duplicate_names) != set(schema.duplicate_column_names):
        raise DataContractError(
            f"Unexpected duplicate columns. "
            f"Expected {schema.duplicate_column_names}, "
            f"found {duplicate_names}."
        )

    feature_count = len(set(header_list) - {LABEL_COLUMN})

    if feature_count != schema.feature_count:
        raise DataContractError(
            f"Expected {schema.feature_count} unique features, "
            f"found {feature_count}."
        )


def validate_traceability(df: pd.DataFrame) -> None:
    """
    Validate sample identifiers and source provenance.

    Each record must have:
    - A unique sample_id.
    - A non-empty source_file.
    - A positive integer source_row_number.
    - An identifier consistent with its source information.
    """

    for column in TRACEABILITY_COLUMNS:
        _require_single_column(df, column)

    sample_ids = df[SAMPLE_ID_COLUMN].astype("string")
    source_files = df[SOURCE_FILE_COLUMN].astype("string")

    if (
        sample_ids.isna().any()
        or sample_ids.str.strip().eq("").any()
    ):
        raise DataContractError(
            "sample_id contains missing or empty values."
        )

    if not sample_ids.is_unique:
        raise DataContractError(
            "sample_id values must be unique."
        )

    if (
        source_files.isna().any()
        or source_files.str.strip().eq("").any()
    ):
        raise DataContractError(
            "source_file contains missing or empty values."
        )

    row_numbers = pd.to_numeric(
        df[SOURCE_ROW_NUMBER_COLUMN],
        errors="coerce",
    )

    if row_numbers.isna().any():
        raise DataContractError(
            "source_row_number contains invalid or missing values."
        )

    numeric_rows = row_numbers.to_numpy(dtype=np.float64)

    if not np.isfinite(numeric_rows).all():
        raise DataContractError(
            "source_row_number must contain finite values."
        )

    if (row_numbers <= 0).any():
        raise DataContractError(
            "source_row_number values must be positive."
        )

    if (row_numbers % 1 != 0).any():
        raise DataContractError(
            "source_row_number values must be integers."
        )

    if (row_numbers > np.iinfo(np.int64).max).any():
        raise DataContractError(
            "source_row_number exceeds the supported integer range."
        )

    row_numbers = row_numbers.astype("int64")

    source_records = pd.DataFrame(
        {
            SOURCE_FILE_COLUMN: source_files,
            SOURCE_ROW_NUMBER_COLUMN: row_numbers,
        }
    )

    if source_records.duplicated().any():
        raise DataContractError(
            "The same source record appears more than once."
        )

    expected_ids = [
        f"{source}::row-{row:09d}"
        for source, row in zip(
            source_files.tolist(),
            row_numbers.tolist(),
            strict=True,
        )
    ]

    if sample_ids.tolist() != expected_ids:
        raise DataContractError(
            "sample_id does not match source_file "
            "and source_row_number."
        )


def validate_loaded_dataframe(
    df: pd.DataFrame,
    schema: DatasetSchema | None = None,
) -> None:
    """
    Validate a DataFrame immediately after raw data loading.

    Duplicate original feature names are permitted.
    Missing and infinite feature values are also permitted
    at this stage.
    """

    _require_single_column(df, LABEL_COLUMN)
    _require_single_column(df, RAW_LABEL_COLUMN)

    validate_traceability(df)

    labels = df[LABEL_COLUMN]
    raw_labels = df[RAW_LABEL_COLUMN]

    if labels.isna().any():
        raise DataContractError(
            "Label contains missing values."
        )

    if raw_labels.isna().any():
        raise DataContractError(
            "label_raw contains missing values."
        )

    if not labels.equals(raw_labels):
        raise DataContractError(
            "label_raw must match Label immediately after loading."
        )

    if schema is None:
        return

    raw_positions = [
        position
        for position, column in enumerate(df.columns)
        if column not in LOADER_METADATA_COLUMNS
    ]

    raw_headers = [
        df.columns[position]
        for position in raw_positions
    ]

    validate_raw_headers(raw_headers, schema)

    source_files = df[SOURCE_FILE_COLUMN].unique().tolist()

    unexpected_files = sorted(
        set(source_files).difference(schema.source_files)
    )

    if unexpected_files:
        raise DataContractError(
            f"Unexpected source files for {schema.name}: "
            f"{unexpected_files}"
        )

    if len(df.columns) != schema.expected_loader_column_count:
        raise DataContractError(
            f"Expected {schema.expected_loader_column_count} "
            f"loaded columns, found {len(df.columns)}."
        )

    non_numeric_features = [
        str(df.columns[position])
        for position in raw_positions
        if df.columns[position] != LABEL_COLUMN
        and not is_numeric_dtype(df.iloc[:, position])
    ]

    if non_numeric_features:
        raise DataContractError(
            f"Expected numeric source features. "
            f"Non-numeric columns: {non_numeric_features}"
        )


def validate_model_feature_columns(
    feature_columns: Iterable[str],
) -> None:
    """
    Prevent labels and traceability metadata from entering model inputs.
    """

    columns = list(feature_columns)

    if not columns:
        raise DataContractError(
            "At least one model feature must be specified."
        )

    if any(not isinstance(column, str) for column in columns):
        raise DataContractError(
            "All model feature names must be strings."
        )

    if len(columns) != len(set(columns)):
        raise DataContractError(
            "Model feature names must be unique."
        )

    forbidden = sorted(
        set(columns).intersection(NON_FEATURE_COLUMNS)
    )

    if forbidden:
        raise DataContractError(
            f"Non-feature columns cannot be model inputs: {forbidden}"
        )


def validate_processed_dataframe(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    schema: DatasetSchema | None = None,
) -> None:
    """
    Validate processed data before training or evaluation.

    Processed data must have:
    - Unique column names.
    - Valid record identifiers and provenance.
    - Both Label and label_raw columns.
    - Numeric model features.
    - No unexpected or missing columns.
    """

    if not df.columns.is_unique:
        raise DataContractError(
            "Processed data must have unique column names."
        )

    _require_single_column(df, LABEL_COLUMN)
    _require_single_column(df, RAW_LABEL_COLUMN)

    validate_traceability(df)

    columns = list(feature_columns)

    validate_model_feature_columns(columns)

    missing_features = sorted(
        set(columns).difference(df.columns)
    )

    if missing_features:
        raise DataContractError(
            f"Model features are missing: {missing_features}"
        )

    non_numeric_features = [
        column
        for column in columns
        if not is_numeric_dtype(df[column])
    ]

    if non_numeric_features:
        raise DataContractError(
            f"Model features must be numeric: {non_numeric_features}"
        )

    if df[LABEL_COLUMN].isna().any():
        raise DataContractError(
            "Processed Label contains missing values."
        )

    if df[RAW_LABEL_COLUMN].isna().any():
        raise DataContractError(
            "Processed label_raw contains missing values."
        )

    expected_columns = set(columns).union(NON_FEATURE_COLUMNS)
    actual_columns = set(df.columns)

    missing_columns = sorted(
        expected_columns.difference(actual_columns)
    )

    unexpected_columns = sorted(
        actual_columns.difference(expected_columns)
    )

    if missing_columns or unexpected_columns:
        raise DataContractError(
            f"Processed schema mismatch. "
            f"Missing: {missing_columns}; "
            f"unexpected: {unexpected_columns}."
        )

    if schema is not None:
        if len(columns) != schema.feature_count:
            raise DataContractError(
                f"Expected {schema.feature_count} features, "
                f"found {len(columns)}."
            )

        if len(df.columns) != schema.expected_processed_column_count:
            raise DataContractError(
                f"Expected {schema.expected_processed_column_count} "
                f"processed columns, found {len(df.columns)}."
            )
