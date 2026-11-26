
"""Tests for the CARE-IPS data contract."""

import csv

import pandas as pd
import pytest

from care_ips.data.contracts import (
    CIC_IDS2017_SCHEMA,
    LABEL_COLUMN,
    RAW_LABEL_COLUMN,
    SAMPLE_ID_COLUMN,
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_NUMBER_COLUMN,
)
from care_ips.data.loader import load_raw_csv
from care_ips.data.validation import (
    DataContractError,
    validate_processed_dataframe,
    validate_raw_headers,
)


def _build_cic_headers() -> list[str]:
    """Build a synthetic header matching the CIC-IDS2017 schema."""

    features = [
        f"Feature_{index:02d}"
        for index in range(76)
    ]

    features.append("Fwd Header Length")

    return features + [
        LABEL_COLUMN,
        "Fwd Header Length",
    ]


def _write_csv(path, headers, rows):
    """Write a synthetic CSV file."""

    with path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    return path


def _build_valid_processed_dataframe():
    """Build a valid processed DataFrame with 77 numeric features."""

    source_file = CIC_IDS2017_SCHEMA.source_files[0]

    data = {
        f"feature_{index:02d}": [
            float(index),
            float(index + 1),
        ]
        for index in range(77)
    }

    data.update({
        SAMPLE_ID_COLUMN: [
            f"{source_file}::row-000000001",
            f"{source_file}::row-000000002",
        ],
        SOURCE_FILE_COLUMN: [
            source_file,
            source_file,
        ],
        SOURCE_ROW_NUMBER_COLUMN: [1, 2],
        LABEL_COLUMN: ["BENIGN", "DDoS"],
        RAW_LABEL_COLUMN: ["BENIGN", "DDoS"],
    })

    feature_columns = [
        f"feature_{index:02d}"
        for index in range(77)
    ]

    return pd.DataFrame(data), feature_columns


def test_cic_schema_accepts_expected_raw_headers():
    """The CIC schema must accept 79 columns and one duplicate."""

    headers = _build_cic_headers()

    validate_raw_headers(
        headers,
        schema=CIC_IDS2017_SCHEMA,
    )

    assert len(headers) == 79
    assert len(set(headers)) == 78


def test_cic_schema_rejects_wrong_raw_column_count():
    """The CIC schema must reject an incomplete raw header."""

    headers = _build_cic_headers()[:-1]

    with pytest.raises(DataContractError):
        validate_raw_headers(
            headers,
            schema=CIC_IDS2017_SCHEMA,
        )


def test_loader_accepts_valid_cic_schema(tmp_path):
    """The loader must validate a structurally valid CIC CSV."""

    source_file = CIC_IDS2017_SCHEMA.source_files[0]
    path = tmp_path / source_file

    headers = _build_cic_headers()

    row = (
        [str(index) for index in range(76)]
        + ["100", "BENIGN", "100"]
    )

    _write_csv(path, headers, [row])

    df = load_raw_csv(
        path,
        schema=CIC_IDS2017_SCHEMA,
    )

    assert len(df) == 1
    assert len(df.columns) == 83
    assert df[SAMPLE_ID_COLUMN].is_unique


def test_valid_processed_dataframe_passes_contract():
    """A valid processed DataFrame must satisfy the CIC contract."""

    df, features = _build_valid_processed_dataframe()

    validate_processed_dataframe(
        df,
        feature_columns=features,
        schema=CIC_IDS2017_SCHEMA,
    )

    assert len(features) == 77
    assert len(df.columns) == 82


def test_processed_dataframe_rejects_metadata_as_feature():
    """Traceability metadata must never enter model inputs."""

    df, features = _build_valid_processed_dataframe()

    invalid_features = features + [SAMPLE_ID_COLUMN]

    with pytest.raises(DataContractError):
        validate_processed_dataframe(
            df,
            feature_columns=invalid_features,
            schema=CIC_IDS2017_SCHEMA,
        )


def test_processed_dataframe_rejects_non_numeric_features():
    """All model input features must be numeric."""

    df, features = _build_valid_processed_dataframe()

    df[features[0]] = ["invalid", "value"]

    with pytest.raises(DataContractError, match="numeric"):
        validate_processed_dataframe(
            df,
            feature_columns=features,
            schema=CIC_IDS2017_SCHEMA,
        )


def test_processed_dataframe_rejects_unexpected_columns():
    """Processed data must not contain undeclared columns."""

    df, features = _build_valid_processed_dataframe()

    df["unexpected_column"] = [1, 2]

    with pytest.raises(DataContractError, match="schema mismatch"):
        validate_processed_dataframe(
            df,
            feature_columns=features,
            schema=CIC_IDS2017_SCHEMA,
        )


def test_processed_dataframe_rejects_duplicate_columns():
    """Processed data must not contain duplicate column names."""

    df, features = _build_valid_processed_dataframe()

    # Assign duplicate names through a regular Python list.
    # Direct modification of Index.values is not supported.
    column_names = df.columns.tolist()
    column_names[-1] = column_names[-2]

    df.columns = column_names

    with pytest.raises(DataContractError, match="unique column"):
        validate_processed_dataframe(
            df,
            feature_columns=features,
            schema=CIC_IDS2017_SCHEMA,
        )
