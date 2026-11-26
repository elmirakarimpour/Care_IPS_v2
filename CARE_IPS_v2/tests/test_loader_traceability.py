
"""Tests for raw CSV loading and record traceability."""

import csv

import numpy as np
import pandas as pd
import pytest

from care_ips.data.contracts import (
    LABEL_COLUMN,
    RAW_LABEL_COLUMN,
    SAMPLE_ID_COLUMN,
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_NUMBER_COLUMN,
)
from care_ips.data.loader import inspect_csv, load_raw_csv


def _write_csv(path, headers, rows):
    """Create a small CSV file for loader tests."""

    with path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)

    return path


def _create_sample_csv(tmp_path):
    """Create a CSV with duplicate headers and invalid numeric values."""

    return _write_csv(
        tmp_path / "sample.csv",
        headers=[
            "Feature",
            " Fwd Header Length",
            " Fwd Header Length",
            "Label",
        ],
        rows=[
            ["1", "10", "10", "BENIGN"],
            ["", "20", "20", "DDoS"],
            ["inf", "30", "30", "BENIGN"],
        ],
    )


def test_loader_preserves_rows_and_duplicate_columns(tmp_path):
    """The loader must preserve all source records and duplicate headers."""

    path = _create_sample_csv(tmp_path)

    df = load_raw_csv(path)

    assert len(df) == 3
    assert df.columns.duplicated().sum() == 1

    assert df.columns.tolist() == [
        SAMPLE_ID_COLUMN,
        SOURCE_FILE_COLUMN,
        SOURCE_ROW_NUMBER_COLUMN,
        "Feature",
        "Fwd Header Length",
        "Fwd Header Length",
        LABEL_COLUMN,
        RAW_LABEL_COLUMN,
    ]

    # Missing and infinite values must remain unchanged.
    assert pd.isna(df.iloc[1, 3])
    assert np.isinf(df.iloc[2, 3])

    assert df[LABEL_COLUMN].tolist() == [
        "BENIGN",
        "DDoS",
        "BENIGN",
    ]


def test_loader_creates_stable_traceability_identifiers(tmp_path):
    """Identifiers must remain stable across repeated loading."""

    path = _create_sample_csv(tmp_path)

    first_load = load_raw_csv(path)
    second_load = load_raw_csv(path)

    expected_ids = [
        f"{path.name}::row-{row_number:09d}"
        for row_number in range(1, 4)
    ]

    assert first_load[SAMPLE_ID_COLUMN].tolist() == expected_ids
    assert second_load[SAMPLE_ID_COLUMN].tolist() == expected_ids

    assert first_load[SAMPLE_ID_COLUMN].is_unique

    assert first_load[SOURCE_FILE_COLUMN].tolist() == [
        path.name,
        path.name,
        path.name,
    ]

    assert first_load[SOURCE_ROW_NUMBER_COLUMN].tolist() == [1, 2, 3]


def test_loader_preserves_original_labels(tmp_path):
    """label_raw must retain the same values as the original Label."""

    path = _create_sample_csv(tmp_path)

    df = load_raw_csv(path)

    assert df[RAW_LABEL_COLUMN].equals(df[LABEL_COLUMN])


def test_inspect_csv_excludes_loader_metadata(tmp_path):
    """Inspection statistics must describe the original CSV columns."""

    path = _create_sample_csv(tmp_path)

    report = inspect_csv(path)

    assert report["rows"] == 3
    assert report["columns"] == 4

    assert report["column_names"] == [
        "Feature",
        "Fwd Header Length",
        "Fwd Header Length",
        "Label",
    ]

    assert report["missing_cells"] == 1
    assert report["infinite_cells"] == 1

    assert report["labels"] == {
        "BENIGN": 2,
        "DDoS": 1,
    }


def test_loader_rejects_csv_without_label(tmp_path):
    """A source CSV must contain exactly one Label column."""

    path = _write_csv(
        tmp_path / "missing_label.csv",
        headers=["Feature"],
        rows=[["1"]],
    )

    with pytest.raises(ValueError, match="exactly one"):
        load_raw_csv(path)


def test_loader_rejects_reserved_metadata_column_names(tmp_path):
    """Raw CSV headers must not conflict with generated metadata names."""

    path = _write_csv(
        tmp_path / "reserved_column.csv",
        headers=["sample_id", "Label"],
        rows=[["old-id", "BENIGN"]],
    )

    with pytest.raises(ValueError, match="reserved metadata"):
        load_raw_csv(path)
