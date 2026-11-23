"""Audit raw CIC-IDS2017 CSV files without modifying the source data."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_COLUMN = "Label"
CSV_ENCODING = "utf-8-sig"
DEFAULT_CHUNKSIZE = 100_000
EXCLUDED_FILES = {"global test set.csv"}
EXPECTED_RAW_FILES = {
    "friday-workinghours-afternoon-ddos.pcap_iscx.csv",
    "friday-workinghours-afternoon-portscan.pcap_iscx.csv",
    "friday-workinghours-morning.pcap_iscx.csv",
    "monday-workinghours.pcap_iscx.csv",
    "thursday-workinghours-afternoon-infilteration.pcap_iscx.csv",
    "thursday-workinghours-morning-webattacks.pcap_iscx.csv",
    "tuesday-workinghours.pcap_iscx.csv",
    "wednesday-workinghours.pcap_iscx.csv",
}
REPLACEMENT_CHARACTER = "\ufffd"
MOJIBAKE_MARKERS = ("ï¿½", "Ã", "â€")


def _read_header(file_path: Path) -> list[str]:
    """Read the original CSV header without pandas renaming duplicate names."""

    try:
        with file_path.open(
            "r", encoding=CSV_ENCODING, errors="strict", newline=""
        ) as file:
            header = next(csv.reader(file), None)
    except UnicodeDecodeError as error:
        raise UnicodeError(
            f"Could not decode {file_path.name} as {CSV_ENCODING}. "
            "The source encoding must be investigated before auditing."
        ) from error

    if not header:
        raise ValueError(f"CSV file is empty or has no header: {file_path}")

    return [str(name) for name in header]


def _detect_bom(file_path: Path) -> str:
    """Identify a Unicode byte-order mark, if one exists."""

    with file_path.open("rb") as file:
        prefix = file.read(4)

    if prefix.startswith(b"\xef\xbb\xbf"):
        return "UTF-8 BOM"
    if prefix.startswith(b"\xff\xfe\x00\x00"):
        return "UTF-32 LE BOM"
    if prefix.startswith(b"\x00\x00\xfe\xff"):
        return "UTF-32 BE BOM"
    if prefix.startswith(b"\xff\xfe"):
        return "UTF-16 LE BOM"
    if prefix.startswith(b"\xfe\xff"):
        return "UTF-16 BE BOM"
    return "None"


def _make_unique_column_names(columns: list[str]) -> list[str]:
    """Create unique internal names while preserving the original header report."""

    occurrences: Counter[str] = Counter()
    used: set[str] = set()
    unique_names: list[str] = []

    for position, original_name in enumerate(columns, start=1):
        base_name = original_name.strip()
        if not base_name:
            base_name = f"__unnamed_column_{position}"

        occurrences[original_name.strip()] += 1
        occurrence = occurrences[original_name.strip()]
        candidate = (
            base_name
            if occurrence == 1
            else f"{base_name}__duplicate_{occurrence}"
        )

        collision_index = 2
        while candidate in used:
            candidate = f"{base_name}__duplicate_{occurrence}_{collision_index}"
            collision_index += 1

        used.add(candidate)
        unique_names.append(candidate)

    return unique_names


def _percentage(count: int, total: int) -> float:
    """Return a percentage rounded for a readable JSON report."""

    if total == 0:
        return 0.0
    return round((count / total) * 100, 6)


def _hash_normalized_labels(labels: pd.Series) -> np.ndarray:
    """Hash trimmed labels consistently while keeping blank labels explicit."""

    normalized = labels.astype("string").str.strip()
    normalized = normalized.mask(
        normalized.isna() | normalized.eq(""),
        "<MISSING>",
    )
    return pd.util.hash_pandas_object(
        normalized,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64, copy=False)


def _single_label_hash(label: str) -> int:
    """Return the same stable pandas hash used for a normalized label."""

    normalized = pd.Series([label], dtype="string")
    return int(
        pd.util.hash_pandas_object(
            normalized,
            index=False,
            categorize=True,
        ).iloc[0]
    )


def _summarize_feature_label_conflicts(
    feature_hashes: np.ndarray,
    label_hashes: np.ndarray,
    labels: list[str],
) -> dict:
    """Identify identical feature vectors associated with multiple labels."""

    if feature_hashes.size != label_hashes.size:
        raise ValueError("Feature and label hash arrays must have equal length.")

    label_hash_to_name = {
        _single_label_hash(label): label
        for label in labels
    }

    pairs = np.empty(
        feature_hashes.size,
        dtype=[("feature_hash", np.uint64), ("label_hash", np.uint64)],
    )
    pairs["feature_hash"] = feature_hashes
    pairs["label_hash"] = label_hashes

    unique_pairs = np.unique(pairs)

    if unique_pairs.size == 0:
        return {
            "unique_feature_label_pairs": 0,
            "feature_vectors_with_multiple_labels": 0,
            "rows_belonging_to_conflicting_feature_vectors": 0,
            "additional_label_assignments_beyond_one_per_feature_vector": 0,
            "label_pair_group_counts": {},
            "method": (
                "64-bit hashes of feature cells and normalized labels; "
                "hash collisions are theoretically possible."
            ),
        }

    unique_feature_hashes, first_indices, label_counts = np.unique(
        unique_pairs["feature_hash"],
        return_index=True,
        return_counts=True,
    )
    conflict_mask = label_counts > 1
    conflicting_feature_hashes = unique_feature_hashes[conflict_mask]

    label_pair_counts: Counter[str] = Counter()

    for group_index in np.flatnonzero(conflict_mask):
        start = int(first_indices[group_index])
        stop = start + int(label_counts[group_index])
        group_label_hashes = unique_pairs["label_hash"][start:stop]

        group_labels = sorted(
            {
                label_hash_to_name.get(int(label_hash), "<UNKNOWN_LABEL_HASH>")
                for label_hash in group_label_hashes
            }
        )

        for first_label, second_label in combinations(group_labels, 2):
            label_pair_counts[
                f"{first_label} || {second_label}"
            ] += 1

    rows_in_conflicting_groups = int(
        np.isin(feature_hashes, conflicting_feature_hashes).sum()
    )

    return {
        "unique_feature_label_pairs": int(unique_pairs.size),
        "feature_vectors_with_multiple_labels": int(
            conflicting_feature_hashes.size
        ),
        "rows_belonging_to_conflicting_feature_vectors": (
            rows_in_conflicting_groups
        ),
        "additional_label_assignments_beyond_one_per_feature_vector": int(
            unique_pairs.size - unique_feature_hashes.size
        ),
        "label_pair_group_counts": dict(label_pair_counts),
        "method": (
            "64-bit hashes of feature cells and normalized labels; "
            "hash collisions are theoretically possible."
        ),
    }


def _negative_value_summary(
    numeric_statistics: dict[str, dict[str, float | int | None]],
) -> dict[str, dict[str, float | int | None]]:
    """Summarize numeric columns containing negative finite values."""

    return {
        column: {
            "minimum": stats["minimum"],
            "maximum": stats["maximum"],
            "negative_count": stats["negative_count"],
        }
        for column, stats in numeric_statistics.items()
        if stats["negative_count"] > 0
    }


def _update_numeric_statistics(
    chunk: pd.DataFrame,
    infinite_counts: Counter,
    numeric_statistics: dict[str, dict[str, float | int | None]],
) -> None:
    """Update numeric quality counts and finite value ranges for one chunk."""

    numeric = chunk.select_dtypes(include=[np.number])

    for column in numeric.columns:
        values = numeric[column].to_numpy(dtype=np.float64, na_value=np.nan)
        infinite_mask = np.isinf(values)
        finite_values = values[np.isfinite(values)]

        infinite_counts[column] += int(infinite_mask.sum())

        stats = numeric_statistics.setdefault(
            column,
            {
                "minimum": None,
                "maximum": None,
                "zero_count": 0,
                "negative_count": 0,
                "finite_value_count": 0,
            },
        )

        if finite_values.size:
            chunk_minimum = float(finite_values.min())
            chunk_maximum = float(finite_values.max())

            stats["minimum"] = (
                chunk_minimum
                if stats["minimum"] is None
                else min(float(stats["minimum"]), chunk_minimum)
            )
            stats["maximum"] = (
                chunk_maximum
                if stats["maximum"] is None
                else max(float(stats["maximum"]), chunk_maximum)
            )
            stats["zero_count"] += int((finite_values == 0).sum())
            stats["negative_count"] += int((finite_values < 0).sum())
            stats["finite_value_count"] += int(finite_values.size)


def _audit_csv(
    file_path: Path, chunksize: int
) -> tuple[dict, np.ndarray, np.ndarray, dict[str, dict]]:
    """Audit one CSV and return its report plus unique row/feature hashes."""

    raw_columns = _read_header(file_path)
    normalized_columns = [name.strip() for name in raw_columns]

    if LABEL_COLUMN not in normalized_columns:
        raise ValueError(f"Missing '{LABEL_COLUMN}' column in {file_path.name}")

    internal_columns = _make_unique_column_names(raw_columns)
    label_position = normalized_columns.index(LABEL_COLUMN)
    label_internal_name = internal_columns[label_position]

    header_positions: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(normalized_columns):
        header_positions[name].append(index)

    duplicate_header_groups = {
        name: positions
        for name, positions in header_positions.items()
        if len(positions) > 1
    }

    duplicate_column_stats: dict[str, dict] = {}
    for name, positions in duplicate_header_groups.items():
        for duplicate_position in positions[1:]:
            key = f"{name} (columns {positions[0] + 1} and {duplicate_position + 1})"
            duplicate_column_stats[key] = {
                "column_name": name,
                "first_position": positions[0] + 1,
                "duplicate_position": duplicate_position + 1,
                "equal_values": 0,
                "different_values": 0,
                "both_missing": 0,
                "missing_in_first": 0,
                "missing_in_duplicate": 0,
            }

    columns_with_whitespace = [
        {
            "position": index + 1,
            "original_name": original,
            "normalized_name": original.strip(),
        }
        for index, original in enumerate(raw_columns)
        if original != original.strip()
    ]

    missing_counts: Counter = Counter()
    infinite_counts: Counter = Counter()
    label_counts: Counter = Counter()
    observed_dtypes: dict[str, set[str]] = defaultdict(set)
    numeric_statistics: dict[str, dict[str, float | int | None]] = {}

    first_feature_values: dict[str, object] = {}
    varying_features: set[str] = set()

    label_missing_rows = 0
    label_whitespace_rows = 0
    replacement_character_rows = 0
    replacement_character_count = 0
    mojibake_rows = 0
    mojibake_marker_counts: Counter = Counter()

    total_rows = 0
    row_hash_parts: list[np.ndarray] = []
    feature_hash_parts: list[np.ndarray] = []
    label_hash_parts: list[np.ndarray] = []

    for chunk in pd.read_csv(
        file_path,
        header=0,
        names=internal_columns,
        chunksize=chunksize,
        low_memory=False,
        encoding=CSV_ENCODING,
        encoding_errors="strict",
    ):
        total_rows += len(chunk)

        for column in chunk.columns:
            missing_counts[column] += int(chunk[column].isna().sum())
            observed_dtypes[column].add(str(chunk[column].dtype))

            if column == label_internal_name or column in varying_features:
                continue

            non_missing = chunk[column].dropna()
            if non_missing.empty:
                continue

            if column not in first_feature_values:
                first_feature_values[column] = non_missing.iloc[0]

            if bool(non_missing.ne(first_feature_values[column]).any()):
                varying_features.add(column)

        _update_numeric_statistics(
            chunk,
            infinite_counts,
            numeric_statistics,
        )

        raw_labels = chunk[label_internal_name].astype("string")
        trimmed_labels = raw_labels.str.strip()

        label_whitespace_rows += int(
            (raw_labels.notna() & raw_labels.ne(trimmed_labels)).sum()
        )

        blank_labels = trimmed_labels.isna() | trimmed_labels.eq("")
        label_missing_rows += int(blank_labels.sum())
        normalized_labels = trimmed_labels.mask(blank_labels, "<MISSING>")

        label_counts.update(
            {
                str(label): int(count)
                for label, count in normalized_labels.value_counts(
                    dropna=False
                ).items()
            }
        )

        label_text = trimmed_labels.fillna("")
        replacement_mask = label_text.str.contains(
            REPLACEMENT_CHARACTER, regex=False
        )
        replacement_character_rows += int(replacement_mask.sum())
        replacement_character_count += int(
            label_text.str.count(REPLACEMENT_CHARACTER).sum()
        )

        any_mojibake_mask = pd.Series(False, index=chunk.index)
        for marker in MOJIBAKE_MARKERS:
            marker_mask = label_text.str.contains(marker, regex=False)
            marker_rows = int(marker_mask.sum())
            if marker_rows:
                mojibake_marker_counts[marker] += marker_rows
                any_mojibake_mask |= marker_mask
        mojibake_rows += int(any_mojibake_mask.sum())

        for key, stats in duplicate_column_stats.items():
            first = chunk.iloc[:, stats["first_position"] - 1]
            duplicate = chunk.iloc[:, stats["duplicate_position"] - 1]

            both_missing = first.isna() & duplicate.isna()
            equal_mask = first.eq(duplicate).fillna(False) | both_missing

            stats["equal_values"] += int(equal_mask.sum())
            stats["different_values"] += int((~equal_mask).sum())
            stats["both_missing"] += int(both_missing.sum())
            stats["missing_in_first"] += int(first.isna().sum())
            stats["missing_in_duplicate"] += int(duplicate.isna().sum())


    # A second string-based pass makes duplicate detection independent of
    # pandas' chunk-specific numeric dtype inference.
    hash_pass_rows = 0
    for hash_chunk in pd.read_csv(
        file_path,
        header=0,
        names=internal_columns,
        chunksize=chunksize,
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        encoding=CSV_ENCODING,
        encoding_errors="strict",
    ):
        hash_pass_rows += len(hash_chunk)
        row_hash_parts.append(
            pd.util.hash_pandas_object(
                hash_chunk, index=False, categorize=True
            ).to_numpy(dtype=np.uint64, copy=False)
        )

        feature_hash_chunk = hash_chunk.drop(columns=[label_internal_name])
        feature_hash_parts.append(
            pd.util.hash_pandas_object(
                feature_hash_chunk, index=False, categorize=True
            ).to_numpy(dtype=np.uint64, copy=False)
        )
        label_hash_parts.append(
            _hash_normalized_labels(hash_chunk[label_internal_name])
        )

    if hash_pass_rows != total_rows:
        raise RuntimeError(
            f"Row count changed while auditing {file_path.name}: "
            f"data pass={total_rows}, duplicate-check pass={hash_pass_rows}."
        )

    row_hashes = (
        np.concatenate(row_hash_parts)
        if row_hash_parts
        else np.array([], dtype=np.uint64)
    )
    feature_hashes = (
        np.concatenate(feature_hash_parts)
        if feature_hash_parts
        else np.array([], dtype=np.uint64)
    )
    label_hashes = (
        np.concatenate(label_hash_parts)
        if label_hash_parts
        else np.array([], dtype=np.uint64)
    )

    unique_row_hashes = np.unique(row_hashes)
    unique_feature_hashes = np.unique(feature_hashes)

    all_missing_columns = sorted(
        column
        for column, count in missing_counts.items()
        if count == total_rows and total_rows > 0
    )

    constant_features = sorted(
        column
        for column in first_feature_values
        if column not in varying_features
    )

    class_distribution = dict(label_counts)
    class_percentages = {
        label: _percentage(count, total_rows)
        for label, count in class_distribution.items()
    }
    feature_label_conflicts = _summarize_feature_label_conflicts(
        feature_hashes,
        label_hashes,
        list(class_distribution),
    )

    dtype_report = {
        column: sorted(dtypes)
        for column, dtypes in observed_dtypes.items()
    }
    dtype_variations = {
        column: dtypes
        for column, dtypes in dtype_report.items()
        if len(dtypes) > 1
    }

    duplicate_comparisons = list(duplicate_column_stats.values())

    file_report = {
        "file_name": file_path.name,
        "row_count": total_rows,
        "column_count": len(raw_columns),
        "columns": normalized_columns,
        "raw_columns": raw_columns,
        "columns_with_whitespace": columns_with_whitespace,
        "duplicate_header_names": {
            name: [position + 1 for position in positions]
            for name, positions in duplicate_header_groups.items()
        },
        "duplicate_column_comparisons": duplicate_comparisons,
        "data_types": dtype_report,
        "data_type_variations": dtype_variations,
        "missing_values": dict(missing_counts),
        "infinite_values": dict(infinite_counts),
        "total_missing": int(sum(missing_counts.values())),
        "total_infinite": int(sum(infinite_counts.values())),
        "all_missing_columns": all_missing_columns,
        "constant_features": constant_features,
        "numeric_statistics": numeric_statistics,
        "negative_value_summary": _negative_value_summary(numeric_statistics),
        "class_distribution": class_distribution,
        "feature_label_conflicts": feature_label_conflicts,
        "class_distribution_percent": class_percentages,
        "label_quality": {
            "missing_or_blank_rows": label_missing_rows,
            "rows_with_leading_or_trailing_whitespace": label_whitespace_rows,
            "rows_with_replacement_character": replacement_character_rows,
            "replacement_character_count": replacement_character_count,
            "rows_with_mojibake_markers": mojibake_rows,
            "mojibake_marker_row_counts": dict(mojibake_marker_counts),
        },
        "encoding_audit": {
            "selected_encoding": CSV_ENCODING,
            "decode_errors": "strict",
            "decode_successful": True,
            "byte_order_mark": _detect_bom(file_path),
        },
        "duplicate_rows": {
            "method": "64-bit hashes of raw CSV cell strings",
            "unique_row_hashes": int(unique_row_hashes.size),
            "duplicate_rows_within_file": int(total_rows - unique_row_hashes.size),
        },
        "duplicate_feature_vectors_excluding_label": {
            "method": "64-bit hashes of raw CSV cell strings, excluding Label",
            "unique_feature_hashes": int(unique_feature_hashes.size),
            "duplicate_feature_rows_within_file": int(
                total_rows - unique_feature_hashes.size
            ),
        },
    }

    constant_feature_profile = {
        column: {
            "has_non_missing_values": column in first_feature_values,
            "first_value": first_feature_values.get(column),
            "varies_within_file": column in varying_features,
        }
        for column in internal_columns
        if column != label_internal_name
    }

    return (
        file_report,
        unique_row_hashes,
        unique_feature_hashes,
        constant_feature_profile,
        feature_hashes,
        label_hashes,
    )


def audit_csv(file_path: Path, chunksize: int = DEFAULT_CHUNKSIZE) -> dict:
    """Inspect one CSV incrementally without modifying the source file."""

    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")

    report, _, _, _, _, _ = _audit_csv(file_path, chunksize)
    return report


def audit_raw_directory(
    raw_dir: Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> dict:
    """Audit all raw CIC-IDS2017 CSV files, excluding the Global Test Set."""

    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw data directory not found: {raw_dir}")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")

    available_csv_files = {
        file_path.name.casefold(): file_path
        for file_path in raw_dir.glob("*.csv")
    }
    missing_expected_files = sorted(
        EXPECTED_RAW_FILES - set(available_csv_files)
    )
    if missing_expected_files:
        raise FileNotFoundError(
            "Missing expected CIC-IDS2017 raw files: "
            + ", ".join(missing_expected_files)
        )

    csv_files = [
        available_csv_files[name]
        for name in sorted(EXPECTED_RAW_FILES)
    ]

    file_reports: list[dict] = []
    global_row_unique_hashes: list[np.ndarray] = []
    global_feature_unique_hashes: list[np.ndarray] = []
    global_feature_hash_parts: list[np.ndarray] = []
    global_label_hash_parts: list[np.ndarray] = []
    constant_feature_profiles: list[dict[str, dict]] = []

    for file_path in csv_files:
        (
            report,
            unique_row_hashes,
            unique_feature_hashes,
            constant_feature_profile,
            feature_hashes,
            label_hashes,
        ) = _audit_csv(file_path, chunksize)
        file_reports.append(report)
        global_row_unique_hashes.append(unique_row_hashes)
        global_feature_unique_hashes.append(unique_feature_hashes)
        global_feature_hash_parts.append(feature_hashes)
        global_label_hash_parts.append(label_hashes)
        constant_feature_profiles.append(constant_feature_profile)

    reference_columns = file_reports[0]["columns"]
    reference_raw_columns = file_reports[0]["raw_columns"]
    schema_differences = {}

    reference_counter = Counter(reference_columns)
    for report in file_reports:
        current_columns = report["columns"]
        current_counter = Counter(current_columns)

        schema_differences[report["file_name"]] = {
            "missing_columns": sorted(
                (reference_counter - current_counter).elements()
            ),
            "extra_columns": sorted(
                (current_counter - reference_counter).elements()
            ),
            "order_matches_reference": current_columns == reference_columns,
            "raw_header_matches_reference": (
                report["raw_columns"] == reference_raw_columns
            ),
        }

    aggregate_labels: Counter = Counter()
    aggregate_missing: Counter = Counter()
    aggregate_infinite: Counter = Counter()
    aggregate_dtypes: dict[str, set[str]] = defaultdict(set)
    aggregate_numeric: dict[str, dict[str, float | int | None]] = {}
    aggregate_mojibake_markers: Counter = Counter()

    total_label_missing = 0
    total_label_whitespace = 0
    total_replacement_rows = 0
    total_replacement_characters = 0
    total_mojibake_rows = 0

    for report in file_reports:
        aggregate_labels.update(report["class_distribution"])
        aggregate_missing.update(report["missing_values"])
        aggregate_infinite.update(report["infinite_values"])

        for column, dtypes in report["data_types"].items():
            aggregate_dtypes[column].update(dtypes)

        for column, stats in report["numeric_statistics"].items():
            overall = aggregate_numeric.setdefault(
                column,
                {
                    "minimum": None,
                    "maximum": None,
                    "zero_count": 0,
                    "negative_count": 0,
                    "finite_value_count": 0,
                },
            )
            if stats["minimum"] is not None:
                overall["minimum"] = (
                    stats["minimum"]
                    if overall["minimum"] is None
                    else min(float(overall["minimum"]), float(stats["minimum"]))
                )
            if stats["maximum"] is not None:
                overall["maximum"] = (
                    stats["maximum"]
                    if overall["maximum"] is None
                    else max(float(overall["maximum"]), float(stats["maximum"]))
                )
            for count_name in ("zero_count", "negative_count", "finite_value_count"):
                overall[count_name] += int(stats[count_name])

        label_quality = report["label_quality"]
        total_label_missing += label_quality["missing_or_blank_rows"]
        total_label_whitespace += label_quality[
            "rows_with_leading_or_trailing_whitespace"
        ]
        total_replacement_rows += label_quality["rows_with_replacement_character"]
        total_replacement_characters += label_quality["replacement_character_count"]
        total_mojibake_rows += label_quality["rows_with_mojibake_markers"]
        aggregate_mojibake_markers.update(
            label_quality["mojibake_marker_row_counts"]
        )

    total_rows = sum(report["row_count"] for report in file_reports)
    total_missing = sum(report["total_missing"] for report in file_reports)
    total_infinite = sum(report["total_infinite"] for report in file_reports)

    all_row_hashes = (
        np.concatenate(global_row_unique_hashes)
        if global_row_unique_hashes
        else np.array([], dtype=np.uint64)
    )
    all_feature_hashes = (
        np.concatenate(global_feature_unique_hashes)
        if global_feature_unique_hashes
        else np.array([], dtype=np.uint64)
    )
    all_feature_row_hashes = (
        np.concatenate(global_feature_hash_parts)
        if global_feature_hash_parts
        else np.array([], dtype=np.uint64)
    )
    all_label_row_hashes = (
        np.concatenate(global_label_hash_parts)
        if global_label_hash_parts
        else np.array([], dtype=np.uint64)
    )

    global_unique_rows = int(np.unique(all_row_hashes).size)
    global_unique_feature_vectors = int(np.unique(all_feature_hashes).size)

    within_file_duplicate_rows = sum(
        report["duplicate_rows"]["duplicate_rows_within_file"]
        for report in file_reports
    )
    within_file_duplicate_features = sum(
        report["duplicate_feature_vectors_excluding_label"][
            "duplicate_feature_rows_within_file"
        ]
        for report in file_reports
    )

    per_file_unique_rows = sum(
        report["duplicate_rows"]["unique_row_hashes"]
        for report in file_reports
    )
    per_file_unique_features = sum(
        report["duplicate_feature_vectors_excluding_label"][
            "unique_feature_hashes"
        ]
        for report in file_reports
    )

    schema_consistent = all(
        report["columns"] == reference_columns for report in file_reports
    )
    raw_schema_consistent = all(
        report["raw_columns"] == reference_raw_columns
        for report in file_reports
    )

    overall_class_distribution = dict(aggregate_labels)
    overall_class_percentages = {
        label: _percentage(count, total_rows)
        for label, count in overall_class_distribution.items()
    }
    overall_feature_label_conflicts = _summarize_feature_label_conflicts(
        all_feature_row_hashes,
        all_label_row_hashes,
        list(overall_class_distribution),
    )

    overall_dtype_report = {
        column: sorted(dtypes)
        for column, dtypes in aggregate_dtypes.items()
    }
    overall_dtype_variations = {
        column: dtypes
        for column, dtypes in overall_dtype_report.items()
        if len(dtypes) > 1
    }

    first_global_feature_values: dict[str, object] = {}
    varying_global_features: set[str] = set()
    for profile in constant_feature_profiles:
        for column, details in profile.items():
            if not details["has_non_missing_values"]:
                continue

            current_value = details["first_value"]
            if column not in first_global_feature_values:
                first_global_feature_values[column] = current_value
            elif current_value != first_global_feature_values[column]:
                varying_global_features.add(column)

            if details["varies_within_file"]:
                varying_global_features.add(column)

    overall_constant_features = sorted(
        column
        for column in first_global_feature_values
        if column not in varying_global_features
    )
    overall_all_missing_columns = sorted(
        column
        for column, count in aggregate_missing.items()
        if count == total_rows and total_rows > 0
    )

    all_csv_names = {
        file_path.name.casefold(): file_path.name
        for file_path in raw_dir.glob("*.csv")
    }
    unexpected_csv_files = sorted(
        original_name
        for normalized_name, original_name in all_csv_names.items()
        if normalized_name not in EXPECTED_RAW_FILES
        and normalized_name not in EXCLUDED_FILES
    )

    return {
        "dataset": "CIC-IDS2017",
        "files_audited": len(file_reports),
        "expected_files": len(EXPECTED_RAW_FILES),
        "audited_files": [report["file_name"] for report in file_reports],
        "excluded_files": [
            file_path.name
            for file_path in sorted(raw_dir.glob("*.csv"))
            if file_path.name.casefold() in EXCLUDED_FILES
        ],
        "unexpected_csv_files_not_audited": unexpected_csv_files,
        "total_rows": total_rows,
        "reference_columns": reference_columns,
        "reference_raw_columns": reference_raw_columns,
        "schema_consistent": schema_consistent,
        "raw_schema_consistent": raw_schema_consistent,
        "schema_differences": schema_differences,
        "overall_class_distribution": overall_class_distribution,
        "overall_class_distribution_percent": overall_class_percentages,
        "total_missing": total_missing,
        "overall_missing_values": dict(aggregate_missing),
        "total_infinite": total_infinite,
        "overall_infinite_values": dict(aggregate_infinite),
        "overall_data_types": overall_dtype_report,
        "overall_data_type_variations": overall_dtype_variations,
        "overall_numeric_statistics": aggregate_numeric,
        "overall_negative_value_summary": _negative_value_summary(aggregate_numeric),
        "overall_feature_label_conflicts": overall_feature_label_conflicts,
        "overall_constant_features": overall_constant_features,
        "overall_all_missing_columns": overall_all_missing_columns,
        "label_quality": {
            "missing_or_blank_rows": total_label_missing,
            "rows_with_leading_or_trailing_whitespace": total_label_whitespace,
            "rows_with_replacement_character": total_replacement_rows,
            "replacement_character_count": total_replacement_characters,
            "rows_with_mojibake_markers": total_mojibake_rows,
            "mojibake_marker_row_counts": dict(aggregate_mojibake_markers),
        },
        "duplicate_analysis": {
            "hash_method": "64-bit hashes of raw CSV cell strings; hash collisions are theoretically possible.",
            "full_rows": {
                "unique_rows": global_unique_rows,
                "total_duplicate_rows": total_rows - global_unique_rows,
                "duplicate_rows_within_files": within_file_duplicate_rows,
                "duplicate_rows_across_files": per_file_unique_rows - global_unique_rows,
            },
            "feature_vectors_excluding_label": {
                "unique_feature_vectors": global_unique_feature_vectors,
                "total_duplicate_feature_rows": (
                    total_rows - global_unique_feature_vectors
                ),
                "duplicate_feature_rows_within_files": within_file_duplicate_features,
                "duplicate_feature_rows_across_files": (
                    per_file_unique_features - global_unique_feature_vectors
                ),
                "note": "Identical feature vectors may have different labels; this metric does not imply identical full rows.",
            },
        },
        "audit_settings": {
            "chunksize": chunksize,
            "encoding": CSV_ENCODING,
            "encoding_errors": "strict",
            "source_files_modified": False,
        },
        "files": file_reports,
    }
