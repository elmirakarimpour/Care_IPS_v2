
"""Data contracts and schema definitions for CARE-IPS."""

from dataclasses import dataclass
from typing import Final


# ---------------------------------------------------------------------
# Contract version
# ---------------------------------------------------------------------

DATA_CONTRACT_VERSION: Final[str] = "1.0"


# ---------------------------------------------------------------------
# Target label definitions
# ---------------------------------------------------------------------

LABEL_COLUMN: Final[str] = "Label"
RAW_LABEL_COLUMN: Final[str] = "label_raw"


# ---------------------------------------------------------------------
# Record identification and provenance
# ---------------------------------------------------------------------

SAMPLE_ID_COLUMN: Final[str] = "sample_id"
SOURCE_FILE_COLUMN: Final[str] = "source_file"
SOURCE_ROW_NUMBER_COLUMN: Final[str] = "source_row_number"

TRACEABILITY_COLUMNS: Final[tuple[str, ...]] = (
    SAMPLE_ID_COLUMN,
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_NUMBER_COLUMN,
)

# Metadata added by the loader, including the preserved original label.
LOADER_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    *TRACEABILITY_COLUMNS,
    RAW_LABEL_COLUMN,
)

# Record numbers start at 1; the CSV header is not counted.
SOURCE_ROW_NUMBER_BASE: Final[int] = 1


# ---------------------------------------------------------------------
# Model input restrictions
# ---------------------------------------------------------------------

NON_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset(
    (*LOADER_METADATA_COLUMNS, LABEL_COLUMN)
)


# ---------------------------------------------------------------------
# CIC-IDS2017 dataset schema
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSchema:
    """Expected structural properties of a source dataset."""

    name: str
    source_files: tuple[str, ...]
    raw_column_count: int
    unique_source_column_count: int
    feature_count: int
    duplicate_column_names: tuple[str, ...]

    @property
    def expected_loader_column_count(self) -> int:
        """Number of columns after loading and adding metadata."""

        return self.raw_column_count + len(LOADER_METADATA_COLUMNS)

    @property
    def expected_processed_column_count(self) -> int:
        """Number of columns after removing duplicate source features."""

        return self.unique_source_column_count + len(LOADER_METADATA_COLUMNS)


CIC_IDS2017_SCHEMA: Final[DatasetSchema] = DatasetSchema(
    name="CIC-IDS2017",
    source_files=(
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
    ),
    raw_column_count=79,
    unique_source_column_count=78,
    feature_count=77,
    duplicate_column_names=("Fwd Header Length",),
)


# ---------------------------------------------------------------------
# Stage-specific data policies
# ---------------------------------------------------------------------

# Raw loading must preserve missing and infinite values.
RAW_MISSING_VALUES_ALLOWED: Final[bool] = True
RAW_INFINITE_VALUES_ALLOWED: Final[bool] = True
RAW_DUPLICATE_COLUMNS_ALLOWED: Final[bool] = True

# Processed data must have unique column names.
PROCESSED_DUPLICATE_COLUMNS_ALLOWED: Final[bool] = False
