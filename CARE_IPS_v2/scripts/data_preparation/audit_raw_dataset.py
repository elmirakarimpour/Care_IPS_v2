
"""Run the raw CIC-IDS2017 data audit."""

import json

from care_ips.config import RAW_DATA_DIR, RESULTS_DIR
from care_ips.data.audit import audit_raw_directory


def main() -> None:
    """Audit raw data and save a JSON report."""

    report = audit_raw_directory(RAW_DATA_DIR)

    output_dir = RESULTS_DIR / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "raw_data_audit.json"

    output_path.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Files audited: {report['files_audited']}")
    print(f"Total rows: {report['total_rows']:,}")
    print(f"Schema consistent: {report['schema_consistent']}")
    print(f"Classes: {report['overall_class_distribution']}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
