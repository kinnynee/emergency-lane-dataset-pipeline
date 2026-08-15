"""Fail-closed QC for a materialized unified YOLO dataset.

The report is intentionally generated from files that were written to disk,
not from pre-filter exporter counters. Empty ``.txt`` labels are valid
negative samples; a missing ``.txt`` for an image is always an error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from yolo_export_qc import audit_yolo_dataset, write_report


def validate_dataset(dataset: Path) -> dict[str, Any]:
    """Return the complete fail-closed QC report without writing media files."""
    report = audit_yolo_dataset(dataset, require_summary=True)
    # Backwards-compatible aliases used by existing release checks.
    report["total_boxes"] = report["box_count_actual"]
    report["sequence_count"] = 0
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_dataset(args.dataset.resolve())
    report_path = args.report or args.dataset / "metadata" / "qc_report.json"
    write_report(report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
