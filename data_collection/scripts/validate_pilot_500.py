"""Validate the deliverable contract for dataset-v0.1 / PILOT_500.

This complements the generic YOLO validator with the exact source and split
allocation agreed for the provisional training pilot.  It does not claim that
the external cross-test is a locked K230 evaluation set.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from validate_yolo_dataset import validate_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "dataset_config.yaml"
REQUIRED_SOURCE_COUNTS = {
    "MIO-TCD Localization": 170,
    "AAU RainSnow": 165,
    "UA-DETRAC Original": 165,
}
REQUIRED_SPLIT_COUNTS = {"train": 421, "val": 51, "cross_test": 28}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_pilot_500(dataset: Path, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Return the combined generic-YOLO and PILOT_500 release report."""
    report = validate_dataset(dataset)
    errors = list(report["errors"])
    try:
        config = _load_yaml(config_path)
    except (OSError, yaml.YAMLError) as exc:
        config = {}
        errors.append(f"INVALID_PILOT_CONFIG:{exc}")
    if config.get("dataset_version") != "dataset-v0.1":
        errors.append("INVALID_DATASET_VERSION")
    if config.get("release_profile") != "PILOT_500":
        errors.append("INVALID_RELEASE_PROFILE")
    if config.get("target_total_images") != 500:
        errors.append("INVALID_PILOT_TARGET_TOTAL")
    if config.get("expected_images_by_source") != REQUIRED_SOURCE_COUNTS:
        errors.append("INVALID_PILOT_SOURCE_ALLOCATION")
    expected_config_splits = {
        "train": config.get("target_train_images"),
        "val": config.get("target_val_images"),
        "cross_test": config.get("target_cross_test_images"),
    }
    if expected_config_splits != REQUIRED_SPLIT_COUNTS:
        errors.append("INVALID_PILOT_SPLIT_ALLOCATION")

    image_rows = _read_rows(dataset / "metadata" / "images.csv")
    source_counts = Counter(row.get("dataset", "") for row in image_rows)
    split_counts = Counter(row.get("split", "") for row in image_rows)
    if len(image_rows) != 500:
        errors.append(f"PILOT_IMAGE_COUNT_MISMATCH:{len(image_rows)}")
    if dict(sorted(source_counts.items())) != REQUIRED_SOURCE_COUNTS:
        errors.append(f"PILOT_SOURCE_COUNT_MISMATCH:{dict(sorted(source_counts.items()))}")
    if dict(sorted(split_counts.items())) != REQUIRED_SPLIT_COUNTS:
        errors.append(f"PILOT_SPLIT_COUNT_MISMATCH:{dict(sorted(split_counts.items()))}")

    summary_path = dataset / "metadata" / "export_summary.json"
    summary: dict[str, Any] = {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"INVALID_PILOT_EXPORT_SUMMARY:{exc}")
    selection = summary.get("selection", {}) if isinstance(summary, dict) else {}
    if selection.get("selection_subset") != "PILOT_500":
        errors.append("MISSING_PILOT_SELECTION_PROVENANCE")
    if selection.get("selection_image_count") != 500:
        errors.append("INVALID_PILOT_SELECTION_COUNT")
    if selection.get("selection_images_by_dataset") != REQUIRED_SOURCE_COUNTS:
        errors.append("INVALID_PILOT_SELECTION_SOURCE_ALLOCATION")
    if not (dataset / "metadata" / "selection_manifest.csv").is_file():
        errors.append("MISSING_PILOT_SELECTION_MANIFEST")

    return {
        **report,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "release_profile": "PILOT_500",
        "expected_images_by_source": REQUIRED_SOURCE_COUNTS,
        "actual_images_by_source": dict(sorted(source_counts.items())),
        "expected_images_by_split": REQUIRED_SPLIT_COUNTS,
        "actual_images_by_split": dict(sorted(split_counts.items())),
        "config": str(config_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_pilot_500(args.dataset.resolve(), args.config.resolve())
    report_path = args.report or args.dataset / "metadata" / "pilot_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
