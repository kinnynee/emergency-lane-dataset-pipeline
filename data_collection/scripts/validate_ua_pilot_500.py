"""Validate the temporary UA-DETRAC-only 500-image training pilot."""

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
DEFAULT_CONFIG = ROOT / "configs" / "pilot_500_ua_only.yaml"
SOURCE = "UA-DETRAC Original"
EXPECTED_SPLITS = {"train": 355, "val": 105, "cross_test": 40}


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_ua_pilot_500(dataset: Path, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    report = validate_dataset(dataset)
    errors = list(report["errors"])
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        config = {}
        errors.append(f"INVALID_PILOT_CONFIG:{exc}")
    if config.get("dataset_version") != "dataset-v0.1" or config.get("release_profile") != "PILOT_500_UA_ONLY":
        errors.append("INVALID_RELEASE_CONFIG")
    if config.get("target_total_images") != 500 or config.get("expected_images_by_split") != EXPECTED_SPLITS:
        errors.append("INVALID_RELEASE_ALLOCATION")

    rows = _rows(dataset / "metadata" / "images.csv")
    source_counts = Counter(row.get("dataset", "") for row in rows)
    split_counts = Counter(row.get("split", "") for row in rows)
    if len(rows) != 500:
        errors.append(f"PILOT_IMAGE_COUNT_MISMATCH:{len(rows)}")
    if dict(source_counts) != {SOURCE: 500}:
        errors.append(f"PILOT_SOURCE_COUNT_MISMATCH:{dict(source_counts)}")
    if dict(sorted(split_counts.items())) != EXPECTED_SPLITS:
        errors.append(f"PILOT_SPLIT_COUNT_MISMATCH:{dict(sorted(split_counts.items()))}")

    try:
        summary = json.loads((dataset / "metadata" / "export_summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        summary = {}
        errors.append(f"INVALID_EXPORT_SUMMARY:{exc}")
    selection = summary.get("selection", {}) if isinstance(summary, dict) else {}
    if selection.get("selection_subset") != config.get("selection_subset"):
        errors.append("INVALID_SELECTION_SUBSET")
    if selection.get("selection_image_count") != 500:
        errors.append("INVALID_SELECTION_COUNT")
    if selection.get("selection_images_by_dataset") != {SOURCE: 500}:
        errors.append("INVALID_SELECTION_SOURCE")
    if not (dataset / "metadata" / "selection_manifest.csv").is_file():
        errors.append("MISSING_SELECTION_MANIFEST")

    return {
        **report,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "release_profile": "PILOT_500_UA_ONLY",
        "actual_images_by_source": dict(source_counts),
        "actual_images_by_split": dict(sorted(split_counts.items())),
        "config": str(config_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_ua_pilot_500(args.dataset.resolve(), args.config.resolve())
    target = args.report or args.dataset / "metadata" / "pilot_validation_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
