"""Validate a unified YOLO dataset before training.

Checks image/label pairing, one-class labels, normalized non-empty boxes,
sequence split leakage, and export-count reconciliation.  It exits non-zero on
any failed gate and writes a machine-readable report next to the dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "cross_test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not directory.exists():
        return result
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in suffixes:
            result[item.stem] = item
    return result


def validate_dataset(dataset: Path) -> dict[str, Any]:
    """Return the complete validation report without writing media files."""
    errors: list[str] = []
    images_by_split: Counter[str] = Counter()
    labels_by_split: Counter[str] = Counter()
    boxes_by_split: Counter[str] = Counter()
    total_boxes = 0
    empty_labels: list[tuple[str, str]] = []
    for split in SPLITS:
        images = _files(dataset / "images" / split, IMAGE_SUFFIXES)
        labels = _files(dataset / "labels" / split, {".txt"})
        images_by_split[split] = len(images)
        labels_by_split[split] = len(labels)
        for stem in sorted(images.keys() - labels.keys()):
            errors.append(f"MISSING_LABEL:{split}/{stem}")
        for stem in sorted(labels.keys() - images.keys()):
            errors.append(f"MISSING_IMAGE:{split}/{stem}")
        for stem, label_path in labels.items():
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                errors.append(f"UNREADABLE_LABEL:{split}/{stem}:{exc}")
                continue
            if not any(line.strip() for line in lines):
                empty_labels.append((split, stem))
            for line_number, line in enumerate(lines, start=1):
                values = line.split()
                if len(values) != 5:
                    errors.append(f"MALFORMED_LABEL:{split}/{stem}:{line_number}")
                    continue
                if values[0] != "0":
                    errors.append(f"INVALID_CLASS_ID:{split}/{stem}:{line_number}:{values[0]}")
                    continue
                try:
                    center_x, center_y, width, height = (float(value) for value in values[1:])
                except ValueError:
                    errors.append(f"NON_NUMERIC_YOLO:{split}/{stem}:{line_number}")
                    continue
                coordinates = (center_x, center_y, width, height)
                if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates):
                    errors.append(f"YOLO_OUT_OF_RANGE:{split}/{stem}:{line_number}")
                if width <= 0.0 or height <= 0.0:
                    errors.append(f"EMPTY_OR_NEGATIVE_BOX:{split}/{stem}:{line_number}")
                total_boxes += 1
                boxes_by_split[split] += 1
    negative_manifest = dataset / "metadata" / "negative_samples.csv"
    registered_negatives: set[tuple[str, str]] = set()
    if negative_manifest.exists():
        for row in _csv_rows(negative_manifest):
            split, image_id = str(row.get("split", "")), str(row.get("image_id", ""))
            if not split or not image_id or split not in SPLITS:
                errors.append(f"INVALID_NEGATIVE_MANIFEST_ROW:{split}/{image_id}")
                continue
            registered_negatives.add((split, image_id))
        empty_set = set(empty_labels)
        for split, image_id in sorted(registered_negatives - empty_set):
            errors.append(f"NEGATIVE_MANIFEST_NOT_EMPTY_LABEL:{split}/{image_id}")
        for split, image_id in sorted(empty_set - registered_negatives):
            errors.append(f"UNREGISTERED_EMPTY_LABEL:{split}/{image_id}")
    sequence_rows = _csv_rows(dataset / "metadata" / "sequence_splits.csv")
    sequence_split: dict[tuple[str, str], str] = {}
    for row in sequence_rows:
        dataset_name, sequence_id, split = row.get("dataset", ""), row.get("sequence_id", ""), row.get("split", "")
        if not dataset_name or not sequence_id or split not in SPLITS:
            errors.append(f"INVALID_SEQUENCE_MANIFEST_ROW:{dataset_name}/{sequence_id}")
            continue
        key = (dataset_name, sequence_id)
        prior = sequence_split.get(key)
        if prior and prior != split:
            errors.append(f"SEQUENCE_LEAKAGE:{dataset_name}/{sequence_id}:{prior}|{split}")
        sequence_split[key] = split
    annotation_rows = _csv_rows(dataset / "metadata" / "annotations.csv")
    invalid_manifest_classes = [row for row in annotation_rows if row.get("class_id") != "0" or row.get("mapped_class") != "vehicle" or not row.get("original_class")]
    if invalid_manifest_classes:
        errors.append(f"INVALID_ANNOTATION_MANIFEST:{len(invalid_manifest_classes)}")
    summary_path = dataset / "metadata" / "export_summary.json"
    summary: dict[str, Any] = {}
    if not summary_path.exists():
        errors.append("MISSING_EXPORT_SUMMARY")
    else:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"INVALID_EXPORT_SUMMARY:{exc}")
    counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
    if counts:
        expected_input = int(counts.get("input_annotations", 0))
        expected_exported = int(counts.get("exported_boxes", 0))
        expected_rejected = int(counts.get("rejected_annotations", 0))
        expected_ignored = int(counts.get("ignored_annotations", 0))
        ignored_rows = _csv_rows(dataset / "metadata" / "ignored_annotations.csv")
        invalid_ignored = [
            row for row in ignored_rows
            if row.get("handling") != "IGNORE_REGION"
            or row.get("reason") != "NON_TARGET_TWO_WHEEL_OR_BICYCLE"
            or not row.get("original_class")
        ]
        if invalid_ignored:
            errors.append(f"INVALID_IGNORED_ANNOTATION_MANIFEST:{len(invalid_ignored)}")
        if expected_input != expected_exported + expected_rejected + expected_ignored:
            errors.append("SUMMARY_ANNOTATION_RECONCILIATION_FAILED")
        if expected_ignored != len(ignored_rows):
            errors.append(
                f"IGNORED_ANNOTATION_COUNT_MISMATCH:summary={expected_ignored},manifest={len(ignored_rows)}"
            )
        if int(counts.get("input_images", 0)) != int(counts.get("exported_images", 0)):
            errors.append("SUMMARY_IMAGE_RECONCILIATION_FAILED")
        if expected_exported != total_boxes or expected_exported != len(annotation_rows):
            errors.append(f"EXPORTED_BOX_COUNT_MISMATCH:summary={expected_exported},labels={total_boxes},manifest={len(annotation_rows)}")
    for split in SPLITS:
        if counts and int(summary.get("images_by_split", {}).get(split, 0)) != images_by_split[split]:
            errors.append(f"IMAGE_COUNT_MISMATCH:{split}")
        if counts and int(summary.get("boxes_by_split", {}).get(split, 0)) != boxes_by_split[split]:
            errors.append(f"BOX_COUNT_MISMATCH:{split}")
    return {
        "dataset": str(dataset.resolve()), "status": "PASS" if not errors else "FAIL", "errors": errors,
        "images_by_split": dict(images_by_split), "labels_by_split": dict(labels_by_split),
        "boxes_by_split": dict(boxes_by_split), "total_boxes": total_boxes,
        "annotation_manifest_count": len(annotation_rows), "sequence_count": len(sequence_split),
        "ignored_annotation_count": len(_csv_rows(dataset / "metadata" / "ignored_annotations.csv")),
        "empty_label_count": len(empty_labels),
        "registered_negative_count": len(registered_negatives),
        "negative_sample_convention": "empty_txt_with_negative_samples_manifest" if negative_manifest.exists() else "empty_txt_allowed_manifest_not_present",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_dataset(args.dataset.resolve())
    report_path = args.report or args.dataset / "metadata" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
