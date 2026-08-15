"""Recount and quality-control a materialized unified YOLO export.

The functions in this module deliberately inspect the written ``images`` and
``labels`` trees.  They never use exporter counters as evidence of what was
actually released.  This makes the module safe to use both when investigating
an older export and as the last step of a new export.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "val", "cross_test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _csv_rows(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    if not path.is_file():
        return []
    def rows() -> Iterable[tuple[int, dict[str, str]]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), start=2):
                yield line_number, row
    return rows()


def _int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_key(split: str, image_id: str) -> tuple[str, str]:
    return (str(split).strip(), str(image_id).strip())


def _asset_index(root: Path, suffixes: set[str]) -> tuple[dict[tuple[str, str], Path], dict[str, list[str]], list[str]]:
    """Return unique assets, duplicate IDs, and unexpected files below a tree."""
    assets: dict[tuple[str, str], Path] = {}
    ids: defaultdict[str, list[str]] = defaultdict(list)
    unexpected: list[str] = []
    if not root.is_dir():
        return assets, {}, [f"MISSING_DIRECTORY:{root}"]
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(root)
        if len(relative.parts) != 2 or relative.parts[0] not in SPLITS or item.suffix.lower() not in suffixes:
            unexpected.append(relative.as_posix())
            continue
        split, image_id = relative.parts[0], item.stem
        key = _as_key(split, image_id)
        ids[image_id].append(relative.as_posix())
        if key in assets:
            # Keep the first path so the pair audit stays deterministic.
            continue
        assets[key] = item
    duplicate_ids = {image_id: paths for image_id, paths in ids.items() if len(paths) > 1}
    return assets, duplicate_ids, unexpected


def _append_error(errors: list[str], code: str, detail: str) -> None:
    errors.append(f"{code}:{detail}" if detail else code)


def _summary_count(summary: dict[str, Any], key: str) -> int | None:
    counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
    if not isinstance(counts, dict) or key not in counts:
        return None
    return _int(counts.get(key))


def _read_summary(path: Path, errors: list[str], require_summary: bool) -> dict[str, Any]:
    if not path.is_file():
        if require_summary:
            _append_error(errors, "MISSING_EXPORT_SUMMARY", "")
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _append_error(errors, "INVALID_EXPORT_SUMMARY", str(exc))
        return {}
    if not isinstance(loaded, dict):
        _append_error(errors, "INVALID_EXPORT_SUMMARY", "not_an_object")
        return {}
    return loaded


def audit_yolo_dataset(dataset: Path, *, require_summary: bool = True) -> dict[str, Any]:
    """Inspect a final YOLO tree and return a detailed, machine-readable QC report.

    Counts in this return value are based on files in ``images/`` and
    ``labels/``.  Metadata is treated as a separate claim that must reconcile
    with those files, never as a source of truth.
    """
    dataset = dataset.resolve()
    metadata = dataset / "metadata"
    errors: list[str] = []

    images, duplicate_images, unexpected_images = _asset_index(dataset / "images", IMAGE_SUFFIXES)
    labels, duplicate_labels, unexpected_labels = _asset_index(dataset / "labels", {".txt"})
    if not (dataset / "images").is_dir():
        _append_error(errors, "MISSING_IMAGES_DIRECTORY", "")
    if not (dataset / "labels").is_dir():
        _append_error(errors, "MISSING_LABELS_DIRECTORY", "")
    for image_id, paths in sorted(duplicate_images.items()):
        _append_error(errors, "DUPLICATE_IMAGE_ID", f"{image_id}:{'|'.join(paths)}")
    for image_id, paths in sorted(duplicate_labels.items()):
        _append_error(errors, "DUPLICATE_LABEL_ID", f"{image_id}:{'|'.join(paths)}")
    for path in unexpected_images:
        _append_error(errors, "UNEXPECTED_IMAGE_PATH", path)
    for path in unexpected_labels:
        _append_error(errors, "UNEXPECTED_LABEL_PATH", path)

    image_keys = set(images)
    label_keys = set(labels)
    missing_labels = sorted(image_keys - label_keys)
    missing_images = sorted(label_keys - image_keys)
    for split, image_id in missing_labels:
        _append_error(errors, "MISSING_LABEL", f"{split}/{image_id}")
    for split, image_id in missing_images:
        _append_error(errors, "MISSING_IMAGE", f"{split}/{image_id}")

    boxes_by_split: Counter[str] = Counter()
    label_box_counts: Counter[tuple[str, str]] = Counter()
    empty_labels: list[dict[str, str]] = []
    invalid_boxes: list[dict[str, Any]] = []
    unknown_classes: list[dict[str, Any]] = []
    duplicate_boxes: list[dict[str, Any]] = []
    malformed_labels: list[dict[str, Any]] = []
    for (split, image_id), label_path in sorted(labels.items()):
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            _append_error(errors, "UNREADABLE_LABEL", f"{split}/{image_id}:{exc}")
            invalid_boxes.append({"split": split, "image_id": image_id, "path": str(label_path), "reason": "UNREADABLE_LABEL"})
            continue
        nonempty = [line for line in lines if line.strip()]
        if not nonempty:
            empty_labels.append({"split": split, "image_id": image_id, "path": label_path.relative_to(dataset).as_posix()})
        seen: set[tuple[str, float, float, float, float]] = set()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            values = line.split()
            location = {"split": split, "image_id": image_id, "path": label_path.relative_to(dataset).as_posix(), "line": line_number}
            if len(values) != 5:
                malformed_labels.append({**location, "reason": "FORMAT", "value": line})
                _append_error(errors, "MALFORMED_LABEL", f"{split}/{image_id}:{line_number}")
                continue
            class_id = values[0]
            if class_id != "0":
                unknown_classes.append({**location, "class_id": class_id})
                _append_error(errors, "INVALID_CLASS_ID", f"{split}/{image_id}:{line_number}:{class_id}")
                continue
            try:
                coords = tuple(float(value) for value in values[1:])
            except ValueError:
                malformed_labels.append({**location, "reason": "NON_NUMERIC", "value": line})
                _append_error(errors, "NON_NUMERIC_YOLO", f"{split}/{image_id}:{line_number}")
                continue
            if not all(math.isfinite(value) for value in coords):
                invalid_boxes.append({**location, "reason": "NAN_OR_INFINITY", "value": line})
                _append_error(errors, "YOLO_OUT_OF_RANGE", f"{split}/{image_id}:{line_number}")
                continue
            center_x, center_y, width, height = coords
            if not all(0.0 <= value <= 1.0 for value in coords) or width <= 0.0 or height <= 0.0:
                reason = "OUT_OF_RANGE" if not all(0.0 <= value <= 1.0 for value in coords) else "EMPTY_OR_NEGATIVE_BOX"
                invalid_boxes.append({**location, "reason": reason, "value": line})
                _append_error(errors, "YOLO_OUT_OF_RANGE" if reason == "OUT_OF_RANGE" else "EMPTY_OR_NEGATIVE_BOX", f"{split}/{image_id}:{line_number}")
                continue
            box = (class_id, center_x, center_y, width, height)
            # A duplicate is still a syntactically valid line in the final
            # label file, so include it in the direct count and fail QC
            # separately for its semantic duplication.
            boxes_by_split[split] += 1
            label_box_counts[(split, image_id)] += 1
            if box in seen:
                duplicate_boxes.append({**location, "value": line})
                _append_error(errors, "DUPLICATE_BBOX", f"{split}/{image_id}:{line_number}")
                continue
            seen.add(box)

    images_csv = metadata / "images.csv"
    manifest_images: dict[tuple[str, str], dict[str, str]] = {}
    manifest_image_duplicates: list[dict[str, Any]] = []
    manifest_image_rows = 0
    if not images_csv.is_file():
        _append_error(errors, "MISSING_IMAGES_MANIFEST", "")
    else:
        for line_number, row in _csv_rows(images_csv):
            manifest_image_rows += 1
            key = _as_key(row.get("split", ""), row.get("image_id", ""))
            if not key[0] or not key[1] or key[0] not in SPLITS:
                _append_error(errors, "INVALID_IMAGE_MANIFEST_ROW", str(line_number))
                continue
            if key in manifest_images:
                manifest_image_duplicates.append({"split": key[0], "image_id": key[1], "line": line_number})
                _append_error(errors, "DUPLICATE_IMAGE_MANIFEST_ID", f"{key[0]}/{key[1]}")
                continue
            manifest_images[key] = row

    manifest_image_keys = set(manifest_images)
    extra_images = sorted(image_keys - manifest_image_keys)
    filtered_but_still_counted = sorted(manifest_image_keys - image_keys)
    for split, image_id in extra_images:
        _append_error(errors, "EXTRA_IMAGE", f"{split}/{image_id}")
    for split, image_id in filtered_but_still_counted:
        _append_error(errors, "FILTERED_BUT_STILL_COUNTED", f"{split}/{image_id}")

    annotations_csv = metadata / "annotations.csv"
    annotation_rows = 0
    annotation_counts_by_key: Counter[tuple[str, str]] = Counter()
    annotation_counts_by_split_source_class: Counter[tuple[str, str, str]] = Counter()
    invalid_annotation_manifest_rows: list[dict[str, Any]] = []
    if not annotations_csv.is_file():
        _append_error(errors, "MISSING_ANNOTATIONS_MANIFEST", "")
    else:
        for line_number, row in _csv_rows(annotations_csv):
            annotation_rows += 1
            split = str(row.get("split", "")).strip()
            image_id = str(row.get("image_id", "")).strip()
            dataset_name = str(row.get("dataset", "")).strip()
            original_class = str(row.get("original_class", "")).strip()
            key = _as_key(split, image_id)
            if split not in SPLITS or not image_id or row.get("class_id") != "0" or row.get("mapped_class") != "vehicle" or not original_class:
                invalid_annotation_manifest_rows.append({"line": line_number, "split": split, "image_id": image_id, "class_id": row.get("class_id", ""), "mapped_class": row.get("mapped_class", ""), "original_class": original_class})
                _append_error(errors, "INVALID_ANNOTATION_MANIFEST", str(line_number))
                continue
            annotation_counts_by_key[key] += 1
            annotation_counts_by_split_source_class[(split, dataset_name, original_class)] += 1

    per_image_box_mismatches: list[dict[str, Any]] = []
    for split, image_id in sorted(set(label_box_counts) | set(annotation_counts_by_key)):
        actual = label_box_counts[(split, image_id)]
        metadata_count = annotation_counts_by_key[(split, image_id)]
        if actual != metadata_count:
            per_image_box_mismatches.append({
                "split": split,
                "image_id": image_id,
                "actual_box_count": actual,
                "metadata_box_count": metadata_count,
                "difference": actual - metadata_count,
            })
            _append_error(errors, "BOX_COUNT_MISMATCH", f"{split}/{image_id}:actual={actual},metadata={metadata_count}")

    negative_manifest = metadata / "negative_samples.csv"
    registered_negatives: set[tuple[str, str]] = set()
    if negative_manifest.is_file():
        for line_number, row in _csv_rows(negative_manifest):
            key = _as_key(row.get("split", ""), row.get("image_id", ""))
            if key[0] not in SPLITS or not key[1]:
                _append_error(errors, "INVALID_NEGATIVE_MANIFEST_ROW", str(line_number))
                continue
            registered_negatives.add(key)
        empty_set = {_as_key(item["split"], item["image_id"]) for item in empty_labels}
        for split, image_id in sorted(registered_negatives - empty_set):
            _append_error(errors, "NEGATIVE_MANIFEST_NOT_EMPTY_LABEL", f"{split}/{image_id}")
        for split, image_id in sorted(empty_set - registered_negatives):
            _append_error(errors, "UNREGISTERED_EMPTY_LABEL", f"{split}/{image_id}")

    sequence_splits: dict[tuple[str, str], str] = {}
    for line_number, row in _csv_rows(metadata / "sequence_splits.csv"):
        dataset_name, sequence_id, split = row.get("dataset", ""), row.get("sequence_id", ""), row.get("split", "")
        if not dataset_name or not sequence_id or split not in SPLITS:
            _append_error(errors, "INVALID_SEQUENCE_MANIFEST_ROW", str(line_number))
            continue
        key = (dataset_name, sequence_id)
        prior = sequence_splits.get(key)
        if prior and prior != split:
            _append_error(errors, "SEQUENCE_LEAKAGE", f"{dataset_name}/{sequence_id}:{prior}|{split}")
        sequence_splits[key] = split

    ignored_annotation_rows = 0
    for line_number, row in _csv_rows(metadata / "ignored_annotations.csv"):
        ignored_annotation_rows += 1
        if row.get("handling") != "IGNORE_REGION" or row.get("reason") != "NON_TARGET_TWO_WHEEL_OR_BICYCLE" or not row.get("original_class"):
            _append_error(errors, "INVALID_IGNORED_ANNOTATION_MANIFEST", str(line_number))

    summary = _read_summary(metadata / "export_summary.json", errors, require_summary)
    summary_images = _summary_count(summary, "exported_images")
    summary_boxes = _summary_count(summary, "exported_boxes")
    actual_image_count = len(images)
    actual_box_count = sum(boxes_by_split.values())
    # Metadata's claimed image count is the number of rows.  Keep that
    # distinct from the unique-key index so duplicated manifest IDs cannot
    # make the count appear reconciled.
    metadata_image_count = manifest_image_rows
    metadata_box_count = annotation_rows
    if summary_images is not None and summary_images != actual_image_count:
        _append_error(errors, "SUMMARY_IMAGE_COUNT_MISMATCH", f"summary={summary_images},actual={actual_image_count}")
    if summary_boxes is not None and summary_boxes != actual_box_count:
        _append_error(errors, "SUMMARY_BOX_COUNT_MISMATCH", f"summary={summary_boxes},actual={actual_box_count}")
    summary_images_by_split = summary.get("images_by_split", {}) if isinstance(summary.get("images_by_split", {}), dict) else {}
    summary_boxes_by_split = summary.get("boxes_by_split", {}) if isinstance(summary.get("boxes_by_split", {}), dict) else {}
    for split in SPLITS:
        if summary and _int(summary_images_by_split.get(split, 0)) != len([key for key in image_keys if key[0] == split]):
            _append_error(errors, "IMAGE_COUNT_MISMATCH", split)
        if summary and _int(summary_boxes_by_split.get(split, 0)) != boxes_by_split[split]:
            _append_error(errors, "BOX_COUNT_MISMATCH", f"summary:{split}")
    expected_input = _summary_count(summary, "input_annotations")
    expected_rejected = _summary_count(summary, "rejected_annotations")
    expected_ignored = _summary_count(summary, "ignored_annotations")
    if None not in (expected_input, summary_boxes, expected_rejected):
        if expected_input != summary_boxes + expected_rejected + (expected_ignored or 0):
            _append_error(errors, "SUMMARY_ANNOTATION_RECONCILIATION_FAILED", "")
    if expected_ignored is not None and expected_ignored != ignored_annotation_rows:
        _append_error(errors, "IGNORED_ANNOTATION_COUNT_MISMATCH", f"summary={expected_ignored},manifest={ignored_annotation_rows}")

    output_counts_by_dataset: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for key, row in manifest_images.items():
        if key in image_keys:
            output_counts_by_dataset[str(row.get("dataset", ""))]["exported_images"] += 1
    for (split, dataset_name, _original_class), count in annotation_counts_by_split_source_class.items():
        output_counts_by_dataset[dataset_name]["exported_boxes"] += count
        output_counts_by_dataset[dataset_name][f"boxes_{split}"] += count
    report = {
        "dataset": str(dataset),
        "status": "PASS" if not errors else "FAIL",
        "image_count_actual": actual_image_count,
        "image_count_metadata": metadata_image_count,
        "image_count_summary": summary_images,
        "image_mismatch": abs(actual_image_count - metadata_image_count),
        "box_count_actual": actual_box_count,
        "box_count_metadata": metadata_box_count,
        "box_count_summary": summary_boxes,
        "box_mismatch": abs(actual_box_count - metadata_box_count),
        "images_by_split": {split: sum(1 for key in image_keys if key[0] == split) for split in SPLITS},
        "labels_by_split": {split: sum(1 for key in label_keys if key[0] == split) for split in SPLITS},
        "boxes_by_split": {split: boxes_by_split[split] for split in SPLITS},
        "missing_images": [{"split": split, "image_id": image_id} for split, image_id in missing_images],
        "missing_labels": [{"split": split, "image_id": image_id} for split, image_id in missing_labels],
        "duplicate_ids": {"images": duplicate_images, "labels": duplicate_labels, "metadata_images": manifest_image_duplicates},
        "invalid_boxes": invalid_boxes,
        "unknown_classes": unknown_classes,
        "empty_labels": empty_labels,
        "empty_label_count": len(empty_labels),
        "registered_negative_count": len(registered_negatives),
        "negative_sample_convention": "empty_txt_with_negative_samples_manifest" if negative_manifest.is_file() else "empty_txt_allowed_manifest_not_present",
        "malformed_labels": malformed_labels,
        "duplicate_boxes": duplicate_boxes,
        "image_mismatch_details": {
            "EXTRA_IMAGE": [{"split": split, "image_id": image_id, "path": images[(split, image_id)].relative_to(dataset).as_posix()} for split, image_id in extra_images],
            "FILTERED_BUT_STILL_COUNTED": [{"split": split, "image_id": image_id, "metadata_exported_image": manifest_images[(split, image_id)].get("exported_image", "")} for split, image_id in filtered_but_still_counted],
            "MISSING_LABEL": [{"split": split, "image_id": image_id} for split, image_id in missing_labels],
            "MISSING_IMAGE": [{"split": split, "image_id": image_id} for split, image_id in missing_images],
            "DUPLICATE_ID": {"images": duplicate_images, "labels": duplicate_labels, "metadata_images": manifest_image_duplicates},
            "OTHER": {"unexpected_images": unexpected_images, "unexpected_labels": unexpected_labels},
        },
        "per_image_box_mismatches": per_image_box_mismatches,
        "bbox_by_split_source_class": [
            {"split": split, "source": source, "class": original_class, "box_count": count}
            for (split, source, original_class), count in sorted(annotation_counts_by_split_source_class.items())
        ],
        "output_counts_by_dataset": {name: dict(sorted(counts.items())) for name, counts in sorted(output_counts_by_dataset.items())},
        "annotation_manifest_count": annotation_rows,
        "ignored_annotation_count": ignored_annotation_rows,
        "errors": errors,
    }
    return report


def write_report(report: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
