"""Audit and safely remediate the T2 findings for an exported YOLO dataset.

The T2 review found two release-integrity defects in ``dataset-v1-full``:

* pseudo-labels with the ``UPX_`` prefix were promoted into validation and
  cross-dataset test splits;
* 66 ``ONLINE_HW_DAY_*`` files have labels on disk but no image metadata.

This utility is deliberately conservative.  Its default mode only writes an
evidence report.  ``--apply`` first copies the metadata it changes to a dated
backup, moves (never deletes) the UPX evaluation assets to ``review_pending``,
adds provenance columns, and backfills the orphaned ONLINE_HW image records.
It does not make a claim that pseudo-labels are human verified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


ACTIVE_SPLITS = ("train", "val", "cross_test")
EVALUATION_SPLITS = ("val", "cross_test")
UPX_PREFIX = "UPX_"
ONLINE_HW_PREFIX = "ONLINE_HW_"
UPX_LABEL_SOURCE = "AUTO_YOLO11N_COCO_2026_08_09"
LEGACY_ONLINE_LABEL_SOURCE = "LEGACY_ONLINE_LABELS_UNVERIFIED"
QUARANTINE_NAME = "t2_upx_auto_label_evaluation_20260816"
# The first run preserved this backup before an interrupted finalization.  A
# retry must never overwrite it; it captures a second, independently
# recoverable snapshot instead.
INTERRUPTED_BACKUP_NAME = "t2_dataset_remediation_20260816"
BACKUP_NAME = "t2_dataset_remediation_20260816_retry_01"
PROVENANCE_FIELDS = (
    "label_source",
    "label_review_status",
    "evaluation_eligibility",
    "provenance_status",
    "remediation_note",
)


@dataclass(frozen=True)
class Asset:
    """One image/label pair that will be quarantined."""

    split: str
    image: Path
    label: Path
    row_index: int


def _normalise_path(value: str) -> str:
    return value.replace("\\", "/").strip().lstrip("./").casefold()


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_csv_atomic(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _count_yolo_boxes(path: Path) -> int:
    """Count non-empty YOLO lines, rejecting no data because it is review input."""
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions for {path}")
    return int(width), int(height)


def _metadata_index(rows: list[dict[str, str]]) -> dict[str, int]:
    indexed: dict[str, int] = {}
    for index, row in enumerate(rows):
        exported = _normalise_path(row.get("exported_image", ""))
        if not exported:
            continue
        if exported in indexed:
            raise ValueError(f"Duplicate exported_image in images.csv: {row.get('exported_image', '')}")
        indexed[exported] = index
    return indexed


def _active_images(dataset: Path, split: str, prefix: str | None = None) -> list[Path]:
    directory = dataset / "images" / split
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing image split directory: {directory}")
    pattern = f"{prefix}*" if prefix else "*"
    return sorted((item for item in directory.glob(pattern) if item.is_file()), key=lambda item: item.name.casefold())


def _asset_for_image(dataset: Path, split: str, image: Path, metadata_index: dict[str, int]) -> Asset:
    label = dataset / "labels" / split / f"{image.stem}.txt"
    if not label.is_file():
        raise FileNotFoundError(f"Missing paired label: {label}")
    exported = _normalise_path(image.relative_to(dataset).as_posix())
    row_index = metadata_index.get(exported)
    if row_index is None:
        raise ValueError(f"No metadata/images.csv record for {image.relative_to(dataset)}")
    return Asset(split=split, image=image, label=label, row_index=row_index)


def _scene_metadata(dataset: Path) -> dict[str, dict[str, str]]:
    path = dataset / "metadata" / "sequence_scene_metadata.csv"
    if not path.is_file():
        return {}
    rows, _ = _read_csv(path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        sequence = row.get("sequence_id", "").strip()
        if sequence:
            result[sequence] = row
    return result


def _split_counts(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row.get("split", "") for row in rows)
    return {split: int(counts[split]) for split in ACTIVE_SPLITS}


def _is_upx_row(row: dict[str, str]) -> bool:
    return Path(row.get("exported_image", "")).name.startswith(UPX_PREFIX)


def _is_online_row(row: dict[str, str]) -> bool:
    return Path(row.get("exported_image", "")).name.startswith("ONLINE_")


def _source_label_metadata(row: dict[str, str]) -> dict[str, str]:
    """Return explicit provenance without inventing a human-review event."""
    name = Path(row.get("exported_image", "")).name
    dataset_name = row.get("dataset", "")
    if name.startswith(UPX_PREFIX):
        return {
            "label_source": UPX_LABEL_SOURCE,
            "label_review_status": "NO_HUMAN_REVIEW_EVIDENCE",
            "evaluation_eligibility": "TRAIN_ONLY_PSEUDOLABEL",
            "provenance_status": "PENDING_SOURCE_PROVENANCE",
            "remediation_note": "Pseudo-label generated from COCO motor-vehicle aggregation; motorcycle handling must be reviewed.",
        }
    if name.startswith("ONLINE_URBAN_TWILIGHT_"):
        return {
            "label_source": UPX_LABEL_SOURCE,
            "label_review_status": "AUTO_QC_ONLY",
            "evaluation_eligibility": "NOT_FOR_HEADLINE_EVALUATION",
            "provenance_status": "PENDING_SOURCE_PROVENANCE",
            "remediation_note": "Legacy online auto-QC label; source and human-review evidence must be retained before release use.",
        }
    if name.startswith(ONLINE_HW_PREFIX):
        return {
            "label_source": LEGACY_ONLINE_LABEL_SOURCE,
            "label_review_status": "PENDING_REVIEW",
            "evaluation_eligibility": "NOT_FOR_EVALUATION",
            "provenance_status": "PENDING_SOURCE_LICENSE",
            "remediation_note": "Legacy ONLINE_HW label; provenance and annotation quality require review.",
        }
    if dataset_name in {"MIO-TCD Localization", "AAU RainSnow", "UA-DETRAC Original"}:
        return {
            "label_source": "SOURCE_DATASET_ANNOTATION",
            "label_review_status": "SOURCE_DATASET_QC",
            "evaluation_eligibility": "YES",
            "provenance_status": "SOURCE_DATASET",
            "remediation_note": "",
        }
    return {
        "label_source": "NOT_RECORDED",
        "label_review_status": "NOT_RECORDED",
        "evaluation_eligibility": "REVIEW_REQUIRED",
        "provenance_status": "NOT_RECORDED",
        "remediation_note": "",
    }


def _upx_distribution(rows: list[dict[str, str]], scenes: dict[str, dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if not _is_upx_row(row):
            continue
        scene = scenes.get(row.get("sequence_id", ""), {})
        road_type = scene.get("road_type", "UNKNOWN") or "UNKNOWN"
        lighting = scene.get("lighting", "UNKNOWN") or "UNKNOWN"
        counts[f"{row.get('split', 'UNKNOWN')} / {road_type} / {lighting}"] += 1
    return dict(sorted(counts.items()))


def audit_dataset(dataset: Path) -> dict[str, Any]:
    """Return an evidence-backed remediation plan without mutating ``dataset``."""
    dataset = dataset.resolve()
    images_csv = dataset / "metadata" / "images.csv"
    if not images_csv.is_file():
        raise FileNotFoundError(f"Missing metadata/images.csv: {images_csv}")
    rows, fieldnames = _read_csv(images_csv)
    if not rows or not fieldnames:
        raise ValueError(f"images.csv is empty: {images_csv}")
    required = {"image_id", "dataset", "split", "sequence_id", "exported_image", "exported_label"}
    missing_fields = sorted(required - set(fieldnames))
    if missing_fields:
        raise ValueError(f"images.csv is missing required columns: {missing_fields}")

    indexed = _metadata_index(rows)
    upx_assets: list[Asset] = []
    for split in EVALUATION_SPLITS:
        for image in _active_images(dataset, split, UPX_PREFIX):
            upx_assets.append(_asset_for_image(dataset, split, image, indexed))

    online_hw_files = _active_images(dataset, "train", ONLINE_HW_PREFIX)
    online_hw_orphans = [
        image for image in online_hw_files
        if _normalise_path(image.relative_to(dataset).as_posix()) not in indexed
    ]
    for image in online_hw_orphans:
        label = dataset / "labels" / "train" / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Orphan ONLINE_HW image has no paired label: {label}")

    scenes = _scene_metadata(dataset)
    all_rows_by_split = _split_counts(rows)
    by_dataset_and_split: dict[str, dict[str, int]] = {}
    for dataset_name in sorted({row.get("dataset", "") for row in rows}):
        source_rows = [row for row in rows if row.get("dataset", "") == dataset_name]
        by_dataset_and_split[dataset_name] = _split_counts(source_rows)

    id_mismatches = {
        prefix: sum(
            1
            for row in rows
            if Path(row.get("exported_image", "")).name.startswith(prefix)
            and row.get("image_id", "") != Path(row.get("exported_image", "")).stem
        )
        for prefix in (UPX_PREFIX, "ONLINE_")
    }
    quarantined_boxes = sum(_count_yolo_boxes(asset.label) for asset in upx_assets)
    orphan_boxes = sum(_count_yolo_boxes(dataset / "labels" / "train" / f"{image.stem}.txt") for image in online_hw_orphans)
    orphan_sequences = Counter(image.stem.rsplit("_f", 1)[0] for image in online_hw_orphans)

    issues: list[str] = []
    expected = {"val": 84, "cross_test": 81}
    actual = Counter(asset.split for asset in upx_assets)
    if {split: actual[split] for split in EVALUATION_SPLITS} != expected:
        issues.append(f"Unexpected UPX evaluation count: actual={dict(actual)}, expected={expected}")
    if len(online_hw_orphans) != 66:
        issues.append(f"Unexpected ONLINE_HW orphan count: actual={len(online_hw_orphans)}, expected=66")

    return {
        "schema_version": "t2-remediation-plan-v1",
        "dataset": str(dataset),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_TO_APPLY" if not issues else "REVIEW_COUNTS_BEFORE_APPLY",
        "issues": issues,
        "active_images_by_split_from_metadata": all_rows_by_split,
        "images_by_dataset_and_split": by_dataset_and_split,
        "upx": {
            "label_source": UPX_LABEL_SOURCE,
            "human_review_evidence": "NOT_FOUND_IN_PROMOTION_METADATA",
            "confidence_threshold": "NOT_RECORDED_IN_AVAILABLE_PSEUDO_ANNOTATION_METADATA",
            "evaluation_assets": [
                {
                    "split": asset.split,
                    "image": asset.image.relative_to(dataset).as_posix(),
                    "label": asset.label.relative_to(dataset).as_posix(),
                    "image_id": rows[asset.row_index].get("image_id", ""),
                    "box_count": _count_yolo_boxes(asset.label),
                }
                for asset in upx_assets
            ],
            "evaluation_images_by_split": {split: int(actual[split]) for split in EVALUATION_SPLITS},
            "evaluation_box_count": quarantined_boxes,
            "distribution_by_road_type_and_lighting": _upx_distribution(rows, scenes),
            "image_id_filename_mismatches": id_mismatches[UPX_PREFIX],
        },
        "online_hw": {
            "orphan_images": [image.relative_to(dataset).as_posix() for image in online_hw_orphans],
            "orphan_image_count": len(online_hw_orphans),
            "orphan_box_count": orphan_boxes,
            "orphan_images_by_sequence": dict(sorted(orphan_sequences.items())),
            "image_id_filename_mismatches": id_mismatches["ONLINE_"],
            "visual_review_status": "REQUIRED_FOR_BACKFILL; DO_NOT_INFER_LICENSE_OR_HUMAN_LABEL_REVIEW",
        },
        "validation_coverage": {
            "mio_tcd_val_images": by_dataset_and_split.get("MIO-TCD Localization", {}).get("val", 0),
            "note": "MIO-TCD is train-only in this release; validation is not representative of all training sources.",
        },
        "join_key_policy": {
            "required_key": "exported_image",
            "reason": "image_id does not reliably equal the image filename for ONLINE_* and UPX_* records.",
        },
    }


def _online_hw_backfill_rows(
    dataset: Path,
    fieldnames: list[str],
    plan: dict[str, Any],
    scenes: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative in plan["online_hw"]["orphan_images"]:
        image = dataset / relative
        sequence, frame_part = image.stem.rsplit("_f", 1)
        frame_id = frame_part.zfill(6)
        scene = scenes.get(sequence, {})
        width, height = _image_size(image)
        base = {field: "" for field in fieldnames}
        base.update({
            # This new ID intentionally equals the filename stem.  Existing
            # legacy rows retain their historical IDs, while new records are
            # safe for a filename-based join as well as exported_image.
            "image_id": image.stem,
            "dataset": "ONLINE_DATA",
            "split": "train",
            "sequence_id": sequence,
            "frame_id": frame_id,
            "source_image": f"LEGACY_ONLINE_IMPORT/{image.name}",
            "exported_image": image.relative_to(dataset).as_posix(),
            "exported_label": (dataset / "labels" / "train" / f"{image.stem}.txt").relative_to(dataset).as_posix(),
            "width": str(width),
            "height": str(height),
            "vehicle_box_count": str(_count_yolo_boxes(dataset / "labels" / "train" / f"{image.stem}.txt")),
            "boundary_clipped_box_count": "0",
            "road_type": scene.get("road_type", "HIGHWAY"),
            "weather": scene.get("weather", "CLEAR"),
            "lighting": scene.get("lighting", "DAY"),
        })
        base.update(_source_label_metadata(base))
        base["remediation_note"] = "Backfilled from a legacy ONLINE_HW file/label pair; provenance and annotation quality require review."
        rows.append(base)
    return rows


def _backup_files(dataset: Path, paths: Iterable[Path]) -> Path:
    backup = dataset / "metadata" / "promotion_backups" / BACKUP_NAME
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite existing remediation backup: {backup}")
    backup.mkdir(parents=True)
    for path in paths:
        if path.is_file():
            shutil.copy2(path, backup / path.name)
    return backup


def _update_summary(
    dataset: Path,
    summary: dict[str, Any],
    plan: dict[str, Any],
    quarantined: list[Asset],
    backfills: list[dict[str, str]],
    backup: Path,
) -> dict[str, Any]:
    active_images = {split: len(_active_images(dataset, split)) for split in ACTIVE_SPLITS}
    # Counting every label file takes long enough to make a recoverable
    # remediation look like it stalled.  The source summary was generated
    # from the complete materialized release, and the only active-label change
    # in this operation is the known UPX quarantine.  Derive the new split
    # totals from that source-of-record plus the audited per-file counts.
    source_boxes = summary.get("boxes_by_split", {})
    removed_boxes: Counter[str] = Counter()
    for item in plan["upx"]["evaluation_assets"]:
        removed_boxes[str(item["split"])] += int(item["box_count"])
    active_boxes = {
        split: int(source_boxes.get(split, 0)) - int(removed_boxes[split])
        for split in ACTIVE_SPLITS
    }
    if any(value < 0 for value in active_boxes.values()):
        raise ValueError(f"Summary box reconciliation became negative: {active_boxes}")
    total_active_images = sum(active_images.values())
    total_active_boxes = sum(active_boxes.values())
    counts = dict(summary.get("counts", {}))
    counts["input_images"] = total_active_images
    counts["exported_images"] = total_active_images
    counts["exported_boxes"] = total_active_boxes
    counts["quarantined_pending_review_images"] = len(quarantined)
    counts["metadata_backfilled_images"] = len(backfills)
    summary["counts"] = counts
    summary["images_by_split"] = active_images
    summary["boxes_by_split"] = active_boxes
    summary["t2_remediation"] = {
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "backup": str(backup),
        "upx_evaluation_action": "MOVED_TO_PENDING_REVIEW_NOT_USED_BY_DATA_YAML",
        "upx_quarantined_images_by_original_split": dict(Counter(asset.split for asset in quarantined)),
        # The label files were moved before this summary is written, so use
        # the immutable audit count rather than their now-stale source paths.
        "upx_quarantined_boxes": int(plan["upx"]["evaluation_box_count"]),
        "online_hw_metadata_backfilled": len(backfills),
        "evaluation_metrics_status": "BLOCKED_FOR_PRE_REMEDIATION_UPX_RESULTS",
        "join_key": "exported_image",
    }
    return summary


def _quarantine_original_split(value: str) -> str:
    parts = Path(value.replace("\\", "/")).parts
    try:
        index = parts.index("images")
        split = parts[index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Cannot determine original split from quarantine path: {value}") from exc
    if split not in EVALUATION_SPLITS:
        raise ValueError(f"Invalid quarantined split {split!r} in {value}")
    return split


def _rewrite_annotation_splits(dataset: Path, image_ids: set[str], backup: Path) -> bool:
    """Keep annotation metadata aligned when that optional large manifest exists."""
    source = dataset / "metadata" / "annotations.csv"
    if not source.is_file() or not image_ids:
        return False
    shutil.copy2(source, backup / source.name)
    with source.open("r", encoding="utf-8-sig", newline="") as reader_handle:
        reader = csv.DictReader(reader_handle)
        fields = list(reader.fieldnames or [])
        if "image_id" not in fields or "split" not in fields:
            return False
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=source.parent, prefix=f".{source.name}.", suffix=".tmp"
        ) as writer_handle:
            temporary = Path(writer_handle.name)
            writer = csv.DictWriter(writer_handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                if row.get("image_id", "") in image_ids:
                    row["split"] = "pending_review"
                writer.writerow(row)
    os.replace(temporary, source)
    return True


def _verify_applied(dataset: Path, expected_backfills: int) -> dict[str, Any]:
    rows, _ = _read_csv(dataset / "metadata" / "images.csv")
    index = _metadata_index(rows)
    upx_in_eval = [
        image.relative_to(dataset).as_posix()
        for split in EVALUATION_SPLITS
        for image in _active_images(dataset, split, UPX_PREFIX)
    ]
    online_hw_missing = [
        image.relative_to(dataset).as_posix()
        for image in _active_images(dataset, "train", ONLINE_HW_PREFIX)
        if _normalise_path(image.relative_to(dataset).as_posix()) not in index
    ]
    backfilled = [row for row in rows if row.get("remediation_note", "").startswith("Backfilled from a legacy ONLINE_HW")]
    errors: list[str] = []
    if upx_in_eval:
        errors.append(f"UPX remains in evaluation: {upx_in_eval[:3]}")
    if online_hw_missing:
        errors.append(f"ONLINE_HW remains unregistered: {online_hw_missing[:3]}")
    if len(backfilled) != expected_backfills:
        errors.append(f"Expected {expected_backfills} backfilled rows, found {len(backfilled)}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "active_images_by_split": {split: len(_active_images(dataset, split)) for split in ACTIVE_SPLITS},
        "backfilled_online_hw_images": len(backfilled),
    }


def apply_remediation(dataset: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Apply the plan in place.  Call only after an explicit operator decision."""
    if plan.get("status") != "READY_TO_APPLY":
        raise RuntimeError(f"Plan is not safe to apply: {plan.get('status')}; {plan.get('issues')}")
    dataset = dataset.resolve()
    images_csv = dataset / "metadata" / "images.csv"
    summary_path = dataset / "metadata" / "export_summary.json"
    rows, fieldnames = _read_csv(images_csv)
    index = _metadata_index(rows)
    upx_assets: list[Asset] = []
    for split in EVALUATION_SPLITS:
        for image in _active_images(dataset, split, UPX_PREFIX):
            upx_assets.append(_asset_for_image(dataset, split, image, index))
    if len(upx_assets) != sum(plan["upx"]["evaluation_images_by_split"].values()):
        raise RuntimeError("Dataset changed after the plan was created; regenerate the plan before applying.")

    scene_rows = _scene_metadata(dataset)
    final_fields = [*fieldnames, *(field for field in PROVENANCE_FIELDS if field not in fieldnames)]
    backfills = _online_hw_backfill_rows(dataset, final_fields, plan, scene_rows)
    backup = _backup_files(dataset, [images_csv, summary_path, dataset / "metadata" / "sequence_splits.csv"])
    quarantine = dataset / "review_pending" / QUARANTINE_NAME
    moved: list[tuple[Path, Path]] = []
    try:
        for asset in upx_assets:
            image_target = quarantine / "images" / asset.split / asset.image.name
            label_target = quarantine / "labels" / asset.split / asset.label.name
            if image_target.exists() or label_target.exists():
                raise FileExistsError(f"Quarantine target already exists: {image_target} or {label_target}")
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(asset.image), str(image_target))
            moved.append((asset.image, image_target))
            shutil.move(str(asset.label), str(label_target))
            moved.append((asset.label, label_target))

        quarantine_by_row = {asset.row_index: asset for asset in upx_assets}
        for row_index, row in enumerate(rows):
            row.update(_source_label_metadata(row))
            asset = quarantine_by_row.get(row_index)
            if asset is not None:
                row["split"] = "pending_review"
                row["exported_image"] = (quarantine / "images" / asset.split / asset.image.name).relative_to(dataset).as_posix()
                row["exported_label"] = (quarantine / "labels" / asset.split / asset.label.name).relative_to(dataset).as_posix()
                row["evaluation_eligibility"] = "NO_PENDING_HUMAN_REVIEW"
                row["remediation_note"] = "T2: removed from evaluation because it is a YOLO11n COCO pseudo-label without human-review evidence."
        rows.extend(backfills)
        _write_csv_atomic(images_csv, final_fields, rows)
        annotations_updated = _rewrite_annotation_splits(
            dataset,
            {rows[asset.row_index].get("image_id", "") for asset in upx_assets},
            backup,
        )

        summary = _load_json(summary_path)
        summary = _update_summary(dataset, summary, plan, upx_assets, backfills, backup)
        _write_json_atomic(summary_path, summary)

        quarantine_rows = [
            {
                "image_id": rows[asset.row_index].get("image_id", ""),
                "original_split": asset.split,
                "quarantine_image": (quarantine / "images" / asset.split / asset.image.name).relative_to(dataset).as_posix(),
                "quarantine_label": (quarantine / "labels" / asset.split / asset.label.name).relative_to(dataset).as_posix(),
                "label_source": UPX_LABEL_SOURCE,
                "reason": "AUTO_LABEL_NOT_PERMITTED_IN_EVALUATION",
            }
            for asset in upx_assets
        ]
        _write_csv_atomic(
            dataset / "metadata" / "t2_upx_evaluation_quarantine.csv",
            ["image_id", "original_split", "quarantine_image", "quarantine_label", "label_source", "reason"],
            quarantine_rows,
        )
        _write_csv_atomic(dataset / "metadata" / "t2_online_hw_metadata_backfill.csv", final_fields, backfills)
        verification = _verify_applied(dataset, len(backfills))
        result = {
            "schema_version": "t2-remediation-result-v1",
            "status": verification["status"],
            "dataset": str(dataset),
            "backup": str(backup),
            "quarantine": str(quarantine),
            "annotations_metadata_updated": annotations_updated,
            "verification": verification,
        }
        _write_json_atomic(dataset / "metadata" / "t2_remediation_report.json", result)
        if result["status"] != "PASS":
            raise RuntimeError(f"Post-apply verification failed: {verification['errors']}")
        return result
    except Exception:
        # If metadata generation fails before the operation completes, restore
        # moved assets.  Existing target paths are never overwritten.
        for original, target in reversed(moved):
            if target.exists() and not original.exists():
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(original))
        raise


def _restore_backup_file(source: Path, destination: Path) -> None:
    """Copy a known backup atomically without exposing a partial CSV/JSON."""
    if not source.is_file():
        raise FileNotFoundError(f"Recovery source is missing: {source}")
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent, prefix=f".{destination.name}.", suffix=".restore") as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def rollback_interrupted_remediation(dataset: Path) -> dict[str, Any]:
    """Restore metadata only when an interrupted run already restored assets.

    The operation has tight guards: it will not touch a populated quarantine,
    and it refuses to proceed unless all 165 UPX image/label pairs are back in
    their original evaluation folders.  This makes a retry start from one
    self-consistent release state.
    """
    dataset = dataset.resolve()
    metadata = dataset / "metadata"
    backup = metadata / "promotion_backups" / INTERRUPTED_BACKUP_NAME
    quarantine = dataset / "review_pending" / QUARANTINE_NAME
    required_backups = [backup / name for name in ("images.csv", "annotations.csv", "export_summary.json", "sequence_splits.csv")]
    missing_backups = [str(path) for path in required_backups if not path.is_file()]
    if missing_backups:
        raise FileNotFoundError(f"Interrupted-run backup is incomplete: {missing_backups}")
    quarantine_files = [path for path in quarantine.rglob("*") if path.is_file()] if quarantine.exists() else []
    if quarantine_files:
        raise RuntimeError("Refusing recovery because the quarantine still contains assets; use --resume instead.")
    actual_images = {split: len(_active_images(dataset, split, UPX_PREFIX)) for split in EVALUATION_SPLITS}
    actual_labels = {
        split: len(list((dataset / "labels" / split).glob(f"{UPX_PREFIX}*.txt")))
        for split in EVALUATION_SPLITS
    }
    expected = {"val": 84, "cross_test": 81}
    if actual_images != expected or actual_labels != expected:
        raise RuntimeError(
            "Refusing recovery because UPX assets are not fully restored to their original splits: "
            f"images={actual_images}, labels={actual_labels}"
        )
    for name in ("images.csv", "annotations.csv", "export_summary.json", "sequence_splits.csv"):
        _restore_backup_file(backup / name, metadata / name)
    return {
        "status": "PASS",
        "dataset": str(dataset),
        "restored_from": str(backup),
        "upx_images_by_split": actual_images,
        "upx_labels_by_split": actual_labels,
        "note": "Only metadata was restored; assets were already back in their original locations.",
    }


def resume_remediation(dataset: Path) -> dict[str, Any]:
    """Finish metadata finalization after an interrupted, already-moved run.

    This is intentionally narrower than ``apply_remediation``.  It accepts
    only the recognizable intermediate state created after assets and metadata
    were updated, but before the summary/report write completed.  It never
    moves an image, label, or annotation again.
    """
    dataset = dataset.resolve()
    metadata = dataset / "metadata"
    backup = metadata / "promotion_backups" / BACKUP_NAME
    if not backup.is_dir():
        raise FileNotFoundError(f"Cannot resume without the T2 metadata backup: {backup}")
    source_summary = backup / "export_summary.json"
    if not source_summary.is_file():
        raise FileNotFoundError(f"Cannot resume without the original summary: {source_summary}")
    rows, fieldnames = _read_csv(metadata / "images.csv")
    quarantine_relative = Path("review_pending") / QUARANTINE_NAME / "images"
    quarantined: list[tuple[dict[str, str], str, Path]] = []
    backfills: list[dict[str, str]] = []
    metadata_correction_needed = False
    for row in rows:
        exported = Path(row.get("exported_image", ""))
        if _normalise_path(row.get("exported_image", "")).startswith(_normalise_path(quarantine_relative.as_posix())):
            split = _quarantine_original_split(row.get("exported_image", ""))
            label = dataset / row.get("exported_label", "")
            if not label.is_file():
                raise FileNotFoundError(f"Quarantined metadata row has no label: {label}")
            quarantined.append((row, split, label))
        # Newly backfilled IDs retain the actual ``_fNNNN`` filename stem;
        # older registered ONLINE_HW rows use a sequence/frame ID instead.
        # This distinction also repairs an interrupted early run that applied
        # a backfill note too broadly to the 47 legacy rows.
        if row.get("image_id", "").startswith(("ONLINE_HW_DAY_01_f", "ONLINE_HW_DAY_02_f")):
            backfills.append(row)
        elif Path(row.get("exported_image", "")).name.startswith(ONLINE_HW_PREFIX):
            corrected = _source_label_metadata(row)
            if any(row.get(field, "") != value for field, value in corrected.items()):
                row.update(corrected)
                metadata_correction_needed = True
    expected = {"val": 84, "cross_test": 81}
    actual = Counter(split for _, split, _ in quarantined)
    if {split: actual[split] for split in EVALUATION_SPLITS} != expected:
        raise RuntimeError(f"Interrupted state has unexpected quarantined UPX count: {dict(actual)}")
    if len(backfills) != 66:
        raise RuntimeError(f"Interrupted state has {len(backfills)} ONLINE_HW backfills, expected 66")

    # Persist the correction to the pre-existing ONLINE_HW provenance notes
    # only when one is needed.  The normal completed state avoids rewriting a
    # 64 MB image manifest merely to create the small final artifacts.
    if metadata_correction_needed:
        _write_csv_atomic(metadata / "images.csv", fieldnames, rows)

    plan = {
        "upx": {
            "evaluation_assets": [
                {"split": split, "box_count": _count_yolo_boxes(label)}
                for _, split, label in quarantined
            ],
            "evaluation_box_count": sum(_count_yolo_boxes(label) for _, _, label in quarantined),
        },
    }
    quarantined_assets = [
        Asset(split=split, image=dataset / row.get("exported_image", ""), label=label, row_index=-1)
        for row, split, label in quarantined
    ]
    summary = _update_summary(
        dataset,
        _load_json(source_summary),
        plan,
        quarantined_assets,
        backfills,
        backup,
    )
    _write_json_atomic(metadata / "export_summary.json", summary)
    _write_csv_atomic(
        metadata / "t2_upx_evaluation_quarantine.csv",
        ["image_id", "original_split", "quarantine_image", "quarantine_label", "label_source", "reason"],
        [
            {
                "image_id": row.get("image_id", ""),
                "original_split": split,
                "quarantine_image": row.get("exported_image", ""),
                "quarantine_label": row.get("exported_label", ""),
                "label_source": row.get("label_source", UPX_LABEL_SOURCE),
                "reason": "AUTO_LABEL_NOT_PERMITTED_IN_EVALUATION",
            }
            for row, split, _ in quarantined
        ],
    )
    _write_csv_atomic(metadata / "t2_online_hw_metadata_backfill.csv", fieldnames, backfills)
    verification = _verify_applied(dataset, len(backfills))
    result = {
        "schema_version": "t2-remediation-result-v1",
        "status": verification["status"],
        "dataset": str(dataset),
        "backup": str(backup),
        "resumed_after_interruption": True,
        "annotations_metadata_updated": (backup / "annotations.csv").is_file(),
        "verification": verification,
    }
    _write_json_atomic(metadata / "t2_remediation_report.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"Post-resume verification failed: {verification['errors']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset-v1-full or a copied release")
    parser.add_argument("--report", type=Path, help="Where to write the dry-run evidence JSON")
    parser.add_argument("--apply", action="store_true", help="Apply the reversible quarantine and metadata remediation")
    parser.add_argument("--resume", action="store_true", help="Finish an interrupted T2 apply after verifying its backup and quarantine state")
    parser.add_argument("--rollback-interrupted", action="store_true", help="Restore metadata from the safeguarded interrupted-run backup after assets were rolled back")
    parser.add_argument("--confirm-apply-t2", action="store_true", help="Required together with --apply")
    args = parser.parse_args()

    selected_actions = sum(bool(value) for value in (args.apply, args.resume, args.rollback_interrupted))
    if selected_actions > 1:
        parser.error("Use only one of --apply, --resume, or --rollback-interrupted")
    if args.rollback_interrupted:
        if not args.confirm_apply_t2:
            parser.error("--rollback-interrupted requires --confirm-apply-t2")
        result = rollback_interrupted_remediation(args.dataset)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.resume:
        if not args.confirm_apply_t2:
            parser.error("--resume requires --confirm-apply-t2")
        result = resume_remediation(args.dataset)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    plan = audit_dataset(args.dataset)
    if args.report:
        _write_json_atomic(args.report.resolve(), plan)
    if not args.apply:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0 if plan["status"] == "READY_TO_APPLY" else 2
    if not args.confirm_apply_t2:
        parser.error("--apply requires --confirm-apply-t2 after reviewing the generated plan")
    result = apply_remediation(args.dataset, plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
