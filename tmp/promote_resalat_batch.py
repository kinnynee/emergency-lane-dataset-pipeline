"""Promote the reviewed Resalat batch to the unified YOLO cross-test split.

The script refuses to overwrite files.  It archives the older unlabelled
Resalat import so one source video cannot appear twice in the dataset, then
updates the sequence and export manifests for the promoted, class-0 labels.
Run without --execute to perform all safety checks without changing data.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import cv2


DATASET = Path(r"D:/UMT_EVIDENCE/dataset-v1-full")
BATCH = DATASET / "review_pending/annotation_batches/cross_test/urban_road_twilight_resalat_v1"
NEW_SEQUENCE = "ONLINE_URBAN_TWILIGHT_01"
OLD_SEQUENCE = "ONLINE_HW_TWILIGHT_01"
TODAY = date.today().isoformat()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def write_csv_atomic(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".promotion_tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def append_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = csv_header(path)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerows(rows)


def last_annotation_id(path: Path) -> int:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        end = handle.tell()
        handle.seek(max(0, end - 8192))
        lines = handle.read().decode("utf-8-sig").splitlines()
    for line in reversed(lines):
        if line and not line.startswith("image_id,"):
            return int(next(csv.reader([line]))[5])
    raise RuntimeError(f"Could not determine final annotation id from {path}")


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    height, width = image.shape[:2]
    return width, height


def yolo_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if len(values) != 5 or values[0] != "0":
            raise ValueError(f"Invalid class-0 YOLO row at {label_path}:{line_number}")
        cx, cy, width, height = (float(value) for value in values[1:])
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)) or width <= 0 or height <= 0:
            raise ValueError(f"Out-of-range YOLO row at {label_path}:{line_number}")
        boxes.append((cx, cy, width, height))
    if not boxes:
        raise ValueError(f"Promotion batch contains an empty label: {label_path}")
    return boxes


def backup(path: Path, backup_dir: Path) -> None:
    destination = backup_dir / f"{path.name}.before_resalat_promotion_{TODAY}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite backup: {destination}")
    shutil.copy2(path, destination)


def main(execute: bool) -> int:
    images_dir = BATCH / "images"
    labels_dir = BATCH / "annotations_pending"
    source_manifest = BATCH / "metadata/source_video_manifest.csv"
    images = sorted(images_dir.glob("*.jpg"))
    labels = {path.stem: path for path in labels_dir.glob("*.txt")}
    if len(images) != 132 or len(labels) != 132:
        raise ValueError(f"Expected 132 paired batch images and labels, found {len(images)} images and {len(labels)} labels")
    if any(image.stem not in labels for image in images):
        raise ValueError("At least one batch image is missing a label")

    image_target = DATASET / "images/cross_test"
    label_target = DATASET / "labels/cross_test"
    target_pairs = [(image, labels[image.stem], f"{NEW_SEQUENCE}_{image.stem}") for image in images]
    collisions = [stem for _image, _label, stem in target_pairs if (image_target / f"{stem}.jpg").exists() or (label_target / f"{stem}.txt").exists()]
    if collisions:
        raise FileExistsError(f"Promotion target already exists for {len(collisions)} files; no files changed")

    legacy_images = sorted((DATASET / "images/test").glob(f"{OLD_SEQUENCE}_*.jpg"))
    legacy_labels = sorted((DATASET / "labels/test").glob(f"{OLD_SEQUENCE}_*.txt"))
    if len(legacy_images) != 27 or len(legacy_labels) != 27:
        raise ValueError(f"Expected 27 legacy Resalat image/label pairs, found {len(legacy_images)} images and {len(legacy_labels)} labels")
    if any(path.read_text(encoding="utf-8") for path in legacy_labels):
        raise ValueError("Legacy Resalat labels are not empty; refusing to archive")

    split_path = DATASET / "metadata/sequence_splits.csv"
    scene_path = DATASET / "metadata/sequence_scene_metadata.csv"
    summary_path = DATASET / "metadata/export_summary.json"
    images_manifest = DATASET / "metadata/images.csv"
    annotations_manifest = DATASET / "metadata/annotations.csv"
    split_rows = csv_rows(split_path)
    scene_rows = csv_rows(scene_path)
    old_split = [row for row in split_rows if row.get("dataset") == "ONLINE_DATA" and row.get("sequence_id") == OLD_SEQUENCE]
    old_scene = [row for row in scene_rows if row.get("dataset_name") == "ONLINE_DATA" and row.get("sequence_id") == OLD_SEQUENCE]
    if len(old_split) != 1 or len(old_scene) != 1:
        raise ValueError("Expected exactly one legacy Resalat sequence row in both metadata manifests")
    if any(row.get("sequence_id") == NEW_SEQUENCE for row in split_rows + scene_rows):
        raise ValueError(f"Sequence {NEW_SEQUENCE} already exists in metadata")

    source_rows = csv_rows(source_manifest)
    if len(source_rows) != 1:
        raise ValueError("Expected one Resalat source manifest row")
    source = source_rows[0]
    box_rows: list[dict[str, str]] = []
    image_rows: list[dict[str, str]] = []
    total_boxes = 0
    widths: set[int] = set()
    heights: set[int] = set()
    for frame_number, (image, label, target_stem) in enumerate(target_pairs, start=1):
        width, height = image_size(image)
        widths.add(width)
        heights.add(height)
        boxes = yolo_boxes(label)
        image_id = f"{NEW_SEQUENCE}_{frame_number:06d}"
        image_rows.append({
            "image_id": image_id,
            "dataset": "ONLINE_DATA",
            "split": "cross_test",
            "sequence_id": NEW_SEQUENCE,
            "frame_id": f"{frame_number:06d}",
            "source_image": source["source_video"],
            "exported_image": f"images/cross_test/{target_stem}.jpg",
            "exported_label": f"labels/cross_test/{target_stem}.txt",
            "width": str(width),
            "height": str(height),
            "vehicle_box_count": str(len(boxes)),
            "boundary_clipped_box_count": "0",
        })
        for box_index, (cx, cy, box_width, box_height) in enumerate(boxes, start=1):
            x1 = (cx - box_width / 2) * width
            y1 = (cy - box_height / 2) * height
            x2 = (cx + box_width / 2) * width
            y2 = (cy + box_height / 2) * height
            box_rows.append({
                "image_id": image_id,
                "dataset": "ONLINE_DATA",
                "split": "cross_test",
                "sequence_id": NEW_SEQUENCE,
                "frame_id": f"{frame_number:06d}",
                "annotation_id": f"{NEW_SEQUENCE}_{frame_number:06d}_{box_index:03d}",
                "track_id": "",
                "original_class": "COCO_motor_vehicle",
                "mapped_class": "vehicle",
                "class_id": "0",
                "source_image": source["source_video"],
                "source_annotation": "AUTO_YOLO11N_COCO_2026_08_09",
                "raw_xmin": f"{x1:.3f}",
                "raw_ymin": f"{y1:.3f}",
                "raw_xmax": f"{x2:.3f}",
                "raw_ymax": f"{y2:.3f}",
                "clipped_xmin": f"{x1:.3f}",
                "clipped_ymin": f"{y1:.3f}",
                "clipped_xmax": f"{x2:.3f}",
                "clipped_ymax": f"{y2:.3f}",
                "clip_applied": "False",
                "clip_adjustments": "",
                "preserve_original_class": "True",
            })
        total_boxes += len(boxes)
    if widths != {1280} or heights != {720}:
        raise ValueError(f"Unexpected promoted dimensions: widths={widths}, heights={heights}")

    new_split = {
        "dataset": "ONLINE_DATA",
        "sequence_id": NEW_SEQUENCE,
        "split": "cross_test",
        "split_source": "PROMOTED_RESALAT_AUTOMATED_QC_2026_08_09",
    }
    new_scene = {
        "dataset_name": "ONLINE_DATA",
        "sequence_id": NEW_SEQUENCE,
        "road_type": "URBAN_ROAD",
        "weather": "CLEAR_OR_UNKNOWN",
        "lighting": "TWILIGHT",
        "camera_view": "DASHCAM",
        "traffic_density": "LOW_TO_MODERATE",
        "mean_vehicles_per_image": f"{total_boxes / len(images):.2f}",
        "weather_source": "ONLINE_DATA_EDA_REVIEW_MANIFEST",
        "lighting_source": "ONLINE_DATA_EDA_REVIEW_MANIFEST_DUSK_NORMALIZED_TO_TWILIGHT",
        "camera_view_source": "ONLINE_DATA_EDA_REVIEW_MANIFEST",
        "traffic_density_source": "ONLINE_DATA_EDA_REVIEW_MANIFEST",
        "manual_review_status": "USER_AUTHORIZED_AUTOMATED_QC_2026_08_09",
        "evidence": "Resalat source video; 132 frames at 1 FPS; YOLO11n COCO motor-vehicle boxes visually sampled across timeline and authorized for cross_test promotion.",
    }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    planned = {
        "new_sequence": NEW_SEQUENCE,
        "split": "cross_test",
        "images": len(image_rows),
        "boxes": total_boxes,
        "archive_legacy_images": len(legacy_images),
        "archive_legacy_labels": len(legacy_labels),
        "dimensions": "1280x720",
    }
    if not execute:
        print(json.dumps({"status": "DRY_RUN", **planned}, ensure_ascii=False, indent=2))
        return 0

    backup_dir = DATASET / f"metadata/promotion_backups/resalat_{TODAY}"
    legacy_dir = DATASET / "review_pending/legacy_imports/resalat_empty_labels_test_import_v1"
    if backup_dir.exists() or legacy_dir.exists():
        raise FileExistsError("Promotion backup or legacy archive already exists; refusing to overwrite")
    backup_dir.mkdir(parents=True)
    for path in (split_path, scene_path, summary_path, images_manifest, annotations_manifest):
        backup(path, backup_dir)
    (legacy_dir / "images").mkdir(parents=True)
    (legacy_dir / "labels").mkdir(parents=True)

    for image, label, target_stem in target_pairs:
        shutil.copy2(image, image_target / f"{target_stem}.jpg")
        shutil.copy2(label, label_target / f"{target_stem}.txt")
    for path in legacy_images:
        shutil.move(str(path), legacy_dir / "images" / path.name)
    for path in legacy_labels:
        shutil.move(str(path), legacy_dir / "labels" / path.name)

    split_rows = [row for row in split_rows if not (row.get("dataset") == "ONLINE_DATA" and row.get("sequence_id") == OLD_SEQUENCE)] + [new_split]
    scene_rows = [row for row in scene_rows if not (row.get("dataset_name") == "ONLINE_DATA" and row.get("sequence_id") == OLD_SEQUENCE)] + [new_scene]
    write_csv_atomic(split_path, split_rows, csv_header(split_path))
    write_csv_atomic(scene_path, scene_rows, csv_header(scene_path))
    append_csv(images_manifest, image_rows)
    append_csv(annotations_manifest, box_rows)

    summary["counts"]["input_annotations"] += total_boxes
    summary["counts"]["exported_boxes"] += total_boxes
    summary["counts"]["input_images"] += len(image_rows)
    summary["counts"]["exported_images"] += len(image_rows)
    summary["images_by_split"]["cross_test"] += len(image_rows)
    summary["boxes_by_split"]["cross_test"] += total_boxes
    summary["counts_by_dataset"]["ONLINE_DATA Resalat (auto-QC)"] = {
        "exported_boxes": total_boxes,
        "exported_images": len(image_rows),
        "input_annotations": total_boxes,
        "input_images": len(image_rows),
        "rejected_annotations": 0,
    }
    summary["sequence_count"] = len(scene_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (legacy_dir / "archive_manifest.json").write_text(
        json.dumps({"archived_sequence": OLD_SEQUENCE, "reason": "Superseded by labeled Resalat promotion to cross_test", "images": len(legacy_images), "labels": len(legacy_labels)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (BATCH / "metadata/promotion_summary.json").write_text(json.dumps({"status": "COMPLETE", **planned}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", **planned}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Perform promotion after validation; defaults to dry run.")
    args = parser.parse_args()
    raise SystemExit(main(args.execute))
