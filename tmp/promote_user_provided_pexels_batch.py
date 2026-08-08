"""Promote the reviewed 13-video user-provided batch into dataset-v1-full.

The two explicit HIGHWAY×TWILIGHT overrides keep one such sequence in
cross-test while ensuring that train has two matching sequences, as required
by the dataset split policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from PIL import Image


ROOT = Path(r"D:\UMT_EVIDENCE\dataset-v1-full")
BATCH = ROOT / "review_pending" / "annotation_batches" / "user_provided_pexels_20260809_v1"
DATASET_NAME = "ONLINE_DATA User Provided Video"
EVALUATION_SPLITS = {"test", "cross_test"}
SPLIT_OVERRIDES = {
    "UPX_13538225": "train",  # was val; becomes train support for H×TWILIGHT
    "UPX_14388179": "train",  # was cross_test; leaves one H×TWILIGHT in cross_test
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_yolo(path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"Invalid YOLO class/format at {path}:{line_number}")
        cx, cy, bw, bh = (float(value) for value in fields[1:])
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, bw, bh)) or bw <= 0.0 or bh <= 0.0:
            raise ValueError(f"Invalid normalized YOLO coordinates at {path}:{line_number}")
        boxes.append(((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height))
    return boxes


def density(box_count: int, frame_count: int) -> tuple[float, str]:
    mean = box_count / frame_count
    return mean, "LOW" if mean <= 4.0 else "MEDIUM" if mean <= 10.0 else "HIGH"


def camera_view(note: str) -> str:
    note = note.lower()
    if "aerial" in note or "overhead" in note:
        return "OVERHEAD_TOP_DOWN"
    if "dashcam" in note:
        return "FRONT_DASHCAM"
    return "ROADSIDE_LEVEL"


def weather(note: str) -> str:
    return "RAIN_OR_WET_ROAD" if "wet-road" in note.lower() else "UNKNOWN"


def cell_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        " / ".join(key): count
        for key, count in sorted(Counter((row["split"], row["road_type"], row["lighting"]) for row in rows).items())
    }


def assert_test_coverage(rows: list[dict[str, str]], minimum: int = 2) -> None:
    train = Counter((row["road_type"], row["lighting"]) for row in rows if row["split"] == "train")
    failures = []
    for road, lighting in sorted({(row["road_type"], row["lighting"]) for row in rows if row["split"] in EVALUATION_SPLITS}):
        if train[(road, lighting)] < minimum:
            failures.append(f"{road}×{lighting}: train={train[(road, lighting)]} < {minimum}")
    if failures:
        raise ValueError("Hard scene split constraint failed: " + "; ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch", type=Path, default=BATCH)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root, batch = args.root, args.batch
    metadata = root / "metadata"
    summary = json.loads((batch / "metadata" / "pseudo_annotation_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE" or summary.get("validation") != "PASS":
        raise ValueError("Batch pseudo-label validation has not passed")

    manifest_fields, manifest = read_csv(batch / "metadata" / "video_extraction_manifest.csv")
    del manifest_fields
    task_fields, tasks = read_csv(batch / "metadata" / "annotation_tasks.csv")
    del task_fields
    image_fields, image_rows = read_csv(metadata / "images.csv")
    annotation_fields, annotation_rows = read_csv(metadata / "annotations.csv")
    scene_fields, scene_rows = read_csv(metadata / "sequence_scene_metadata.csv")
    split_fields, split_rows = read_csv(metadata / "sequence_splits.csv")

    task_by_image = {Path(row["image_file"]).name: row for row in tasks}
    manifest_by_id = {row["video_id"]: row for row in manifest}
    if len(manifest_by_id) != 13 or len(task_by_image) != int(summary["images"]):
        raise ValueError("Unexpected batch manifest/task counts")
    existing = {(row["dataset"], row["sequence_id"]) for row in split_rows}
    collisions = [(DATASET_NAME, video_id) for video_id in manifest_by_id if (DATASET_NAME, video_id) in existing]
    if collisions:
        raise ValueError(f"Sequence IDs already exist in main dataset: {collisions}")

    new_scene_rows: list[dict[str, str]] = []
    new_image_rows: list[dict[str, object]] = []
    new_annotation_rows: list[dict[str, object]] = []
    per_sequence_boxes: Counter[str] = Counter()
    per_sequence_frames: Counter[str] = Counter()
    assets: list[tuple[Path, Path, Path, Path]] = []
    for image_name, task in sorted(task_by_image.items()):
        video_id = task["source_video_id"]
        source = batch / "images" / image_name
        label = batch / "annotations_pending" / f"{Path(image_name).stem}.txt"
        if not source.is_file() or not label.is_file():
            raise ValueError(f"Missing batch image/label pair for {image_name}")
        with Image.open(source) as opened:
            width, height = opened.size
        boxes = parse_yolo(label, width, height)
        split = SPLIT_OVERRIDES.get(video_id, task["intended_split"])
        destination_image = root / "images" / split / image_name
        destination_label = root / "labels" / split / label.name
        if destination_image.exists() or destination_label.exists():
            raise FileExistsError(f"Destination exists for {image_name}")
        frame_id = task["task_id"].rsplit("_", 1)[-1]
        image_id = task["task_id"]
        per_sequence_frames[video_id] += 1
        per_sequence_boxes[video_id] += len(boxes)
        assets.append((source, label, destination_image, destination_label))
        new_image_rows.append({
            "image_id": image_id, "dataset": DATASET_NAME, "split": split, "sequence_id": video_id, "frame_id": frame_id,
            "source_image": task["source_video"], "exported_image": f"images/{split}/{image_name}",
            "exported_label": f"labels/{split}/{label.name}", "width": width, "height": height,
            "vehicle_box_count": len(boxes), "boundary_clipped_box_count": 0,
        })
        for index, (xmin, ymin, xmax, ymax) in enumerate(boxes, start=1):
            new_annotation_rows.append({
                "image_id": image_id, "dataset": DATASET_NAME, "split": split, "sequence_id": video_id, "frame_id": frame_id,
                "annotation_id": f"{image_id}_{index:03d}", "track_id": "", "original_class": "COCO_motor_vehicle",
                "mapped_class": "vehicle", "class_id": 0, "source_image": task["source_video"],
                "source_annotation": "AUTO_YOLO11N_COCO_2026_08_09", "raw_xmin": f"{xmin:.3f}", "raw_ymin": f"{ymin:.3f}",
                "raw_xmax": f"{xmax:.3f}", "raw_ymax": f"{ymax:.3f}", "clipped_xmin": f"{max(0.0, xmin):.3f}",
                "clipped_ymin": f"{max(0.0, ymin):.3f}", "clipped_xmax": f"{min(float(width), xmax):.3f}",
                "clipped_ymax": f"{min(float(height), ymax):.3f}", "clip_applied": "False", "clip_adjustments": "",
                "preserve_original_class": "True",
            })

    for video_id, row in manifest_by_id.items():
        split = SPLIT_OVERRIDES.get(video_id, row["intended_split"])
        mean, traffic = density(per_sequence_boxes[video_id], per_sequence_frames[video_id])
        new_scene_rows.append({
            "dataset_name": DATASET_NAME, "sequence_id": video_id, "road_type": row["target_road_type"], "weather": weather(row["review_note"]),
            "lighting": row["lighting"], "camera_view": camera_view(row["review_note"]), "traffic_density": traffic,
            "mean_vehicles_per_image": f"{mean:.2f}", "weather_source": "USER_PROVIDED_VIDEO_VISUAL_REVIEW",
            "lighting_source": "USER_PROVIDED_VIDEO_VISUAL_REVIEW", "camera_view_source": "USER_PROVIDED_VIDEO_VISUAL_REVIEW",
            "traffic_density_source": "AUTO_YOLO11N_PSEUDOLABEL_MEAN", "manual_review_status": "USER_APPROVED_PROMOTION_AUTO_LABELS",
            "evidence": f"{row['source_group']}; {row['review_note']} 1 FPS extraction; YOLO11n pseudo-labels, class 0; source provenance: {row['license_status']}.",
        })
        split_rows.append({"dataset": DATASET_NAME, "sequence_id": video_id, "split": split, "split_source": "USER_APPROVED_PROMOTION_WITH_HARD_SCENE_CONSTRAINT"})

    current_scene_by_key = {(row["dataset_name"], row["sequence_id"]): row for row in scene_rows}
    split_by_key = {(row["dataset"], row["sequence_id"]): row["split"] for row in split_rows}
    combined = [
        {"split": split_by_key[(dataset, sequence)], "road_type": row["road_type"], "lighting": row["lighting"]}
        for (dataset, sequence), row in current_scene_by_key.items()
    ] + [{"split": split_by_key[(DATASET_NAME, row["sequence_id"])], "road_type": row["road_type"], "lighting": row["lighting"]} for row in new_scene_rows]
    assert_test_coverage(combined)
    audit: dict[str, object] = {
        "status": "DRY_RUN", "batch": str(batch), "images": len(new_image_rows), "boxes": len(new_annotation_rows),
        "sequences": len(new_scene_rows), "split_overrides": SPLIT_OVERRIDES, "after_distribution": cell_distribution(combined),
        "source_provenance": "USER_PROVIDED_PENDING_SOURCE_PROVENANCE",
    }
    if not args.apply:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0

    stamp = date.today().isoformat()
    backup = metadata / "promotion_backups" / f"user_provided_pexels_{stamp}"
    if backup.exists():
        raise FileExistsError(f"Backup directory already exists: {backup}")
    backup.mkdir(parents=True)
    for source in (metadata / "images.csv", metadata / "annotations.csv", metadata / "sequence_scene_metadata.csv", metadata / "sequence_splits.csv", metadata / "export_summary.json"):
        shutil.copy2(source, backup / source.name)
    for source_image, source_label, destination_image, destination_label in assets:
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, destination_image)
        shutil.copy2(source_label, destination_label)
    image_rows.extend(new_image_rows)
    annotation_rows.extend(new_annotation_rows)
    scene_rows.extend(new_scene_rows)
    write_csv(metadata / "images.csv", image_fields, image_rows)
    write_csv(metadata / "annotations.csv", annotation_fields, annotation_rows)
    write_csv(metadata / "sequence_scene_metadata.csv", scene_fields, scene_rows)
    write_csv(metadata / "sequence_splits.csv", split_fields, split_rows)
    summary_path = metadata / "export_summary.json"
    export_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    image_counts = Counter(row["split"] for row in image_rows)
    box_counts = Counter(row["split"] for row in annotation_rows)
    export_summary["images_by_split"] = dict(sorted(image_counts.items()))
    export_summary["boxes_by_split"] = dict(sorted(box_counts.items()))
    export_summary["counts"]["exported_images"] = sum(image_counts.values())
    export_summary["counts"]["input_images"] = sum(image_counts.values())
    export_summary["counts"]["exported_boxes"] = sum(box_counts.values())
    export_summary["sequence_count"] = len(split_rows)
    export_summary.setdefault("counts_by_dataset", {})["ONLINE_DATA User Provided Video (auto-QC)"] = {
        "exported_images": len(new_image_rows), "exported_boxes": len(new_annotation_rows),
        "input_images": len(new_image_rows), "input_annotations": len(new_annotation_rows), "rejected_annotations": 0,
    }
    summary_path.write_text(json.dumps(export_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit.update({"status": "APPLIED", "backup": str(backup), "images_by_split": dict(image_counts), "boxes_by_split": dict(box_counts)})
    (metadata / f"promotion_user_provided_pexels_{stamp}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
