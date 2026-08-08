"""Rebalance dataset-v1-full by sequence with hard test-to-train scene coverage.

The split decision is never made from a frame hash.  Every image of a sequence
is moved together, requested sequence overrides are explicit, and every
``road_type × lighting`` cell that remains in test/cross_test must have at
least ``--minimum-train-per-test-cell`` sequences in train.

Run without ``--apply`` first.  Applying creates a metadata backup and a JSON
audit trail before moving assets.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable

from PIL import Image


DEFAULT_ROOT = Path(r"D:\UMT_EVIDENCE\dataset-v1-full")
DEFAULT_REPAIR = DEFAULT_ROOT / "review_pending" / "annotation_batches" / "legacy_highway_night_repair_20260809_v1" / "annotations_pending"
EVALUATION_SPLITS = {"test", "cross_test"}
FIXED_MOVES = {
    ("ONLINE_DATA", "ONLINE_HW_NIGHT_01"): "train",
    ("ONLINE_DATA", "ONLINE_HW_NIGHT_02"): "train",
    ("AAU RainSnow", "Ringvej-3"): "train",
    ("ONLINE_DATA", "ONLINE_URBAN_TWILIGHT_01"): "train",
}
LEGACY_PREFIXES = {"ONLINE_HW_NIGHT_01", "ONLINE_HW_NIGHT_02"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scene_rows(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    _scene_fields, scenes = read_csv(root / "metadata" / "sequence_scene_metadata.csv")
    _split_fields, splits = read_csv(root / "metadata" / "sequence_splits.csv")
    return scenes, splits


def joined_scenes(scenes: list[dict[str, str]], splits: list[dict[str, str]]) -> list[dict[str, str]]:
    split_map = {(row["dataset"], row["sequence_id"]): row["split"] for row in splits}
    rows: list[dict[str, str]] = []
    for scene in scenes:
        key = (scene["dataset_name"], scene["sequence_id"])
        if key not in split_map:
            raise ValueError(f"Missing split metadata for {key}")
        rows.append({
            "dataset": key[0], "sequence_id": key[1], "split": split_map[key],
            "road_type": scene["road_type"], "lighting": scene["lighting"],
        })
    return rows


def distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        " / ".join(key): count
        for key, count in sorted(Counter((row["split"], row["road_type"], row["lighting"]) for row in rows).items())
    }


def coverage_violations(rows: list[dict[str, str]], minimum: int) -> list[dict[str, object]]:
    train = Counter((row["road_type"], row["lighting"]) for row in rows if row["split"] == "train")
    test_cells = sorted({(row["road_type"], row["lighting"]) for row in rows if row["split"] in EVALUATION_SPLITS})
    return [
        {"road_type": road, "lighting": lighting, "train_sequences": train[(road, lighting)], "minimum": minimum}
        for road, lighting in test_cells
        if train[(road, lighting)] < minimum
    ]


def apply_fixed_moves(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    current = {(row["dataset"], row["sequence_id"]): row for row in rows}
    missing = sorted(set(FIXED_MOVES) - set(current))
    if missing:
        raise ValueError(f"Required sequences are absent from scene metadata: {missing}")
    moves: list[dict[str, str]] = []
    for key, target in FIXED_MOVES.items():
        row = current[key]
        if row["split"] != target:
            moves.append({"dataset": key[0], "sequence_id": key[1], "from_split": row["split"], "to_split": target})
            row["split"] = target
    return moves


def yolo_boxes(path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Invalid class/format in {path}:{line_number}")
        cx, cy, bw, bh = (float(value) for value in parts[1:])
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
            raise ValueError(f"Invalid normalized box in {path}:{line_number}")
        boxes.append(((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height))
    return boxes


def legacy_frame_id(stem: str) -> str:
    match = re.search(r"_f(\d+)$", stem)
    if not match:
        raise ValueError(f"Cannot infer frame id from {stem}")
    return match.group(1).zfill(6)


def image_and_label_paths(root: Path, image_row: dict[str, str]) -> tuple[Path, Path]:
    image = root / image_row["exported_image"]
    label = root / image_row["exported_label"]
    return image, label


def preflight(root: Path, repair_labels: Path, images: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tracked = {(row["dataset"], row["sequence_id"]): [] for row in images}
    for row in images:
        key = (row["dataset"], row["sequence_id"])
        if key in FIXED_MOVES:
            tracked[key].append(row)

    file_moves: list[dict[str, object]] = []
    legacy_records: list[dict[str, object]] = []
    for key, target in FIXED_MOVES.items():
        source_rows = tracked.get(key, [])
        if source_rows:
            for row in source_rows:
                image, label = image_and_label_paths(root, row)
                if not image.is_file() or not label.is_file() or label.stat().st_size == 0:
                    raise ValueError(f"Tracked sequence is missing a non-empty image/label pair: {key}, {image.name}")
                file_moves.append({"image": image, "label": label, "target_split": target, "legacy": False})
            continue

        dataset, sequence = key
        if dataset != "ONLINE_DATA" or sequence not in LEGACY_PREFIXES:
            raise ValueError(f"No image metadata for non-legacy sequence {key}")
        source_split = "test" if sequence.endswith("01") else "val"
        source_images = sorted((root / "images" / source_split).glob(f"{sequence}_*.jpg"))
        if not source_images:
            raise ValueError(f"Legacy images not found for {sequence}")
        for image in source_images:
            source_label = root / "labels" / source_split / f"{image.stem}.txt"
            repaired_label = repair_labels / f"{image.stem}.txt"
            if not source_label.is_file() or source_label.stat().st_size != 0:
                raise ValueError(f"Legacy source label must be an existing empty file: {source_label}")
            if not repaired_label.is_file():
                raise ValueError(f"Missing repaired label: {repaired_label}")
            with Image.open(image) as opened:
                width, height = opened.size
            boxes = yolo_boxes(repaired_label, width, height)
            file_moves.append({"image": image, "label": source_label, "repaired_label": repaired_label, "target_split": target, "legacy": True})
            legacy_records.append({"sequence_id": sequence, "image": image, "width": width, "height": height, "boxes": boxes})
    return file_moves, legacy_records


def update_metadata(root: Path, images: list[dict[str, str]], legacy_records: list[dict[str, object]], backup: Path) -> dict[str, dict[str, int]]:
    metadata = root / "metadata"
    image_fields, image_rows = read_csv(metadata / "images.csv")
    annotation_fields, annotation_rows = read_csv(metadata / "annotations.csv")
    split_fields, split_rows = read_csv(metadata / "sequence_splits.csv")
    for source in (metadata / "images.csv", metadata / "annotations.csv", metadata / "sequence_splits.csv", metadata / "export_summary.json"):
        shutil.copy2(source, backup / source.name)

    move_keys = set(FIXED_MOVES)
    for row in split_rows:
        key = (row["dataset"], row["sequence_id"])
        if key in move_keys:
            row["split"] = "train"
    for row in image_rows:
        key = (row["dataset"], row["sequence_id"])
        if key in move_keys:
            old = row["split"]
            row["split"] = "train"
            row["exported_image"] = row["exported_image"].replace(f"images/{old}/", "images/train/")
            row["exported_label"] = row["exported_label"].replace(f"labels/{old}/", "labels/train/")
    for row in annotation_rows:
        if (row["dataset"], row["sequence_id"]) in move_keys:
            row["split"] = "train"

    image_keys = {row["image_id"] for row in image_rows}
    for record in legacy_records:
        image = record["image"]
        assert isinstance(image, Path)
        sequence = str(record["sequence_id"])
        image_id = f"{sequence}_{legacy_frame_id(image.stem)}"
        if image_id in image_keys:
            raise ValueError(f"Duplicate legacy image id: {image_id}")
        boxes = record["boxes"]
        assert isinstance(boxes, list)
        image_rows.append({
            "image_id": image_id, "dataset": "ONLINE_DATA", "split": "train", "sequence_id": sequence,
            "frame_id": legacy_frame_id(image.stem), "source_image": f"LEGACY_ONLINE_IMPORT/{image.name}",
            "exported_image": f"images/train/{image.name}", "exported_label": f"labels/train/{image.stem}.txt",
            "width": record["width"], "height": record["height"], "vehicle_box_count": len(boxes), "boundary_clipped_box_count": 0,
        })
        width, height = int(record["width"]), int(record["height"])
        for index, (xmin, ymin, xmax, ymax) in enumerate(boxes, start=1):
            annotation_rows.append({
                "image_id": image_id, "dataset": "ONLINE_DATA", "split": "train", "sequence_id": sequence,
                "frame_id": legacy_frame_id(image.stem), "annotation_id": f"{image_id}_{index:03d}", "track_id": "",
                "original_class": "COCO_motor_vehicle", "mapped_class": "vehicle", "class_id": 0,
                "source_image": f"LEGACY_ONLINE_IMPORT/{image.name}", "source_annotation": "AUTO_YOLO11N_COCO_2026_08_09",
                "raw_xmin": f"{xmin:.3f}", "raw_ymin": f"{ymin:.3f}", "raw_xmax": f"{xmax:.3f}", "raw_ymax": f"{ymax:.3f}",
                "clipped_xmin": f"{max(0.0, xmin):.3f}", "clipped_ymin": f"{max(0.0, ymin):.3f}",
                "clipped_xmax": f"{min(float(width), xmax):.3f}", "clipped_ymax": f"{min(float(height), ymax):.3f}",
                "clip_applied": "False", "clip_adjustments": "", "preserve_original_class": "True",
            })

    write_csv(metadata / "sequence_splits.csv", split_fields, split_rows)
    write_csv(metadata / "images.csv", image_fields, image_rows)
    write_csv(metadata / "annotations.csv", annotation_fields, annotation_rows)
    image_counts = Counter(row["split"] for row in image_rows)
    box_counts = Counter(row["split"] for row in annotation_rows)
    summary_path = metadata / "export_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["images_by_split"] = dict(sorted(image_counts.items()))
    summary["boxes_by_split"] = dict(sorted(box_counts.items()))
    summary["counts"]["exported_images"] = sum(image_counts.values())
    summary["counts"]["input_images"] = sum(image_counts.values())
    summary["counts"]["exported_boxes"] = sum(box_counts.values())
    summary["sequence_count"] = len(split_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"images_by_split": dict(image_counts), "boxes_by_split": dict(box_counts)}


def move_assets(root: Path, file_moves: list[dict[str, object]], backup: Path) -> None:
    empty_backup = backup / "replaced_empty_labels"
    empty_backup.mkdir(parents=True, exist_ok=True)
    for item in file_moves:
        image, label = item["image"], item["label"]
        assert isinstance(image, Path) and isinstance(label, Path)
        target_split = str(item["target_split"])
        destination_image = root / "images" / target_split / image.name
        destination_label = root / "labels" / target_split / label.name
        if destination_image.exists() or destination_label.exists():
            raise FileExistsError(f"Destination already exists: {destination_image} / {destination_label}")
    for item in file_moves:
        image, label = item["image"], item["label"]
        assert isinstance(image, Path) and isinstance(label, Path)
        target_split = str(item["target_split"])
        destination_image = root / "images" / target_split / image.name
        destination_label = root / "labels" / target_split / label.name
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        if bool(item["legacy"]):
            shutil.copy2(label, empty_backup / label.name)
            label.unlink()
            repaired = item["repaired_label"]
            assert isinstance(repaired, Path)
            shutil.copy2(repaired, destination_label)
            shutil.move(str(image), str(destination_image))
        else:
            shutil.move(str(image), str(destination_image))
            shutil.move(str(label), str(destination_label))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repair-labels", type=Path, default=DEFAULT_REPAIR)
    parser.add_argument("--minimum-train-per-test-cell", type=int, default=2)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.minimum_train_per_test_cell < 1:
        raise ValueError("minimum must be at least one")

    scenes, splits = scene_rows(args.root)
    before = joined_scenes(scenes, splits)
    planned = [dict(row) for row in before]
    moves = apply_fixed_moves(planned)
    violations = coverage_violations(planned, args.minimum_train_per_test_cell)
    audit: dict[str, object] = {
        "policy": {"minimum_train_per_test_cell": args.minimum_train_per_test_cell, "evaluation_splits": sorted(EVALUATION_SPLITS), "hash_assignment_allowed": False},
        "fixed_moves": moves, "before_distribution": distribution(before), "after_distribution": distribution(planned),
        "coverage_violations": violations, "status": "DRY_RUN",
    }
    if violations:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        raise SystemExit("Hard scene coverage rule is not satisfied; refusing to apply.")

    image_fields, images = read_csv(args.root / "metadata" / "images.csv")
    del image_fields
    file_moves, legacy_records = preflight(args.root, args.repair_labels, images)
    audit["assets_to_move"] = len(file_moves)
    audit["legacy_images_to_register"] = len(legacy_records)
    if args.apply:
        stamp = date.today().isoformat()
        backup = args.root / "metadata" / "promotion_backups" / f"scene_rebalance_{stamp}"
        if backup.exists():
            raise FileExistsError(f"Backup directory already exists: {backup}")
        backup.mkdir(parents=True)
        move_assets(args.root, file_moves, backup)
        counts = update_metadata(args.root, images, legacy_records, backup)
        audit["updated_counts"] = counts
        audit["backup"] = str(backup)
        audit["status"] = "APPLIED"
        (args.root / "metadata" / "split_constraints_v1.json").write_text(json.dumps(audit["policy"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.root / "metadata" / f"scene_rebalance_{stamp}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
