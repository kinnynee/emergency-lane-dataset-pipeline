"""Merge chunked pseudo-label reports and validate a review-pending YOLO batch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_label(path: Path) -> int:
    boxes = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Invalid class/format at {path}:{number}")
        cx, cy, width, height = (float(value) for value in parts[1:])
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)) or width <= 0 or height <= 0:
            raise ValueError(f"Invalid normalized box at {path}:{number}")
        boxes += 1
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    images_dir = args.batch / "images"
    labels_dir = args.batch / "annotations_pending"
    metadata = args.batch / "metadata"
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    reports = sorted(metadata.glob("pseudo_annotation_chunk_*.csv"))
    if not images or not reports:
        raise ValueError("Images or chunk reports are missing")
    records = [row for report in reports for row in read_csv(report)]
    report_by_image = {row["image_file"]: row for row in records}
    if len(report_by_image) != len(records):
        raise ValueError("Duplicate image rows across chunk reports")
    expected = {path.name for path in images}
    if set(report_by_image) != expected:
        raise ValueError(f"Report/image mismatch: missing={sorted(expected - set(report_by_image))[:3]}, extra={sorted(set(report_by_image) - expected)[:3]}")

    total_boxes = 0
    empty = 0
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise ValueError(f"Missing label for {image.name}")
        boxes = validate_label(label)
        if boxes == 0:
            empty += 1
        if boxes != int(report_by_image[image.name]["vehicle_box_count"]):
            raise ValueError(f"Label/report box mismatch for {image.name}")
        total_boxes += boxes

    ordered = [report_by_image[path.name] for path in images]
    write_csv(metadata / "pseudo_annotation_report.csv", ordered, list(ordered[0]))
    tasks_path = metadata / "annotation_tasks.csv"
    if tasks_path.is_file():
        tasks = read_csv(tasks_path)
        for task in tasks:
            image_file = Path(task.get("image_file", "")).name
            report = report_by_image.get(image_file)
            if report:
                task["auto_annotation_status"] = report["annotation_status"]
                task["auto_vehicle_box_count"] = report["vehicle_box_count"]
                task["expected_yolo_label"] = report["expected_yolo_label"]
        fields = list(tasks[0]) if tasks else []
        write_csv(metadata / "annotation_tasks_auto.csv", tasks, fields)
    summary = {
        "status": "COMPLETE",
        "images": len(images),
        "labels": len(images),
        "boxes": total_boxes,
        "images_without_detections": empty,
        "mapped_dataset_class": 0,
        "validation": "PASS",
        "human_review_required": True,
        "chunk_reports": [path.name for path in reports],
    }
    (metadata / "pseudo_annotation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
