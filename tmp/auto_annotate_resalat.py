"""Create review-required YOLO vehicle labels from a local COCO detector.

Only COCO motor-vehicle classes are mapped to the unified dataset class 0:
car, motorcycle, bus, and truck.  Every generated label remains a pseudo-label
until a human reviewer approves it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from ultralytics import YOLO


VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}


def normalized_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(x1, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0.0 or box_height <= 0.0:
        raise ValueError("Detector returned a non-positive bounding box")
    return (
        (x1 + x2) / (2.0 * width),
        (y1 + y2) / (2.0 * height),
        box_width / width,
        box_height / height,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based offset after deterministic filename sorting.")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--overwrite-empty-labels",
        action="store_true",
        help="Replace only pre-existing zero-byte labels for the selected images; never overwrite non-empty labels.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    images = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    images = images[args.start_index :]
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise ValueError(f"No images found in {args.images}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model weights are missing: {args.model}")
    if not args.dry_run:
        existing = [args.labels / f"{image_path.stem}.txt" for image_path in images]
        nonempty = [path for path in existing if path.exists() and path.stat().st_size > 0]
        empty = [path for path in existing if path.exists() and path.stat().st_size == 0]
        if nonempty:
            raise FileExistsError(
                "Refusing to overwrite non-empty labels for selected images: "
                + ", ".join(str(path) for path in nonempty[:3])
            )
        if empty and not args.overwrite_empty_labels:
            raise FileExistsError(
                "Refusing to overwrite existing empty labels without --overwrite-empty-labels: "
                + ", ".join(str(path) for path in empty[:3])
            )

    model = YOLO(str(args.model))
    model_names = model.names
    vehicle_ids = [class_id for class_id, name in model_names.items() if str(name).lower() in VEHICLE_NAMES]
    if len(vehicle_ids) != len(VEHICLE_NAMES):
        raise RuntimeError(f"COCO vehicle classes are incomplete in model: {model_names}")

    records: list[dict[str, object]] = []
    total_boxes = 0
    for result_index, result in enumerate(model.predict(
        source=[str(path) for path in images],
        stream=True,
        save=False,
        verbose=False,
        classes=vehicle_ids,
        conf=args.confidence,
        iou=args.iou,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
    )):
        if result_index >= len(images):
            raise RuntimeError("Detector returned more results than input images")
        image_path = images[result_index]
        height, width = result.orig_shape
        boxes: list[tuple[float, float, float, float]] = []
        confidences: list[float] = []
        if result.boxes is not None:
            for xyxy, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                boxes.append(normalized_box(*xyxy, width, height))
                confidences.append(float(confidence))
        if not args.dry_run:
            args.labels.mkdir(parents=True, exist_ok=True)
            label_path = args.labels / f"{image_path.stem}.txt"
            label_path.write_text(
                "".join(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for cx, cy, bw, bh in boxes),
                encoding="utf-8",
            )
        total_boxes += len(boxes)
        records.append({
            "image_file": image_path.name,
            "expected_yolo_label": f"{image_path.stem}.txt",
            "vehicle_box_count": len(boxes),
            "min_confidence": f"{min(confidences):.4f}" if confidences else "",
            "max_confidence": f"{max(confidences):.4f}" if confidences else "",
            "mean_confidence": f"{mean(confidences):.4f}" if confidences else "",
            "annotation_status": "AUTO_ANNOTATED_REVIEW_REQUIRED" if boxes else "NO_VEHICLE_DETECTED_REVIEW_REQUIRED",
        })

    summary = {
        "status": "DRY_RUN" if args.dry_run else "COMPLETE",
        "model": str(args.model),
        "coco_vehicle_classes": sorted(VEHICLE_NAMES),
        "mapped_dataset_class": 0,
        "confidence_threshold": args.confidence,
        "iou_threshold": args.iou,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "images": len(records),
        "boxes": total_boxes,
        "images_without_detections": sum(record["vehicle_box_count"] == 0 for record in records),
        "human_review_required": True,
    }
    if not args.dry_run:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
