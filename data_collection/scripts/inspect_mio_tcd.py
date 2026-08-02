"""EDA streaming cho MIO-TCD Localization, không chạm phần Classification."""

from __future__ import annotations

import csv
import io
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from external_eda_common import (
    distance_proxy,
    image_from_bytes,
    image_quality,
    letterbox_box_metrics,
    perceptual_hash,
    relative_size_category,
    reservoir_sample,
    sha256_bytes,
    validate_bbox,
)

DATASET = "MIO-TCD Localization"
VEHICLE_CLASSES = {
    "articulated_truck",
    "bus",
    "car",
    "motorcycle",
    "motorized_vehicle",
    "pickup_truck",
    "single_unit_truck",
    "work_van",
}


def _is_image(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png"))


def inspect_mio(
    path: Path,
    sample_size: int,
    full_scan: bool = False,
    skip_images: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    if "classification" in str(path).lower():
        raise ValueError("MIO-TCD Classification không được phép dùng trong EDA.")
    if not path.is_file() or path.suffix.lower() != ".tar":
        raise ValueError(f"Hiện parser MIO yêu cầu TAR Localization: {path}")

    train_images: list[str] = []
    test_count = 0
    csv_name = ""
    readme_name = ""
    with tarfile.open(path, "r:") as archive:
        iterator = tqdm(archive, desc="MIO inventory", disable=not progress)
        for member in iterator:
            name = member.name.replace("\\", "/")
            lower = name.lower()
            if _is_image(name):
                if "/train/" in lower:
                    train_images.append(name)
                elif "/test/" in lower:
                    test_count += 1
            elif lower.endswith("/gt_train.csv"):
                csv_name = name
            elif lower.endswith("/readme.txt"):
                readme_name = name

    total_train = len(train_images)
    image_budget = total_train if full_scan else min(sample_size, total_train)
    sampled_names = set(reservoir_sample(train_images, image_budget, seed=230))
    sampled_ids = {Path(name).stem for name in sampled_names}
    dimensions: dict[str, tuple[int, int]] = {}
    quality_rows: list[dict[str, Any]] = []
    image_records: dict[str, dict[str, Any]] = {}

    if not skip_images and sampled_names:
        with tarfile.open(path, "r:") as archive:
            iterator = tqdm(archive, desc="MIO image sample", disable=not progress)
            for member in iterator:
                if member.name not in sampled_names:
                    continue
                handle = archive.extractfile(member)
                if not handle:
                    continue
                data = handle.read()
                image = image_from_bytes(data)
                row = image_quality(image, member.size, member.name)
                row.update(
                    dataset_name=DATASET,
                    sequence_name="SEQUENCE_NOT_PROVIDED",
                    split="train",
                )
                quality_rows.append(row)
                if image is not None:
                    height, width = image.shape[:2]
                    image_id = Path(member.name).stem
                    dimensions[image_id] = (width, height)
                    image_records[image_id] = {
                        "dataset_name": DATASET,
                        "sequence_name": "SEQUENCE_NOT_PROVIDED",
                        "source_file": member.name,
                        "sha256": sha256_bytes(data),
                        "phash": perceptual_hash(image),
                        "width": width,
                        "height": height,
                    }

    if not csv_name:
        raise RuntimeError("Không tìm thấy gt_train.csv trong MIO Localization TAR.")

    class_counts: Counter[str] = Counter()
    valid_class_counts: Counter[str] = Counter()
    boxes_per_image: Counter[str] = Counter()
    annotated_ids: set[str] = set()
    bbox_category_counts: Counter[str] = Counter()
    bbox_size_counts: Counter[str] = Counter()
    distance_counts: Counter[str] = Counter()
    invalid_rows: list[dict[str, Any]] = []
    bbox_samples: list[dict[str, Any]] = []
    total_annotations = 0
    analyzed_annotations = 0
    duplicate_box_keys: Counter[tuple[str, str, float, float, float, float]] = Counter()

    with tarfile.open(path, "r:") as archive:
        handle = archive.extractfile(csv_name)
        if not handle:
            raise RuntimeError("Không đọc được gt_train.csv.")
        text = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
        for line_number, row in enumerate(csv.reader(text), start=1):
            if len(row) < 6:
                invalid_rows.append(
                    {
                        "dataset_name": DATASET,
                        "sequence_name": "SEQUENCE_NOT_PROVIDED",
                        "source_file": csv_name,
                        "annotation_id": line_number,
                        "issue_type": "MALFORMED_ROW",
                        "severity": "ERROR",
                        "details": f"expected>=6 columns; got={len(row)}",
                        "recommended_action": "MANUAL_REVIEW",
                    }
                )
                continue
            image_id, original_class = row[0].strip(), row[1].strip()
            total_annotations += 1
            class_counts[original_class] += 1
            annotated_ids.add(image_id)
            boxes_per_image[image_id] += 1
            if image_id not in sampled_ids and not full_scan:
                continue
            try:
                xmin, ymin, xmax, ymax = map(float, row[2:6])
            except ValueError:
                invalid_rows.append(
                    {
                        "dataset_name": DATASET,
                        "sequence_name": "SEQUENCE_NOT_PROVIDED",
                        "source_file": csv_name,
                        "annotation_id": line_number,
                        "issue_type": "NON_NUMERIC_BBOX",
                        "severity": "ERROR",
                        "details": ",".join(row[2:6]),
                        "recommended_action": "MANUAL_REVIEW",
                    }
                )
                continue
            width_height = dimensions.get(image_id)
            image_width = width_height[0] if width_height else None
            image_height = width_height[1] if width_height else None
            issues = validate_bbox(xmin, ymin, xmax, ymax, image_width, image_height)
            if issues:
                for issue in issues:
                    invalid_rows.append(
                        {
                            "dataset_name": DATASET,
                            "sequence_name": "SEQUENCE_NOT_PROVIDED",
                            "source_file": f"MIO-TCD-Localization/train/{image_id}.jpg",
                            "annotation_id": line_number,
                            "issue_type": issue,
                            "severity": "ERROR",
                            "details": f"{xmin},{ymin},{xmax},{ymax}",
                            "recommended_action": "MANUAL_REVIEW",
                        }
                    )
                continue
            valid_class_counts[original_class] += 1
            analyzed_annotations += 1
            box_width, box_height = xmax - xmin, ymax - ymin
            area_ratio = (
                box_width * box_height / (image_width * image_height)
                if image_width and image_height
                else None
            )
            letterbox = (
                letterbox_box_metrics(box_width, box_height, image_width, image_height)
                if image_width and image_height
                else {"box_320_category": "NOT_COMPUTED"}
            )
            bbox_category_counts[letterbox["box_320_category"]] += 1
            if area_ratio is not None:
                bbox_size_counts[relative_size_category(area_ratio)] += 1
                distance_counts[distance_proxy(area_ratio)] += 1
            duplicate_box_keys[(image_id, original_class, xmin, ymin, xmax, ymax)] += 1
            if len(bbox_samples) < max(20000, sample_size * 20):
                bbox_samples.append(
                    {
                        "dataset_name": DATASET,
                        "sequence_name": "SEQUENCE_NOT_PROVIDED",
                        "source_file": f"MIO-TCD-Localization/train/{image_id}.jpg",
                        "original_class": original_class,
                        "mapped_class": "vehicle" if original_class in VEHICLE_CLASSES else "",
                        "box_width": round(box_width, 6),
                        "box_height": round(box_height, 6),
                        "box_area_ratio": round(area_ratio, 8) if area_ratio is not None else "",
                        "bbox_size_category": relative_size_category(area_ratio) if area_ratio is not None else "UNKNOWN",
                        "distance": distance_proxy(area_ratio) if area_ratio is not None else "UNKNOWN",
                        **letterbox,
                    }
                )

    exact_duplicate_boxes = sum(count - 1 for count in duplicate_box_keys.values() if count > 1)
    train_ids = {Path(name).stem for name in train_images}
    images_without_boxes = len(train_ids - annotated_ids)
    annotation_without_image = len(annotated_ids - train_ids)
    elapsed = time.perf_counter() - started
    return {
        "dataset_name": DATASET,
        "path": str(path.resolve()),
        "status": "ANALYZED",
        "data_type": "STILL_IMAGES_WITH_LOCALIZATION",
        "image_count": total_train + test_count,
        "train_image_count": total_train,
        "test_image_count": test_count,
        "video_count": 0,
        "sequence_count": 0,
        "annotation_status": "PROVIDED_FOR_TRAIN_LOCALIZATION",
        "annotation_file_count": 1,
        "annotation_row_count": total_annotations,
        "bbox_count": total_annotations,
        "bbox_analyzed_count": analyzed_annotations,
        "track_count": 0,
        "class_counts": dict(class_counts),
        "valid_class_counts": dict(valid_class_counts),
        "images_without_boxes": images_without_boxes,
        "annotation_without_image": annotation_without_image,
        "invalid_annotations": invalid_rows,
        "quality_rows": quality_rows,
        "bbox_samples": bbox_samples,
        "bbox_320_counts": dict(bbox_category_counts),
        "bbox_size_counts": dict(bbox_size_counts),
        "distance_counts": dict(distance_counts),
        "boxes_per_image": boxes_per_image,
        "image_records": list(image_records.values()),
        "sequences": [],
        "conditions": [
            {
                "dataset_name": DATASET,
                "condition": "lighting/weather",
                "value": "NOT_PROVIDED",
                "count": total_train + test_count,
                "unit": "image",
                "assessment_source": "DATASET_METADATA_REVIEW",
            }
        ],
        "exact_duplicate_boxes": exact_duplicate_boxes,
        "readme_present": bool(readme_name),
        "files_processed_successfully": len(quality_rows) + 1,
        "files_failed": sum(1 for row in quality_rows if row.get("read_status") != "OK"),
        "elapsed_seconds": round(elapsed, 3),
        "analysis_scope": "FULL_SCAN" if full_scan else f"SAMPLE_{image_budget}",
    }


__all__ = ["inspect_mio"]
