"""EDA cho AAU RainSnow: video/sequence, COCO RGB và thermal."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from external_eda_common import (
    distance_proxy,
    image_quality,
    letterbox_box_metrics,
    perceptual_hash,
    relative_size_category,
    reservoir_sample,
    sha256_bytes,
    validate_bbox,
)

DATASET = "AAU RainSnow"
VEHICLE_CLASSES = {"bicycle", "car", "motorbike", "bus", "truck"}


def _canonical_files(root: Path) -> tuple[list[Path], list[Path], Counter[str]]:
    images: list[Path] = []
    videos: list[Path] = []
    extensions: Counter[str] = Counter()
    for current, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name.lower() != "aaurainsnow"]
        base = Path(current)
        for name in files:
            path = base / name
            suffix = path.suffix.lower()
            extensions[suffix] += 1
            if suffix in {".png", ".jpg", ".jpeg"}:
                images.append(path)
            elif suffix in {".mkv", ".mp4", ".avi", ".mov"}:
                videos.append(path)
    return images, videos, extensions


def _video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {
            "sequence_name": path.parent.name,
            "video_file": str(path),
            "read_status": "UNREADABLE",
        }
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    codec_int = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((codec_int >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")
    brightness: list[float] = []
    blur: list[float] = []
    for ratio in (0.1, 0.5, 0.9):
        if frame_count > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_count * ratio) - 1))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(gray.mean()))
        blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    capture.release()
    mean_brightness = sum(brightness) / len(brightness) if brightness else None
    if mean_brightness is None:
        lighting = "UNKNOWN"
    elif mean_brightness < 55:
        lighting = "NIGHT_AUTOMATIC_ESTIMATE"
    elif mean_brightness < 95:
        lighting = "TWILIGHT_AUTOMATIC_ESTIMATE"
    else:
        lighting = "DAY_AUTOMATIC_ESTIMATE"
    return {
        "sequence_name": path.parent.name,
        "camera_name": path.stem,
        "video_file": str(path),
        "read_status": "OK",
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "duration_seconds": round(frame_count / fps, 6) if fps else "",
        "width": width,
        "height": height,
        "codec": codec or "UNKNOWN",
        "camera_motion": "FIXED_FROM_DATASET_STRUCTURE",
        "lighting": lighting,
        "weather": "RAIN_OR_SNOW_NOT_SEPARATED",
        "weather_source": "DATASET_IDENTITY_MANUAL_REVIEW_REQUIRED",
        "mean_brightness_sample": round(mean_brightness, 4) if mean_brightness is not None else "",
        "mean_blur_sample": round(sum(blur) / len(blur), 4) if blur else "",
    }


def inspect_aau(
    path: Path,
    sample_size: int,
    full_scan: bool = False,
    skip_images: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not path.is_dir():
        raise ValueError(f"AAU parser yêu cầu thư mục đã giải nén: {path}")
    rgb_json = path / "aauRainSnow-rgb.json"
    thermal_json = path / "aauRainSnow-thermal.json"
    if not rgb_json.exists():
        raise RuntimeError(f"Không tìm thấy aauRainSnow-rgb.json tại {path}")

    with rgb_json.open(encoding="utf-8") as handle:
        rgb = json.load(handle)
    thermal: dict[str, Any] = {}
    if thermal_json.exists():
        with thermal_json.open(encoding="utf-8") as handle:
            thermal = json.load(handle)

    canonical_images, video_files, extension_counts = _canonical_files(path)
    categories = {int(row["id"]): str(row["name"]) for row in rgb.get("categories", [])}
    image_meta = {int(row["id"]): row for row in rgb.get("images", [])}
    rgb_images = list(image_meta.values())
    image_budget = len(rgb_images) if full_scan else min(sample_size, len(rgb_images))
    sampled_meta = reservoir_sample(rgb_images, image_budget, seed=231)
    sampled_ids = {int(row["id"]) for row in sampled_meta}
    quality_rows: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []

    if not skip_images:
        for meta in tqdm(sampled_meta, desc="AAU RGB image sample", disable=not progress):
            source = path / str(meta["file_name"])
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            file_size = source.stat().st_size if source.exists() else 0
            row = image_quality(image, file_size, str(source.relative_to(path)))
            row.update(
                dataset_name=DATASET,
                sequence_name=Path(str(meta["file_name"])).parent.name,
                split="NOT_PROVIDED",
            )
            quality_rows.append(row)
            if image is not None and source.exists():
                data = source.read_bytes()
                image_records.append(
                    {
                        "dataset_name": DATASET,
                        "sequence_name": Path(str(meta["file_name"])).parent.name,
                        "source_file": str(source.relative_to(path)).replace("\\", "/"),
                        "sha256": sha256_bytes(data),
                        "phash": perceptual_hash(image),
                        "width": image.shape[1],
                        "height": image.shape[0],
                    }
                )

    class_counts: Counter[str] = Counter()
    valid_class_counts: Counter[str] = Counter()
    bbox_320_counts: Counter[str] = Counter()
    bbox_size_counts: Counter[str] = Counter()
    distance_counts: Counter[str] = Counter()
    boxes_per_image: Counter[str] = Counter()
    invalid_rows: list[dict[str, Any]] = []
    bbox_samples: list[dict[str, Any]] = []
    annotated_ids: set[int] = set()
    valid_bbox_count = 0
    analyzed_bbox_count = 0

    for annotation in rgb.get("annotations", []):
        image_id = int(annotation.get("image_id", -1))
        category_id = int(annotation.get("category_id", -1))
        original_class = categories.get(category_id, f"UNKNOWN_CATEGORY_{category_id}")
        class_counts[original_class] += 1
        annotated_ids.add(image_id)
        boxes_per_image[str(image_id)] += 1
        bbox = annotation.get("bbox") or []
        meta = image_meta.get(image_id)
        if len(bbox) != 4 or not meta:
            invalid_rows.append(
                {
                    "dataset_name": DATASET,
                    "sequence_name": "UNKNOWN",
                    "source_file": str(meta.get("file_name", "")) if meta else "",
                    "annotation_id": annotation.get("id", ""),
                    "issue_type": "MALFORMED_BBOX_OR_MISSING_IMAGE_METADATA",
                    "severity": "ERROR",
                    "details": str(bbox),
                    "recommended_action": "MANUAL_REVIEW",
                }
            )
            continue
        x, y, width, height = map(float, bbox)
        xmin, ymin, xmax, ymax = x, y, x + width, y + height
        issues = validate_bbox(
            xmin,
            ymin,
            xmax,
            ymax,
            int(meta.get("width", 0)),
            int(meta.get("height", 0)),
        )
        if issues:
            for issue in issues:
                invalid_rows.append(
                    {
                        "dataset_name": DATASET,
                        "sequence_name": Path(str(meta["file_name"])).parent.name,
                        "source_file": meta["file_name"],
                        "annotation_id": annotation.get("id", ""),
                        "issue_type": issue,
                        "severity": "ERROR",
                        "details": str(bbox),
                        "recommended_action": "MANUAL_REVIEW",
                    }
                )
            continue
        valid_bbox_count += 1
        valid_class_counts[original_class] += 1
        if image_id not in sampled_ids and not full_scan:
            continue
        analyzed_bbox_count += 1
        image_width, image_height = int(meta["width"]), int(meta["height"])
        area_ratio = width * height / (image_width * image_height)
        letterbox = letterbox_box_metrics(width, height, image_width, image_height)
        bbox_320_counts[letterbox["box_320_category"]] += 1
        bbox_size_counts[relative_size_category(area_ratio)] += 1
        distance_counts[distance_proxy(area_ratio)] += 1
        if len(bbox_samples) < max(20000, sample_size * 20):
            bbox_samples.append(
                {
                    "dataset_name": DATASET,
                    "sequence_name": Path(str(meta["file_name"])).parent.name,
                    "source_file": meta["file_name"],
                    "original_class": original_class,
                    "mapped_class": "vehicle" if original_class in VEHICLE_CLASSES else "",
                    "box_width": round(width, 6),
                    "box_height": round(height, 6),
                    "box_area_ratio": round(area_ratio, 8),
                    "bbox_size_category": relative_size_category(area_ratio),
                    "distance": distance_proxy(area_ratio),
                    **letterbox,
                }
            )

    videos = [
        _video_metadata(video)
        for video in tqdm(sorted(video_files), desc="AAU video metadata", disable=not progress)
    ]
    unique_sequences = sorted({row["sequence_name"] for row in videos})
    lighting_counts = Counter(row.get("lighting", "UNKNOWN") for row in videos)
    conditions = [
        {
            "dataset_name": DATASET,
            "condition": "lighting",
            "value": key,
            "count": value,
            "unit": "video",
            "assessment_source": "AUTOMATIC_ESTIMATE",
        }
        for key, value in sorted(lighting_counts.items())
    ]
    conditions.append(
        {
            "dataset_name": DATASET,
            "condition": "weather",
            "value": "RAIN_OR_SNOW_NOT_SEPARATED",
            "count": len(unique_sequences),
            "unit": "sequence",
            "assessment_source": "DATASET_IDENTITY_MANUAL_REVIEW_REQUIRED",
        }
    )
    elapsed = time.perf_counter() - started
    return {
        "dataset_name": DATASET,
        "path": str(path.resolve()),
        "status": "ANALYZED",
        "data_type": "PAIRED_RGB_THERMAL_VIDEO_AND_COCO_INSTANCES",
        "image_count": len(rgb.get("images", [])) + len(thermal.get("images", [])),
        "rgb_image_count": len(rgb.get("images", [])),
        "thermal_image_count": len(thermal.get("images", [])),
        "canonical_png_file_count": extension_counts.get(".png", 0),
        "video_count": len(videos),
        "sequence_count": len(unique_sequences),
        "annotation_status": "PROVIDED_COCO_INSTANCE_ANNOTATIONS",
        "annotation_file_count": 1 + int(thermal_json.exists()),
        "annotation_row_count": len(rgb.get("annotations", [])),
        "bbox_count": len(rgb.get("annotations", [])),
        "valid_bbox_count": valid_bbox_count,
        "bbox_analyzed_count": analyzed_bbox_count,
        "track_count": 0,
        "class_counts": dict(class_counts),
        "valid_class_counts": dict(valid_class_counts),
        "images_without_boxes": len(set(image_meta) - annotated_ids),
        "annotation_without_image": len(annotated_ids - set(image_meta)),
        "invalid_annotations": invalid_rows,
        "quality_rows": quality_rows,
        "bbox_samples": bbox_samples,
        "bbox_320_counts": dict(bbox_320_counts),
        "bbox_size_counts": dict(bbox_size_counts),
        "distance_counts": dict(distance_counts),
        "boxes_per_image": boxes_per_image,
        "image_records": image_records,
        "sequences": videos,
        "conditions": conditions,
        "files_processed_successfully": sum(row.get("read_status") == "OK" for row in quality_rows)
        + sum(row.get("read_status") == "OK" for row in videos)
        + 1,
        "files_failed": sum(row.get("read_status") != "OK" for row in quality_rows)
        + sum(row.get("read_status") != "OK" for row in videos),
        "elapsed_seconds": round(elapsed, 3),
        "analysis_scope": "FULL_RGB_ANNOTATION_AND_IMAGE_SCAN" if full_scan else f"FULL_ANNOTATION_SAMPLE_{image_budget}_IMAGES",
        "duplicate_tree_excluded": True,
    }


__all__ = ["inspect_aau"]
