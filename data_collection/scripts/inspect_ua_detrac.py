"""EDA UA-DETRAC Original từ ZIP, gồm track và stationary candidate heuristic."""

from __future__ import annotations

import random
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from external_eda_common import (
    clip_bbox_to_image,
    distance_proxy,
    image_from_bytes,
    image_quality,
    letterbox_box_metrics,
    perceptual_hash,
    relative_size_category,
    reservoir_sample,
    safe_sequence,
    sha256_bytes,
    validate_bbox,
)

DATASET = "UA-DETRAC Original"


def _split_from_xml_path(name: str) -> str:
    lower = name.lower()
    if "test-annotations" in lower:
        return "test"
    if "train-annotations" in lower:
        return "train"
    return "UNKNOWN"


def inspect_ua_detrac(
    path: Path,
    sample_size: int,
    full_scan: bool = False,
    skip_images: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise ValueError(f"UA-DETRAC parser yêu cầu ZIP: {path}")

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        image_infos = [
            info
            for info in infos
            if info.filename.lower().endswith((".jpg", ".jpeg", ".png"))
            and "/detrac-images/" in info.filename.lower()
        ]
        xml_infos = [
            info
            for info in infos
            if info.filename.lower().endswith(".xml")
            and "annotations-xml" in info.filename.lower()
        ]
        by_sequence: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        image_path_by_frame: dict[tuple[str, int], str] = {}
        for info in image_infos:
            sequence = safe_sequence(info.filename)
            by_sequence[sequence].append(info)
            try:
                frame_number = int(Path(info.filename).stem.lower().replace("img", ""))
                image_path_by_frame[(sequence, frame_number)] = info.filename
            except ValueError:
                pass
        for values in by_sequence.values():
            values.sort(key=lambda item: item.filename)

        first_per_sequence = [values[0] for values in by_sequence.values() if values]
        if full_scan:
            sampled_infos = image_infos
        elif sample_size <= len(first_per_sequence):
            sampled_infos = reservoir_sample(first_per_sequence, sample_size, seed=232)
        else:
            first_names = {info.filename for info in first_per_sequence}
            remaining = (info for info in image_infos if info.filename not in first_names)
            sampled_infos = first_per_sequence + reservoir_sample(
                remaining, sample_size - len(first_per_sequence), seed=232
            )
        sampled_names = {info.filename for info in sampled_infos}

        dimensions_by_sequence: dict[str, tuple[int, int]] = {}
        quality_rows: list[dict[str, Any]] = []
        image_records: list[dict[str, Any]] = []
        dimension_infos = {info.filename: info for info in first_per_sequence}
        for info in tqdm(
            list({**dimension_infos, **{info.filename: info for info in sampled_infos}}.values()),
            desc="UA image sample",
            disable=not progress,
        ):
            data = archive.read(info)
            image = image_from_bytes(data)
            sequence = safe_sequence(info.filename)
            if image is not None:
                dimensions_by_sequence[sequence] = (image.shape[1], image.shape[0])
            if info.filename not in sampled_names or skip_images:
                continue
            row = image_quality(image, info.file_size, info.filename)
            row.update(
                dataset_name=DATASET,
                sequence_name=sequence,
                split="FROM_ANNOTATION_DIRECTORY",
            )
            quality_rows.append(row)
            if image is not None:
                image_records.append(
                    {
                        "dataset_name": DATASET,
                        "sequence_name": sequence,
                        "source_file": info.filename,
                        "sha256": sha256_bytes(data),
                        "phash": perceptual_hash(image),
                        "width": image.shape[1],
                        "height": image.shape[0],
                    }
                )

        class_counts: Counter[str] = Counter()
        bbox_320_counts: Counter[str] = Counter()
        bbox_size_counts: Counter[str] = Counter()
        distance_counts: Counter[str] = Counter()
        boxes_per_image: Counter[str] = Counter()
        invalid_rows: list[dict[str, Any]] = []
        bbox_samples: list[dict[str, Any]] = []
        bbox_sample_limit = max(100000, sample_size * 20)
        bbox_seen = 0
        bbox_rng = random.Random(233)
        sequence_rows: list[dict[str, Any]] = []
        stationary_candidates: list[dict[str, Any]] = []
        total_boxes = 0
        valid_boxes = 0
        total_tracks = 0
        total_occluded = 0
        total_truncated = 0
        empty_frames = 0
        annotation_missing_image = 0
        ignored_region_count = 0
        boundary_clipped_bbox_count = 0
        boundary_clip_samples: list[dict[str, Any]] = []

        for xml_info in tqdm(xml_infos, desc="UA XML annotations", disable=not progress):
            split = _split_from_xml_path(xml_info.filename)
            root = ET.fromstring(archive.read(xml_info))
            sequence = root.attrib.get("name") or Path(xml_info.filename).stem
            sequence_attribute = root.find("sequence_attribute")
            weather = (
                sequence_attribute.attrib.get("sence_weather", "UNKNOWN")
                if sequence_attribute is not None
                else "UNKNOWN"
            )
            camera_state = (
                sequence_attribute.attrib.get("camera_state", "UNKNOWN")
                if sequence_attribute is not None
                else "UNKNOWN"
            )
            ignored_region_count += len(root.findall("./ignored_region/box"))
            image_width, image_height = dimensions_by_sequence.get(sequence, (0, 0))
            diagonal = (image_width**2 + image_height**2) ** 0.5 if image_width and image_height else 0
            seq_boxes = 0
            seq_occluded = 0
            seq_truncated = 0
            seq_frames = 0
            track_stats: dict[str, dict[str, Any]] = {}

            for frame in root.findall("frame"):
                seq_frames += 1
                frame_number = int(frame.attrib.get("num", seq_frames))
                targets = frame.findall("./target_list/target")
                if not targets:
                    empty_frames += 1
                if (sequence, frame_number) not in image_path_by_frame:
                    annotation_missing_image += 1
                image_key = f"{sequence}:{frame_number}"
                boxes_per_image[image_key] = len(targets)
                for target in targets:
                    total_boxes += 1
                    seq_boxes += 1
                    track_id = target.attrib.get("id", "UNKNOWN")
                    box = target.find("box")
                    attribute = target.find("attribute")
                    if box is None:
                        invalid_rows.append(
                            {
                                "dataset_name": DATASET,
                                "sequence_name": sequence,
                                "source_file": xml_info.filename,
                                "annotation_id": f"{frame_number}:{track_id}",
                                "issue_type": "MISSING_BOX",
                                "severity": "ERROR",
                                "details": "",
                                "recommended_action": "MANUAL_REVIEW",
                            }
                        )
                        continue
                    try:
                        left = float(box.attrib["left"])
                        top = float(box.attrib["top"])
                        width = float(box.attrib["width"])
                        height = float(box.attrib["height"])
                    except (KeyError, ValueError):
                        invalid_rows.append(
                            {
                                "dataset_name": DATASET,
                                "sequence_name": sequence,
                                "source_file": xml_info.filename,
                                "annotation_id": f"{frame_number}:{track_id}",
                                "issue_type": "NON_NUMERIC_BBOX",
                                "severity": "ERROR",
                                "details": str(box.attrib),
                                "recommended_action": "MANUAL_REVIEW",
                            }
                        )
                        continue
                    original_class = (
                        attribute.attrib.get("vehicle_type", "UNKNOWN")
                        if attribute is not None
                        else "UNKNOWN"
                    )
                    class_counts[original_class] += 1
                    raw_right = left + width
                    raw_bottom = top + height
                    issues = validate_bbox(
                        left,
                        top,
                        raw_right,
                        raw_bottom,
                        image_width or None,
                        image_height or None,
                    )
                    fatal_issues = [
                        issue
                        for issue in issues
                        if issue in {"NAN_OR_INFINITY", "NON_POSITIVE_SIZE"}
                    ]
                    if fatal_issues or (issues and not image_width and not image_height):
                        for issue in fatal_issues or issues:
                            invalid_rows.append(
                                {
                                    "dataset_name": DATASET,
                                    "sequence_name": sequence,
                                    "source_file": image_path_by_frame.get((sequence, frame_number), xml_info.filename),
                                    "annotation_id": f"{frame_number}:{track_id}",
                                    "issue_type": issue,
                                    "severity": "ERROR",
                                    "details": f"{left},{top},{width},{height}",
                                    "recommended_action": "REVIEW_MALFORMED_BOX",
                                }
                            )
                        continue
                    (clipped_left, clipped_top, clipped_right, clipped_bottom), adjustments = (
                        clip_bbox_to_image(
                            left,
                            top,
                            raw_right,
                            raw_bottom,
                            image_width,
                            image_height,
                        )
                    )
                    clipped_issues = validate_bbox(
                        clipped_left,
                        clipped_top,
                        clipped_right,
                        clipped_bottom,
                        image_width,
                        image_height,
                    )
                    if clipped_issues:
                        for issue in clipped_issues:
                            invalid_rows.append(
                                {
                                    "dataset_name": DATASET,
                                    "sequence_name": sequence,
                                    "source_file": image_path_by_frame.get((sequence, frame_number), xml_info.filename),
                                    "annotation_id": f"{frame_number}:{track_id}",
                                    "issue_type": issue,
                                    "severity": "ERROR",
                                    "details": f"raw={left},{top},{width},{height};clipped={clipped_left},{clipped_top},{clipped_right},{clipped_bottom}",
                                    "recommended_action": "EXCLUDE_ONLY_NO_VISIBLE_AREA_AFTER_CLIP",
                                }
                            )
                        continue
                    if adjustments:
                        boundary_clipped_bbox_count += 1
                        if len(boundary_clip_samples) < 1000:
                            boundary_clip_samples.append(
                                {
                                    "dataset_name": DATASET,
                                    "sequence_name": sequence,
                                    "source_file": image_path_by_frame.get((sequence, frame_number), ""),
                                    "annotation_id": f"{frame_number}:{track_id}",
                                    "original_class": original_class,
                                    "raw_bbox": f"{left},{top},{width},{height}",
                                    "clipped_bbox_xyxy": f"{clipped_left},{clipped_top},{clipped_right},{clipped_bottom}",
                                    "adjustments": "|".join(adjustments),
                                    "action": "KEEP_OBJECT_USE_CLIPPED_BOX",
                                }
                            )
                    left, top = clipped_left, clipped_top
                    width = clipped_right - clipped_left
                    height = clipped_bottom - clipped_top
                    valid_boxes += 1
                    area_ratio = (
                        width * height / (image_width * image_height)
                        if image_width and image_height
                        else 0.0
                    )
                    letterbox = (
                        letterbox_box_metrics(width, height, image_width, image_height)
                        if image_width and image_height
                        else {"box_320_category": "NOT_COMPUTED"}
                    )
                    bbox_320_counts[letterbox["box_320_category"]] += 1
                    if image_width and image_height:
                        bbox_size_counts[relative_size_category(area_ratio)] += 1
                        distance_counts[distance_proxy(area_ratio)] += 1
                    truncation = 0.0
                    if attribute is not None:
                        try:
                            truncation = float(attribute.attrib.get("truncation_ratio", 0) or 0)
                        except ValueError:
                            truncation = 0.0
                    is_truncated = truncation > 0
                    is_occluded = target.find("occlusion") is not None
                    total_truncated += int(is_truncated)
                    total_occluded += int(is_occluded)
                    seq_truncated += int(is_truncated)
                    seq_occluded += int(is_occluded)
                    center_x = left + width / 2
                    center_y = top + height / 2
                    stats = track_stats.setdefault(
                        track_id,
                        {
                            "first_frame": frame_number,
                            "last_frame": frame_number,
                            "count": 0,
                            "min_x": center_x,
                            "max_x": center_x,
                            "min_y": center_y,
                            "max_y": center_y,
                            "class": original_class,
                        },
                    )
                    stats["first_frame"] = min(stats["first_frame"], frame_number)
                    stats["last_frame"] = max(stats["last_frame"], frame_number)
                    stats["count"] += 1
                    stats["min_x"] = min(stats["min_x"], center_x)
                    stats["max_x"] = max(stats["max_x"], center_x)
                    stats["min_y"] = min(stats["min_y"], center_y)
                    stats["max_y"] = max(stats["max_y"], center_y)
                    bbox_seen += 1
                    bbox_sample = {
                        "dataset_name": DATASET,
                        "sequence_name": sequence,
                        "source_file": image_path_by_frame.get((sequence, frame_number), ""),
                        "original_class": original_class,
                        "mapped_class": "vehicle",
                        "raw_box_left": round(float(box.attrib["left"]), 6),
                        "raw_box_top": round(float(box.attrib["top"]), 6),
                        "raw_box_width": round(float(box.attrib["width"]), 6),
                        "raw_box_height": round(float(box.attrib["height"]), 6),
                        "boundary_clipped": bool(adjustments),
                        "boundary_clip_sides": "|".join(adjustments),
                        "box_width": round(width, 6),
                        "box_height": round(height, 6),
                        "box_area_ratio": round(area_ratio, 8) if image_width and image_height else "",
                        "bbox_size_category": relative_size_category(area_ratio) if image_width and image_height else "UNKNOWN",
                        "distance": distance_proxy(area_ratio) if image_width and image_height else "UNKNOWN",
                        "occluded": is_occluded,
                        "truncation_ratio": truncation,
                        **letterbox,
                    }
                    if len(bbox_samples) < bbox_sample_limit:
                        bbox_samples.append(bbox_sample)
                    else:
                        replacement = bbox_rng.randint(0, bbox_seen - 1)
                        if replacement < bbox_sample_limit:
                            bbox_samples[replacement] = bbox_sample

            total_tracks += len(track_stats)
            for track_id, stats in track_stats.items():
                if stats["count"] < 15 or not diagonal:
                    continue
                displacement = (
                    (stats["max_x"] - stats["min_x"]) ** 2
                    + (stats["max_y"] - stats["min_y"]) ** 2
                ) ** 0.5
                ratio = displacement / diagonal
                if ratio < 0.005:
                    confidence = "HIGH"
                elif ratio < 0.015:
                    confidence = "MEDIUM"
                else:
                    continue
                stationary_candidates.append(
                    {
                        "dataset_name": DATASET,
                        "sequence_name": sequence,
                        "track_id": track_id,
                        "original_class": stats["class"],
                        "first_frame": stats["first_frame"],
                        "last_frame": stats["last_frame"],
                        "track_frame_count": stats["count"],
                        "normalized_center_extent": round(ratio, 8),
                        "stationary_candidate": True,
                        "confidence": confidence,
                        "status": "STATIONARY_CANDIDATE",
                        "ground_truth": False,
                        "manual_review_status": "PENDING",
                    }
                )
            sequence_rows.append(
                {
                    "sequence_name": sequence,
                    "split": split,
                    "camera_state": camera_state,
                    "weather": weather,
                    "weather_source": "FROM_XML_METADATA",
                    "frame_count": seq_frames,
                    "image_count": len(by_sequence.get(sequence, [])),
                    "bbox_count": seq_boxes,
                    "track_count": len(track_stats),
                    "occluded_box_count": seq_occluded,
                    "truncated_box_count": seq_truncated,
                    "width": image_width,
                    "height": image_height,
                }
            )

    condition_counts = Counter(row["weather"] for row in sequence_rows)
    conditions = [
        {
            "dataset_name": DATASET,
            "condition": "weather",
            "value": key,
            "count": value,
            "unit": "sequence",
            "assessment_source": "FROM_XML_METADATA",
        }
        for key, value in sorted(condition_counts.items())
    ]
    elapsed = time.perf_counter() - started
    return {
        "dataset_name": DATASET,
        "path": str(path.resolve()),
        "status": "ANALYZED",
        "data_type": "IMAGE_SEQUENCES_WITH_TRACK_XML",
        "image_count": len(image_infos),
        "video_count": 0,
        "sequence_count": len(by_sequence),
        "annotation_status": "PROVIDED_TRACK_XML",
        "annotation_file_count": len(xml_infos),
        "annotation_row_count": total_boxes,
        "bbox_count": total_boxes,
        "valid_bbox_count": valid_boxes,
        "bbox_analyzed_count": valid_boxes,
        "track_count": total_tracks,
        "class_counts": dict(class_counts),
        "valid_class_counts": dict(class_counts),
        "images_without_boxes": empty_frames,
        "annotation_without_image": annotation_missing_image,
        "invalid_annotations": invalid_rows,
        "boundary_clipped_bbox_count": boundary_clipped_bbox_count,
        "boundary_clip_samples": boundary_clip_samples,
        "quality_rows": quality_rows,
        "bbox_samples": bbox_samples,
        "bbox_320_counts": dict(bbox_320_counts),
        "bbox_size_counts": dict(bbox_size_counts),
        "distance_counts": dict(distance_counts),
        "boxes_per_image": boxes_per_image,
        "image_records": image_records,
        "sequences": sequence_rows,
        "conditions": conditions,
        "stationary_candidates": stationary_candidates,
        "occluded_bbox_count": total_occluded,
        "truncated_bbox_count": total_truncated,
        "ignored_region_count": ignored_region_count,
        "files_processed_successfully": len(xml_infos)
        + sum(row.get("read_status") == "OK" for row in quality_rows),
        "files_failed": sum(row.get("read_status") != "OK" for row in quality_rows),
        "elapsed_seconds": round(elapsed, 3),
        "analysis_scope": "FULL_ANNOTATION_FULL_IMAGE_SCAN" if full_scan else f"FULL_ANNOTATION_SAMPLE_{len(sampled_infos)}_IMAGES",
    }


__all__ = ["inspect_ua_detrac"]
