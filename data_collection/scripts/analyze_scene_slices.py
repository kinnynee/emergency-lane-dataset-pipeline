"""Phân tích cross-test theo road/weather/lighting/camera/density ở cấp sequence."""

from __future__ import annotations

from collections import Counter
from typing import Any

SCENE_DIMENSIONS = ("road_type", "weather", "lighting", "camera_view", "traffic_density")


def build_scene_slice_analysis(
    results: list[dict[str, Any]],
    scene_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_dataset = {str(result["dataset_name"]): result for result in results}
    sequence_rows: list[dict[str, Any]] = []
    class_counts: Counter[tuple[str, str, str, str]] = Counter()
    class_totals: Counter[tuple[str, str]] = Counter()
    bbox_counts: Counter[tuple[str, str, str, str]] = Counter()
    bbox_totals: Counter[tuple[str, str, str]] = Counter()

    for dataset_name, sequences in scene_config.get("assessments", {}).items():
        result = by_dataset.get(str(dataset_name), {})
        metadata_rows = result.get("sequences", [])
        quality_rows = result.get("quality_rows", [])
        all_bbox_rows = result.get("bbox_samples", [])
        for sequence_id, assessment in sequences.items():
            sequence_metadata = [
                row for row in metadata_rows if str(row.get("sequence_name")) == str(sequence_id)
            ]
            sequence_quality = [
                row for row in quality_rows if str(row.get("sequence_name")) == str(sequence_id)
            ]
            sequence_bboxes = [
                row for row in all_bbox_rows if str(row.get("sequence_name")) == str(sequence_id)
            ]
            metadata_image_counts = [
                int(row.get("image_count", 0) or 0) for row in sequence_metadata
                if int(row.get("image_count", 0) or 0) > 0
            ]
            metadata_bbox_counts = [
                int(row.get("bbox_count", 0) or 0) for row in sequence_metadata
                if int(row.get("bbox_count", 0) or 0) > 0
            ]
            image_count = max(metadata_image_counts) if metadata_image_counts else len(
                {str(row.get("source_file", "")) for row in sequence_quality}
            )
            bbox_count = max(metadata_bbox_counts) if metadata_bbox_counts else len(sequence_bboxes)
            bbox_sample_count = len(sequence_bboxes)
            size_320 = Counter(str(row.get("box_320_category", "NOT_COMPUTED")) for row in sequence_bboxes)
            difficult = size_320["EXTREMELY_TINY"] + size_320["VERY_SMALL"]
            sequence_rows.append(
                {
                    "dataset_name": dataset_name,
                    "sequence_id": sequence_id,
                    **{dimension: assessment.get(dimension, "UNKNOWN") for dimension in SCENE_DIMENSIONS},
                    "image_count_in_scope": image_count,
                    "bbox_count_in_scope": bbox_count,
                    "mean_vehicles_per_image": assessment.get("mean_vehicles_per_image", ""),
                    "bbox_size_sample_count": bbox_sample_count,
                    "extremely_tiny_320_count": size_320["EXTREMELY_TINY"],
                    "very_small_320_count": size_320["VERY_SMALL"],
                    "small_320_count": size_320["SMALL"],
                    "usable_320_count": size_320["USABLE"],
                    "difficult_under_8px_ratio": round(difficult / bbox_sample_count, 8)
                    if bbox_sample_count else "",
                    "count_scope": "FULL_SEQUENCE_XML"
                    if dataset_name == "UA-DETRAC Original"
                    else "FULL_ANNOTATED_RGB_IMAGES",
                    "bbox_size_scope": "ANALYSIS_SAMPLE"
                    if bbox_sample_count < bbox_count else "FULL_ANNOTATION",
                    "manual_review_status": assessment.get("manual_review_status", "PENDING"),
                }
            )

            for bbox in sequence_bboxes:
                original = str(bbox.get("original_class", "UNKNOWN") or "UNKNOWN")
                mapped = str(bbox.get("mapped_class", "") or "EXCLUDED_OR_UNMAPPED")
                for dimension in SCENE_DIMENSIONS:
                    value = str(assessment.get(dimension, "UNKNOWN"))
                    class_counts[(dimension, value, original, mapped)] += 1
                    class_totals[(dimension, value)] += 1
                for metric, field in (
                    ("ORIGINAL_RELATIVE_SIZE", "bbox_size_category"),
                    ("POST_RESIZE_320", "box_320_category"),
                ):
                    category = str(bbox.get(field, "NOT_COMPUTED") or "NOT_COMPUTED")
                    for dimension in SCENE_DIMENSIONS:
                        value = str(assessment.get(dimension, "UNKNOWN"))
                        bbox_counts[(dimension, value, metric, category)] += 1
                        bbox_totals[(dimension, value, metric)] += 1

    class_rows = [
        {
            "dimension": dimension,
            "value": value,
            "original_class": original,
            "mapped_class": mapped,
            "bbox_count_in_analysis": count,
            "ratio_within_slice": round(count / class_totals[(dimension, value)], 8),
            "analysis_scope": "BBOX_ANALYSIS_SAMPLE",
        }
        for (dimension, value, original, mapped), count in sorted(class_counts.items())
    ]
    bbox_rows = [
        {
            "dimension": dimension,
            "value": value,
            "metric": metric,
            "category": category,
            "bbox_count_in_analysis": count,
            "ratio_within_slice": round(count / bbox_totals[(dimension, value, metric)], 8),
            "analysis_scope": "BBOX_ANALYSIS_SAMPLE",
        }
        for (dimension, value, metric, category), count in sorted(bbox_counts.items())
    ]
    return sequence_rows, class_rows, bbox_rows


__all__ = ["SCENE_DIMENSIONS", "build_scene_slice_analysis"]
