"""Chạy EDA thật cho MIO-TCD Localization, AAU RainSnow và UA-DETRAC."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_scene_slices import build_scene_slice_analysis
from analyze_viewpoint_suitability import assess_viewpoints
from build_quality_audit import build_quality_audit
from create_balanced_subset_plan import create_plan
from create_external_contact_sheets import create_contact_sheet
from detect_external_duplicates import detect_duplicates
from detect_sequence_leakage import assert_sequence_split, detect_leakage
from discover_external_datasets import discover
from external_eda_common import DEFAULT_OUTPUT, ROOT, git_commit, load_yaml, read_csv, write_csv
from inspect_aau_rainsnow import inspect_aau
from inspect_mio_tcd import inspect_mio
from inspect_ua_detrac import inspect_ua_detrac
from validate_split_policy import build_split_audit

VERSION_DATE = "2026-08-02"
CACHE_SCHEMA_VERSION = 2

INVENTORY_FIELDS = [
    "dataset_name", "version_or_download_date", "source_path", "status", "data_type",
    "image_count", "video_count", "sequence_count", "annotation_status",
    "annotation_file_count", "annotation_row_count", "bbox_count", "bbox_analyzed_count",
    "track_count", "class_count", "images_without_boxes", "invalid_annotation_count",
    "invalid_issue_count", "boundary_clipped_bbox_count", "files_processed_successfully", "files_failed", "analysis_scope",
    "elapsed_seconds", "git_commit",
]
INVALID_FIELDS = [
    "dataset_name", "sequence_name", "source_file", "annotation_id", "issue_type",
    "severity", "details", "recommended_action",
]
QUALITY_SAMPLE_FIELDS = [
    "dataset_name", "sequence_name", "source_file", "split", "read_status", "width",
    "height", "aspect_ratio", "file_size_bytes", "mean_brightness", "brightness_std",
    "contrast", "laplacian_variance", "blur_score", "dark_pixel_ratio",
    "bright_pixel_ratio", "mean_saturation", "black_suspect", "white_suspect",
    "underexposed_suspect", "overexposed_suspect", "blur_suspect", "assessment_source",
]
BBOX_SAMPLE_FIELDS = [
    "dataset_name", "sequence_name", "source_file", "original_class", "mapped_class",
    "source_mapped_class", "class_mapping_status",
    "raw_box_left", "raw_box_top", "raw_box_width", "raw_box_height",
    "boundary_clipped", "boundary_clip_sides",
    "box_width", "box_height", "box_area_ratio", "bbox_size_category", "distance",
    "box_width_320", "box_height_320", "box_area_320", "box_320_category",
    "occluded", "truncation_ratio",
]
VIEWPOINT_FIELDS = [
    "dataset_name", "sequence_name", "camera_motion", "camera_height_estimate",
    "view_direction", "pitch_category", "road_area_visibility",
    "vehicle_scale_similarity", "fixed_camera_similarity", "emergency_lane_similarity",
    "night_suitability", "rain_suitability", "overall_score", "relevance_level",
    "assessment_source", "manual_review_status", "notes",
]
DUPLICATE_FIELDS = [
    "duplicate_group_id", "dataset_name", "sequence_name", "proposed_split", "file_path", "duplicate_type",
    "similarity_score", "recommended_keep", "recommended_action", "review_status",
]
LEAKAGE_FIELDS = [
    "dataset_name", "sequence_name", "leakage_type", "splits", "severity", "evidence",
    "recommended_action", "review_status",
]
MANIFEST_FIELDS = [
    "selection_id", "dataset_name", "sequence_id", "source_sequence_id", "source_file", "annotation_file",
    "original_class", "mapped_class", "lighting", "weather", "distance",
    "road_type", "bbox_size_category", "camera_view", "traffic_density",
    "mean_vehicles_per_image", "scene_metadata_source", "scene_metadata_review_status",
    "selection_reason", "target_subset", "selected", "manual_review_status", "notes",
]

DATASET_MAPPING_KEYS = {
    "MIO-TCD Localization": "mio_tcd",
    "AAU RainSnow": "aau_rainsnow",
    "UA-DETRAC Original": "ua_detrac",
}


def _portable_report_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return f"<EXTERNAL_DATA_ROOT>/{path.name}"


def _ratio(numerator: int, denominator: int) -> float | str:
    return round(numerator / denominator, 8) if denominator else ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_stat_signature(path: Path) -> dict[str, Any]:
    """Create a cheap source identity without hashing multi-gigabyte media."""

    resolved = path.resolve()
    if resolved.is_file():
        stat = resolved.stat()
        return {
            "kind": "FILE",
            "path": str(resolved),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for item in sorted((entry for entry in resolved.rglob("*") if entry.is_file())):
        stat = item.stat()
        relative = item.relative_to(resolved).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
        file_count += 1
        total_bytes += stat.st_size
    return {
        "kind": "DIRECTORY",
        "path": str(resolved),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_stat_sha256": digest.hexdigest(),
    }


def _cache_identity(
    dataset_key: str,
    source_path: Path,
    inspector: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    inspector_module = sys.modules[inspector.__module__]
    code_paths = [
        Path(__file__).resolve(),
        Path(inspector_module.__file__).resolve(),
        (ROOT / "scripts" / "external_eda_common.py").resolve(),
    ]
    config_paths = sorted((ROOT / "configs").glob("*.yaml"))
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset_key": dataset_key,
        "source": _source_stat_signature(source_path),
        "options": {
            "sample_size": int(args.sample_size),
            "full_scan": bool(args.full_scan),
            "skip_images": bool(args.skip_images),
        },
        "code_sha256": {path.name: _sha256_file(path) for path in code_paths},
        "config_sha256": {path.name: _sha256_file(path) for path in config_paths},
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "fingerprint": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "payload": payload,
    }


def _save_cache(
    path: Path,
    result: dict[str, Any],
    identity: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content: dict[str, Any]
    if identity is None:
        content = result
    else:
        content = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "fingerprint": identity["fingerprint"],
            "identity": identity["payload"],
            "result": result,
        }
    path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")


def _load_cache(
    path: Path,
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if "result" not in content or "fingerprint" not in content:
        return content if expected_identity is None else None
    if expected_identity is not None and content["fingerprint"] != expected_identity["fingerprint"]:
        return None
    return content["result"]


def _summary_numeric(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
    if not values:
        return {"count": 0, "min": "", "mean": "", "median": "", "max": ""}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": round(float(array.min()), 6),
        "mean": round(float(array.mean()), 6),
        "median": round(float(np.median(array)), 6),
        "max": round(float(array.max()), 6),
    }


def _inventory(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    commit = git_commit()
    rows = []
    for result in results:
        invalid_rows = result.get("invalid_annotations", [])
        unique_invalid = {
            (str(row.get("source_file", "")), str(row.get("annotation_id", "")))
            for row in invalid_rows
        }
        rows.append(
            {
                "dataset_name": result["dataset_name"],
                "version_or_download_date": VERSION_DATE,
                "source_path": _portable_report_path(result.get("path", "")) if result.get("path") else "",
                "status": result.get("status", "UNKNOWN"),
                "data_type": result.get("data_type", ""),
                "image_count": result.get("image_count", 0),
                "video_count": result.get("video_count", 0),
                "sequence_count": result.get("sequence_count", 0),
                "annotation_status": result.get("annotation_status", ""),
                "annotation_file_count": result.get("annotation_file_count", 0),
                "annotation_row_count": result.get("annotation_row_count", 0),
                "bbox_count": result.get("bbox_count", 0),
                "bbox_analyzed_count": result.get("bbox_analyzed_count", 0),
                "track_count": result.get("track_count", 0),
                "class_count": len(result.get("class_counts", {})),
                "images_without_boxes": result.get("images_without_boxes", 0),
                "invalid_annotation_count": len(unique_invalid),
                "invalid_issue_count": len(invalid_rows),
                "boundary_clipped_bbox_count": result.get("boundary_clipped_bbox_count", 0),
                "files_processed_successfully": result.get("files_processed_successfully", 0),
                "files_failed": result.get("files_failed", 0),
                "analysis_scope": result.get("analysis_scope", ""),
                "elapsed_seconds": result.get("elapsed_seconds", 0),
                "git_commit": commit,
            }
        )
    return rows


def _class_rows(results: list[dict[str, Any]], mapping: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        mapping_rows = mapping.get(DATASET_MAPPING_KEYS[result["dataset_name"]], {})
        for original_class, count in sorted(result.get("class_counts", {}).items()):
            rule = mapping_rows.get(original_class, {})
            requires_review = rule.get("review_required", True)
            track_exclusions = rule.get("track_exclusions", [])
            rows.append(
                {
                    "dataset_name": result["dataset_name"],
                    "original_class": original_class,
                    "count": count,
                    "mapped_class": rule.get("mapped_class", ""),
                    "include_for_training": (
                        "TRUE_WITH_TRACK_EXCLUSION"
                        if rule.get("include") and track_exclusions
                        else rule.get("include", "")
                    ),
                    "reason": rule.get("review_note", "SUPERVISOR_CLASS_POLICY_2026_08_02"),
                    "manual_review_required": requires_review,
                    "mapping_status": (
                        "UNMAPPED"
                        if original_class not in mapping_rows
                        else "DEFINED_REVIEW_REQUIRED"
                        if requires_review
                        else "DATA_LEAD_APPROVED_WITH_TRACK_EXCLUSION"
                        if track_exclusions
                        else "APPROVED"
                    ),
                }
            )
    return rows


def _normalize_bbox_class_mapping(
    results: list[dict[str, Any]],
    mapping: dict[str, Any],
) -> None:
    for result in results:
        rules = mapping.get(DATASET_MAPPING_KEYS[result["dataset_name"]], {})
        for row in result.get("bbox_samples", []):
            original_class = str(row.get("original_class", ""))
            source_mapped = str(row.get("mapped_class", "") or "")
            rule = rules.get(original_class)
            expected = str((rule or {}).get("mapped_class") or "")
            row["source_mapped_class"] = source_mapped
            row["mapped_class"] = expected
            if rule is None:
                row["class_mapping_status"] = "UNMAPPED_CLASS"
            elif source_mapped == expected:
                row["class_mapping_status"] = "CONSISTENT"
            else:
                row["class_mapping_status"] = "CORRECTED_TO_CONFIG"


def _annotation_quality(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        checked = int(result.get("bbox_analyzed_count", 0))
        invalid_rows = result.get("invalid_annotations", [])
        invalid = len(
            {
                (str(row.get("source_file", "")), str(row.get("annotation_id", "")))
                for row in invalid_rows
            }
        )
        valid = int(result.get("valid_bbox_count", checked))
        rows.append(
            {
                "dataset_name": result["dataset_name"],
                "version_or_download_date": VERSION_DATE,
                "annotation_status": result.get("annotation_status", ""),
                "annotation_files_checked": result.get("annotation_file_count", 0),
                "annotations_reported": result.get("annotation_row_count", 0),
                "bounding_boxes_checked": checked,
                "valid_bounding_boxes": valid,
                "invalid_annotations_unique": invalid,
                "invalid_issue_count": len(invalid_rows),
                "boundary_clipped_bbox_count": result.get("boundary_clipped_bbox_count", 0),
                "invalid_annotation_rate": _ratio(invalid, valid + invalid),
                "images_without_boxes": result.get("images_without_boxes", 0),
                "annotation_without_image": result.get("annotation_without_image", 0),
                "assessment_source": "SOURCE_ANNOTATION_SCAN",
                "bbox_boundary_policy": "CLIP_TO_IMAGE_AND_KEEP_OBJECT",
                "notes": "Một annotation có thể sinh nhiều issue; unique và issue được tách riêng.",
            }
        )
    return rows


def _image_quality(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples = [row for result in results for row in result.get("quality_rows", [])]
    summaries: list[dict[str, Any]] = []
    for result in results:
        rows = [row for row in result.get("quality_rows", []) if row.get("read_status") == "OK"]
        brightness = _summary_numeric(rows, "mean_brightness")
        contrast = _summary_numeric(rows, "contrast")
        blur = _summary_numeric(rows, "blur_score")
        size = _summary_numeric(rows, "file_size_bytes")
        summaries.append(
            {
                "dataset_name": result["dataset_name"],
                "version_or_download_date": VERSION_DATE,
                "sample_images_checked": len(rows),
                "images_failed": sum(row.get("read_status") != "OK" for row in result.get("quality_rows", [])),
                "brightness_mean": brightness["mean"],
                "brightness_median": brightness["median"],
                "contrast_mean": contrast["mean"],
                "blur_score_mean": blur["mean"],
                "blur_score_median": blur["median"],
                "file_size_mean_bytes": size["mean"],
                "underexposed_suspects": sum(bool(row.get("underexposed_suspect")) for row in rows),
                "overexposed_suspects": sum(bool(row.get("overexposed_suspect")) for row in rows),
                "blur_suspects": sum(bool(row.get("blur_suspect")) for row in rows),
                "assessment_source": "AUTOMATIC_ESTIMATE",
                "analysis_scope": result.get("analysis_scope", ""),
            }
        )
    return summaries, samples


def _bbox_statistics(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    samples = [row for result in results for row in result.get("bbox_samples", [])]
    for result in results:
        counts = Counter(result.get("bbox_320_counts", {}))
        analyzed = int(result.get("bbox_analyzed_count", 0))
        difficult = counts["EXTREMELY_TINY"] + counts["VERY_SMALL"]
        rows = result.get("bbox_samples", [])
        area = _summary_numeric(rows, "box_area_ratio")
        width_320 = _summary_numeric(rows, "box_width_320")
        height_320 = _summary_numeric(rows, "box_height_320")
        summaries.append(
            {
                "dataset_name": result["dataset_name"],
                "version_or_download_date": VERSION_DATE,
                "bbox_reported": result.get("bbox_count", 0),
                "bbox_analyzed": analyzed,
                "extremely_tiny_count": counts["EXTREMELY_TINY"],
                "very_small_count": counts["VERY_SMALL"],
                "small_count": counts["SMALL"],
                "usable_count": counts["USABLE"],
                "not_computed_count": counts["NOT_COMPUTED"],
                "extremely_tiny_ratio": _ratio(counts["EXTREMELY_TINY"], analyzed),
                "very_small_ratio": _ratio(counts["VERY_SMALL"], analyzed),
                "difficult_under_8px_ratio": _ratio(difficult, analyzed),
                "box_area_ratio_mean": area["mean"],
                "box_width_320_mean": width_320["mean"],
                "box_height_320_mean": height_320["mean"],
                "threshold_source": "INITIAL_ANALYSIS_THRESHOLDS_FROM_PROJECT_PROMPT",
            }
        )
    return summaries, samples


def _comparison(
    results: list[dict[str, Any]],
    bbox_stats: list[dict[str, Any]],
    annotation_quality: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bbox_by_name = {row["dataset_name"]: row for row in bbox_stats}
    quality_by_name = {row["dataset_name"]: row for row in annotation_quality}
    rows = []
    for result in results:
        name = result["dataset_name"]
        classes = set(result.get("class_counts", {}))
        conditions = result.get("conditions", [])
        condition_values = " ".join(str(row.get("value", "")).lower() for row in conditions)
        rows.append(
            {
                "dataset_name": name,
                "data_type": result.get("data_type", ""),
                "image_count": result.get("image_count", 0),
                "video_count": result.get("video_count", 0),
                "sequence_count": result.get("sequence_count", 0),
                "has_bounding_box": result.get("bbox_count", 0) > 0,
                "has_track_id": result.get("track_count", 0) > 0,
                "has_weather_label": "FROM_XML_METADATA" if name == "UA-DETRAC Original" else "DATASET_LEVEL_OR_NOT_SEPARATED" if name == "AAU RainSnow" else "NOT_VERIFIED",
                "has_day": "MANUAL_SEQUENCE_REVIEW" if name == "AAU RainSnow" else "NOT_VERIFIED",
                "has_night": "MANUAL_SEQUENCE_REVIEW" if name == "AAU RainSnow" else "NOT_VERIFIED",
                "has_rain": "DATASET_LEVEL" if name == "AAU RainSnow" else ("TRUE" if "rainy" in condition_values else "NOT_VERIFIED"),
                "has_snow": "DATASET_LEVEL_NOT_SEPARATED" if name == "AAU RainSnow" else "NOT_VERIFIED",
                "has_wet_road": "MANUAL_REVIEW_REQUIRED",
                "has_headlight": "MANUAL_REVIEW_REQUIRED",
                "fixed_camera": "TRUE_OR_HIGH_SIMILARITY",
                "elevated_view": "AUTOMATIC_ESTIMATE_REVIEW_REQUIRED",
                "has_motorcycle": bool(classes & {"motorcycle", "motorbike"}),
                "has_car": "car" in classes,
                "has_truck": bool(classes & {"truck", "articulated_truck", "single_unit_truck", "pickup_truck"}),
                "has_bus": "bus" in classes,
                "has_negative_sample": result.get("images_without_boxes", 0) > 0,
                "small_box_ratio": _ratio(
                    int(result.get("bbox_320_counts", {}).get("SMALL", 0)),
                    int(result.get("bbox_analyzed_count", 0)),
                ),
                "very_small_after_320_ratio": bbox_by_name[name].get("difficult_under_8px_ratio", ""),
                "annotation_quality": f"invalid_unique={quality_by_name[name]['invalid_annotations_unique']};issues={quality_by_name[name]['invalid_issue_count']}",
                "duplicate_risk": "TEMPORAL_HIGH" if name != "MIO-TCD Localization" else "SAMPLE_CHECK_REQUIRED",
                "train_suitability": "HIGH_WITH_BALANCED_SUBSET",
                "validation_suitability": "SEQUENCE_LEVEL_ONLY",
                "cross_domain_test_suitability": "YES_SEPARATE_SEQUENCES",
                "main_test_suitability": "NO_K230_MUST_BE_PRIMARY",
                "limitations": "Không có ground-truth xe dừng; điều kiện/góc cần review thủ công.",
                "recommendation": "Dùng train/validation/external test; không thay thế main K230 test.",
            }
        )
    return rows


def _gap_rows() -> list[dict[str, Any]]:
    conditions = [
        "K230 góc cao", "Xe trong làn dừng khẩn cấp", "Xe đang chạy", "Xe chạy chậm",
        "Xe dừng", "Xe dừng từ 2–3 giây", "Xe dừng trên 5 giây", "Xe rời khỏi ROI",
        "Ban ngày", "Ban đêm", "Mưa", "Đường ướt", "Ngược sáng", "Đèn pha",
        "Xe máy", "Ô tô", "Xe tải", "Xe khách", "Xe gần", "Xe xa", "Một xe",
        "Nhiều xe", "Không có xe", "Camera rung", "Bóng cây", "Biển báo",
        "Vật thể gây nhiễu",
    ]
    rows = []
    stationary = {
        "Xe dừng", "Xe dừng từ 2–3 giây", "Xe dừng trên 5 giây", "Xe rời khỏi ROI",
        "Xe trong làn dừng khẩn cấp",
    }
    for condition in conditions:
        if condition in stationary:
            mio = aau = ua = "NOT_VERIFIED"
            status = "NOT_VERIFIED"
            gap = "CRITICAL"
            evidence = "Không có ground-truth trạng thái dừng/ROI."
            action = "Thu và gán nhãn sequence K230 có ROI/tracking."
        elif condition in {"Mưa", "Đường ướt", "Ban đêm", "Đèn pha"}:
            mio = "NOT_VERIFIED"
            aau = "DATASET_LEVEL_OR_MANUAL_SEQUENCE_REVIEW"
            ua = "PARTIAL" if condition == "Mưa" else "NOT_VERIFIED"
            status = "PARTIAL"
            gap = "HIGH"
            evidence = "AAU dataset identity/video estimate; UA XML weather cho mưa."
            action = "Review thủ công và bổ sung K230 cùng điều kiện."
        elif condition == "Xe máy":
            mio = "CLASS_PRESENT"
            aau = "CLASS_PRESENT"
            ua = "CLASS_NOT_OBSERVED_IN_XML"
            status = "PARTIAL"
            gap = "MEDIUM"
            evidence = "Class distribution từ annotation."
            action = "Giữ xe máy MIO/AAU và quay thêm K230."
        else:
            mio = "PARTIAL_OR_NOT_VERIFIED"
            aau = "PARTIAL_OR_NOT_VERIFIED"
            ua = "PARTIAL_OR_NOT_VERIFIED"
            status = "PARTIAL"
            gap = "MEDIUM"
            evidence = "BBox/sequence có thể hỗ trợ nhưng chưa chứng minh điều kiện cụ thể."
            action = "Manual review contact sheet và thu K230."
        rows.append(
            {
                "condition": condition,
                "mio_tcd_coverage": mio,
                "aau_rainsnow_coverage": aau,
                "ua_detrac_coverage": ua,
                "k230_required": True,
                "coverage_status": status,
                "evidence": evidence,
                "gap_level": gap,
                "recommended_action": action,
                "notes": "BBox phương tiện không phải bằng chứng xe đang dừng.",
            }
        )
    return rows


def _boxes_per_image_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        distribution = Counter(int(value) for value in result.get("boxes_per_image", {}).values())
        for boxes_per_image, image_count in sorted(distribution.items()):
            rows.append(
                {
                    "dataset_name": result["dataset_name"],
                    "boxes_per_image": boxes_per_image,
                    "image_count": image_count,
                    "analysis_scope": result.get("analysis_scope", ""),
                }
            )
    return rows


def _ua_annotation_attribute_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ua = next(
        (result for result in results if result["dataset_name"] == "UA-DETRAC Original"),
        {},
    )
    analyzed = int(ua.get("bbox_analyzed_count", 0))
    occluded = int(ua.get("occluded_bbox_count", 0))
    truncated = int(ua.get("truncated_bbox_count", 0))
    return [
        {"metric": "OCCLUDED", "count": occluded, "analysis_scope": ua.get("analysis_scope", "")},
        {"metric": "NOT_MARKED_OCCLUDED", "count": max(0, analyzed - occluded), "analysis_scope": ua.get("analysis_scope", "")},
        {"metric": "TRUNCATED", "count": truncated, "analysis_scope": ua.get("analysis_scope", "")},
        {"metric": "NOT_TRUNCATED", "count": max(0, analyzed - truncated), "analysis_scope": ua.get("analysis_scope", "")},
    ]


def _apply_scene_metadata(
    splits: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    scene_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assessments = scene_config.get("assessments", {})
    for row in splits:
        assessment = (
            assessments.get(str(row["dataset_name"]), {})
            .get(str(row["sequence_id"]), {})
        )
        if assessment:
            row["road_type"] = assessment.get("road_type", "UNKNOWN")
            row["road_type_assessment_source"] = assessment.get("assessment_source", "UNKNOWN")
            row["road_type_manual_review_status"] = assessment.get("manual_review_status", "PENDING")
            row["road_type_notes"] = assessment.get("evidence", "")
            for field in ("weather", "lighting", "camera_view", "traffic_density"):
                row[field] = assessment.get(field, "UNKNOWN")
                row[f"{field}_source"] = assessment.get(f"{field}_source", "UNKNOWN")
            row["mean_vehicles_per_image"] = assessment.get("mean_vehicles_per_image", "")
            row["scene_metadata_manual_review_status"] = assessment.get(
                "manual_review_status", "PENDING"
            )
            row["scene_metadata_notes"] = assessment.get("evidence", "")
    for row in manifest:
        assessment = (
            assessments.get(str(row["dataset_name"]), {})
            .get(str(row["sequence_id"]), {})
        )
        row["road_type"] = assessment.get("road_type", "UNKNOWN") if assessment else "UNKNOWN"
        if assessment:
            for field in ("weather", "lighting", "camera_view", "traffic_density"):
                row[field] = assessment.get(field, "UNKNOWN")
            row["mean_vehicles_per_image"] = assessment.get("mean_vehicles_per_image", "")
            row["scene_metadata_source"] = assessment.get("assessment_source", "UNKNOWN")
            row["scene_metadata_review_status"] = assessment.get(
                "manual_review_status", "PENDING"
            )
    counts: Counter[tuple[str, str, str]] = Counter(
        (
            str(row.get("proposed_split", "UNKNOWN")),
            str(row.get("dataset_name", "UNKNOWN")),
            str(row.get("road_type", "UNKNOWN")),
        )
        for row in splits
    )
    road_type_distribution = [
        {
            "proposed_split": split,
            "dataset_name": dataset,
            "road_type": road_type,
            "sequence_count": count,
            "assessment_scope": "SEQUENCE_LEVEL",
        }
        for (split, dataset, road_type), count in sorted(counts.items())
    ]
    scene_counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in splits:
        for dimension in ("weather", "lighting", "camera_view", "traffic_density"):
            scene_counts[
                (
                    str(row.get("proposed_split", "UNKNOWN")),
                    str(row.get("dataset_name", "UNKNOWN")),
                    dimension,
                    str(row.get(dimension, "UNKNOWN")),
                )
            ] += 1
    scene_distribution = [
        {
            "proposed_split": split,
            "dataset_name": dataset,
            "dimension": dimension,
            "value": value,
            "sequence_count": count,
            "assessment_scope": "SEQUENCE_LEVEL",
        }
        for (split, dataset, dimension, value), count in sorted(scene_counts.items())
    ]
    return road_type_distribution, scene_distribution


def _scene_assessment_rows(scene_config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_name, sequences in scene_config.get("assessments", {}).items():
        for sequence_id, assessment in sequences.items():
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "sequence_id": sequence_id,
                    "road_type": assessment.get("road_type", "UNKNOWN"),
                    "weather": assessment.get("weather", "UNKNOWN"),
                    "lighting": assessment.get("lighting", "UNKNOWN"),
                    "camera_view": assessment.get("camera_view", "UNKNOWN"),
                    "traffic_density": assessment.get("traffic_density", "UNKNOWN"),
                    "mean_vehicles_per_image": assessment.get("mean_vehicles_per_image", ""),
                    "weather_source": assessment.get("weather_source", "UNKNOWN"),
                    "lighting_source": assessment.get("lighting_source", "UNKNOWN"),
                    "camera_view_source": assessment.get("camera_view_source", "UNKNOWN"),
                    "traffic_density_source": assessment.get("traffic_density_source", "UNKNOWN"),
                    "manual_review_status": assessment.get("manual_review_status", "PENDING"),
                    "evidence": assessment.get("evidence", ""),
                }
            )
    return rows


def _plot_no_data(title: str, path: Path, message: str = "CHƯA CÓ DỮ LIỆU XÁC MINH") -> None:
    figure = plt.figure(figsize=(8, 5))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.axis("off")
    plt.title(f"{title}\nDataset version: {VERSION_DATE}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _bar(title: str, ylabel: str, labels: list[str], values: list[float], path: Path) -> None:
    if not labels:
        _plot_no_data(title, path)
        return
    figure = plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title(f"{title}\nDataset version: {VERSION_DATE}")
    plt.xlabel("Nhóm")
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _hist(title: str, xlabel: str, values: list[float], path: Path) -> None:
    if not values:
        _plot_no_data(title, path)
        return
    figure = plt.figure(figsize=(8, 5))
    plt.hist(values, bins=40)
    plt.title(f"{title}\nDataset version: {VERSION_DATE}")
    plt.xlabel(xlabel)
    plt.ylabel("Tần suất")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _hbar(title: str, xlabel: str, labels: list[str], values: list[float], path: Path) -> None:
    if not labels:
        _plot_no_data(title, path)
        return
    height = max(6.0, min(12.0, 0.38 * len(labels) + 2.5))
    figure = plt.figure(figsize=(10, height))
    positions = np.arange(len(labels))
    plt.barh(positions, values)
    plt.yticks(positions, labels, fontsize=8)
    plt.gca().invert_yaxis()
    plt.title(f"{title}\nDataset version: {VERSION_DATE}")
    plt.xlabel(xlabel)
    plt.ylabel("Class gốc")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _figures(
    output: Path,
    inventory: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    quality_samples: list[dict[str, Any]],
    bbox_samples: list[dict[str, Any]],
    bbox_stats: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    viewpoints: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    boxes_per_image_distribution: list[dict[str, Any]],
    ua_annotation_attributes: list[dict[str, Any]],
) -> list[Path]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    names = [row["dataset_name"] for row in inventory]
    created: list[Path] = []

    def destination(number: int, slug: str) -> Path:
        path = figures / f"{number:02d}_{slug}.png"
        created.append(path)
        return path

    _bar("Số ảnh theo dataset", "Số ảnh", names, [float(row["image_count"]) for row in inventory], destination(1, "images_by_dataset"))
    _bar("Số video/sequence theo dataset", "Số video + sequence", names, [float(row["video_count"]) + float(row["sequence_count"]) for row in inventory], destination(2, "videos_sequences_by_dataset"))
    _bar("Số bounding box theo dataset", "Số box", names, [float(row["bbox_count"]) for row in inventory], destination(3, "bboxes_by_dataset"))
    original_labels = [f"{row['dataset_name'][:8]}:{row['original_class']}" for row in class_rows]
    _hbar("Phân bố class gốc", "Số annotation", original_labels, [float(row["count"]) for row in class_rows], destination(4, "original_class_distribution"))
    mapped = Counter()
    for row in class_rows:
        mapped[str(row.get("mapped_class") or "EXCLUDED_OR_PENDING")] += int(row["count"])
    _bar("Phân bố class sau ánh xạ proposal", "Số annotation", list(mapped), list(mapped.values()), destination(5, "mapped_class_distribution"))
    lighting = [row for row in conditions if row.get("condition") == "lighting"]
    _bar("Phân bố ngày/đêm tự động", "Số đơn vị", [str(row["value"]) for row in lighting], [float(row["count"]) for row in lighting], destination(6, "lighting_distribution"))
    weather = [row for row in conditions if row.get("condition") == "weather"]
    _bar("Phân bố điều kiện thời tiết", "Số đơn vị", [f"{row['dataset_name'][:8]}:{row['value']}" for row in weather], [float(row["count"]) for row in weather], destination(7, "weather_distribution"))
    _hist("Phân bố độ sáng", "Mean brightness (0-255)", [float(row["mean_brightness"]) for row in quality_samples if row.get("mean_brightness") not in ("", None)], destination(8, "brightness_distribution"))
    _hist("Phân bố blur score", "Variance of Laplacian", [float(row["blur_score"]) for row in quality_samples if row.get("blur_score") not in ("", None)], destination(9, "blur_distribution"))
    resolution_counts = Counter(f"{row.get('width')}x{row.get('height')}" for row in quality_samples if row.get("width"))
    _bar("Phân bố độ phân giải", "Số ảnh mẫu", list(resolution_counts), list(resolution_counts.values()), destination(10, "resolution_distribution"))
    _hist("Phân bố box area ratio", "Box area / image area", [float(row["box_area_ratio"]) for row in bbox_samples if row.get("box_area_ratio") not in ("", None)], destination(11, "bbox_area_ratio"))
    _hist("Phân bố box width sau letterbox 320", "Pixel", [float(row["box_width_320"]) for row in bbox_samples if row.get("box_width_320") not in ("", None)], destination(12, "bbox_width_320"))
    _hist("Phân bố box height sau letterbox 320", "Pixel", [float(row["box_height_320"]) for row in bbox_samples if row.get("box_height_320") not in ("", None)], destination(13, "bbox_height_320"))
    categories = ["EXTREMELY_TINY", "VERY_SMALL", "SMALL", "USABLE"]
    category_counts = Counter(row.get("box_320_category", "") for row in bbox_samples)
    _bar("Tỷ lệ tiny/small/usable box trong mẫu", "Số box", categories, [category_counts[key] for key in categories], destination(14, "bbox_320_categories"))
    all_counts = [
        int(row["boxes_per_image"])
        for row in boxes_per_image_distribution
        for _ in range(int(row["image_count"]))
    ]
    _hist("Số box mỗi ảnh/frame", "Số box", [float(value) for value in all_counts], destination(15, "boxes_per_image"))
    ua_metrics = {str(row["metric"]): int(row["count"]) for row in ua_annotation_attributes}
    _bar("Occlusion UA-DETRAC", "Số box", ["Occluded", "Not marked occluded"], [float(ua_metrics.get("OCCLUDED", 0)), float(ua_metrics.get("NOT_MARKED_OCCLUDED", 0))], destination(16, "ua_occlusion"))
    _bar("Truncation UA-DETRAC", "Số box", ["Truncated", "Not truncated"], [float(ua_metrics.get("TRUNCATED", 0)), float(ua_metrics.get("NOT_TRUNCATED", 0))], destination(17, "ua_truncation"))
    vp = defaultdict(list)
    for row in viewpoints:
        if row["dataset_name"] != "RADIATE":
            vp[row["dataset_name"]].append(float(row["overall_score"]))
    _bar("Điểm phù hợp góc camera", "Điểm trung bình (1-5)", list(vp), [sum(values) / len(values) for values in vp.values()], destination(18, "viewpoint_score"))
    pilot = [row for row in plans if row["scenario"] == "PILOT_500"]
    _bar("Số ảnh đề xuất chọn — PILOT_500", "Số ảnh proposal", [row["dataset_name"] for row in pilot], [float(row["proposed_images"]) for row in pilot], destination(19, "selection_ratio"))
    gap_counts = Counter(row["gap_level"] for row in gaps)
    _bar("Khoảng trống dữ liệu theo mức", "Số điều kiện", list(gap_counts), list(gap_counts.values()), destination(20, "data_gaps"))
    return created


def _audit_figures(
    output: Path,
    cross_sequence_stats: list[dict[str, Any]],
    quality_audit: list[dict[str, Any]],
    split_distribution: list[dict[str, Any]],
) -> list[Path]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    difficulty_path = figures / "21_cross_test_bbox_difficulty.png"
    _bar(
        "BBox dưới 8 px sau resize theo cross-test sequence",
        "Tỷ lệ",
        [str(row["sequence_id"]) for row in cross_sequence_stats],
        [float(row.get("difficult_under_8px_ratio") or 0) for row in cross_sequence_stats],
        difficulty_path,
    )
    created.append(difficulty_path)

    quality_path = figures / "22_quality_issue_counts.png"
    if quality_audit:
        labels = [str(row["dataset_name"]).replace(" Localization", "") for row in quality_audit]
        metrics = [
            ("Invalid annotation", "invalid_annotations_unique"),
            ("Blur suspect", "blur_suspects"),
            ("Exact duplicate group", "exact_duplicate_groups"),
            ("Near duplicate group", "near_duplicate_groups"),
        ]
        positions = np.arange(len(labels))
        width = 0.18
        figure = plt.figure(figsize=(10, 5.5))
        for index, (label, field) in enumerate(metrics):
            values = [float(row.get(field, 0)) for row in quality_audit]
            plt.bar(positions + (index - 1.5) * width, values, width, label=label)
        plt.yscale("symlog", linthresh=1)
        plt.xticks(positions, labels, rotation=15, ha="right")
        plt.ylabel("Số lượng (symlog)")
        plt.title(f"Quality issues theo dataset\nDataset version: {VERSION_DATE}")
        plt.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(quality_path, dpi=150)
        plt.close(figure)
    else:
        _plot_no_data("Quality issues theo dataset", quality_path)
    created.append(quality_path)

    split_path = figures / "23_split_sequence_distribution.png"
    if split_distribution:
        datasets = sorted({str(row["dataset_name"]) for row in split_distribution})
        split_names = ("EXTERNAL_TRAIN", "EXTERNAL_VALIDATION", "CROSS_DATASET_TEST")
        lookup = {
            (str(row["dataset_name"]), str(row["proposed_split"])): int(row["sequence_or_group_count"])
            for row in split_distribution
        }
        positions = np.arange(len(datasets))
        bottom = np.zeros(len(datasets))
        figure = plt.figure(figsize=(9, 5.5))
        for split in split_names:
            values = np.asarray([lookup.get((dataset, split), 0) for dataset in datasets])
            plt.bar(positions, values, bottom=bottom, label=split)
            bottom += values
        plt.xticks(positions, [name.replace(" Localization", "") for name in datasets], rotation=15, ha="right")
        plt.ylabel("Số sequence/nhóm")
        plt.title(f"Proposal split theo đơn vị chống leakage\nDataset version: {VERSION_DATE}")
        plt.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(split_path, dpi=150)
        plt.close(figure)
    else:
        _plot_no_data("Proposal split theo đơn vị chống leakage", split_path)
    created.append(split_path)
    return created


FIGURE_CSV_SOURCES = {
    "01_images_by_dataset.png": ["dataset_inventory.csv"],
    "02_videos_sequences_by_dataset.png": ["dataset_inventory.csv"],
    "03_bboxes_by_dataset.png": ["dataset_inventory.csv"],
    "04_original_class_distribution.png": ["class_distribution.csv"],
    "05_mapped_class_distribution.png": ["class_distribution.csv"],
    "06_lighting_distribution.png": ["condition_distribution.csv"],
    "07_weather_distribution.png": ["condition_distribution.csv"],
    "08_brightness_distribution.png": ["image_quality_samples.csv"],
    "09_blur_distribution.png": ["image_quality_samples.csv"],
    "10_resolution_distribution.png": ["image_quality_samples.csv"],
    "11_bbox_area_ratio.png": ["bbox_samples.csv"],
    "12_bbox_width_320.png": ["bbox_samples.csv"],
    "13_bbox_height_320.png": ["bbox_samples.csv"],
    "14_bbox_320_categories.png": ["bbox_samples.csv"],
    "15_boxes_per_image.png": ["boxes_per_image_distribution.csv"],
    "16_ua_occlusion.png": ["ua_annotation_attribute_summary.csv"],
    "17_ua_truncation.png": ["ua_annotation_attribute_summary.csv"],
    "18_viewpoint_score.png": ["viewpoint_suitability.csv"],
    "19_selection_ratio.png": ["balanced_subset_plan.csv"],
    "20_data_gaps.png": ["dataset_gap_analysis.csv"],
    "21_cross_test_bbox_difficulty.png": ["cross_test_sequence_statistics.csv"],
    "22_quality_issue_counts.png": ["quality_audit_summary.csv"],
    "23_split_sequence_distribution.png": ["split_distribution.csv"],
}


def _write_figure_provenance(output: Path, figures: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for figure in figures:
        source_names = FIGURE_CSV_SOURCES.get(figure.name, [])
        source_paths = [output / name for name in source_names]
        available = bool(source_paths) and all(path.is_file() for path in source_paths)
        rows.append(
            {
                "figure_path": figure.relative_to(output).as_posix(),
                "figure_sha256": _sha256_file(figure) if figure.is_file() else "",
                "source_csvs": ";".join(source_names),
                "source_sha256s": ";".join(
                    f"{path.name}:{_sha256_file(path)}" for path in source_paths if path.is_file()
                ),
                "generator": "run_external_dataset_eda.py",
                "status": "VERIFIED_CSV_SOURCE" if available else "MISSING_SOURCE_DECLARATION",
            }
        )
    write_csv(
        output / "figure_provenance.csv",
        rows,
        [
            "figure_path",
            "figure_sha256",
            "source_csvs",
            "source_sha256s",
            "generator",
            "status",
        ],
    )
    return rows


def _write_reports(
    output: Path,
    inventory: list[dict[str, Any]],
    bbox_stats: list[dict[str, Any]],
    viewpoints: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    leakage: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    figures: list[Path],
    results: list[dict[str, Any]],
    road_type_distribution: list[dict[str, Any]],
    scene_distribution: list[dict[str, Any]],
    quality_audit: list[dict[str, Any]],
    split_validation: list[dict[str, Any]],
) -> None:
    total_images_checked = sum(len(result.get("quality_rows", [])) for result in results)
    total_annotations = sum(int(result.get("annotation_row_count", 0)) for result in results)
    total_boxes_checked = sum(int(result.get("bbox_analyzed_count", 0)) for result in results)
    bbox_scope_summary = "; ".join(
        f"{row['dataset_name']}={row.get('analysis_scope') or 'NOT_DECLARED'}"
        for row in inventory
        if row.get("status") == "ANALYZED"
    ) or "NOT_AVAILABLE"
    total_invalid = sum(
        len(
            {
                (str(row.get("source_file", "")), str(row.get("annotation_id", "")))
                for row in result.get("invalid_annotations", [])
            }
        )
        for result in results
    )
    total_issues = sum(len(result.get("invalid_annotations", [])) for result in results)
    total_boundary_clipped = sum(
        int(result.get("boundary_clipped_bbox_count", 0)) for result in results
    )
    aau_lighting_counts = Counter(
        str(row.get("value", "UNKNOWN"))
        for result in results
        if result.get("dataset_name") == "AAU RainSnow"
        for row in result.get("conditions", [])
        if row.get("condition") == "lighting"
        for _ in range(int(row.get("count", 0)))
    )
    aau_lighting_summary = ", ".join(
        f"{label}={count}" for label, count in sorted(aau_lighting_counts.items())
    ) or "NOT_AVAILABLE"
    duplicate_groups = len({row["duplicate_group_id"] for row in duplicates})
    critical_leakage = sum(row.get("severity") == "CRITICAL" for row in leakage)
    most_relevant = max(
        (
            (name, sum(float(row["overall_score"]) for row in viewpoints if row["dataset_name"] == name) /
             sum(1 for row in viewpoints if row["dataset_name"] == name))
            for name in {row["dataset_name"] for row in viewpoints if row["dataset_name"] != "RADIATE"}
        ),
        key=lambda item: item[1],
    )
    tiny_rates = {
        row["dataset_name"]: row.get("difficult_under_8px_ratio", "") for row in bbox_stats
    }
    cross_road_types: Counter[str] = Counter()
    for row in road_type_distribution:
        if row.get("proposed_split") == "CROSS_DATASET_TEST":
            cross_road_types[str(row.get("road_type", "UNKNOWN"))] += int(row.get("sequence_count", 0))
    cross_road_summary = ", ".join(
        f"{road_type}={count}" for road_type, count in sorted(cross_road_types.items())
    ) or "CHƯA CÓ"
    cross_scene_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in scene_distribution:
        if row.get("proposed_split") == "CROSS_DATASET_TEST":
            cross_scene_counts[str(row.get("dimension", "UNKNOWN"))][
                str(row.get("value", "UNKNOWN"))
            ] += int(row.get("sequence_count", 0))
    cross_scene_summary = "; ".join(
        f"{dimension}: "
        + ", ".join(f"{value}={count}" for value, count in sorted(values.items()))
        for dimension, values in sorted(cross_scene_counts.items())
    ) or "CHƯA CÓ"
    quality_gate_summary = ", ".join(
        f"{status}={count}"
        for status, count in sorted(Counter(str(row.get("quality_gate")) for row in quality_audit).items())
    ) or "CHƯA CÓ"
    split_check_summary = ", ".join(
        f"{status}={count}"
        for status, count in sorted(Counter(str(row.get("status")) for row in split_validation).items())
    ) or "CHƯA CÓ"
    class_mapping_corrections = sum(
        int(row.get("class_mapping_corrections_in_bbox_sample", 0)) for row in quality_audit
    )
    others_review_rows = read_csv(
        ROOT / "reports" / "external_eda" / "ua_others_stratified_review_queue.csv"
    )
    others_review_counts = Counter(
        str(row.get("visual_assessment", "PENDING_REVIEW")) for row in others_review_rows
    )
    others_review_summary = ", ".join(
        f"{label}={count}" for label, count in sorted(others_review_counts.items())
    ) or "NOT_AVAILABLE"
    executive = f"""# Executive summary — External Dataset EDA

- Ngày chạy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Dataset: MIO-TCD Localization, AAU RainSnow, UA-DETRAC Original.
- AAU lighting manual review (22 sequence): **{aau_lighting_summary}**.
- UA-DETRAC boundary-crossing bbox clipped and kept: **{total_boundary_clipped:,}**.
- RADIATE: `EXCLUDED_VIEWPOINT_MISMATCH`, không chạy EDA.
- Ảnh/frame kiểm tra chất lượng thật: **{total_images_checked:,}**.
- Annotation rows đọc: **{total_annotations:,}**.
- Tổng bounding box trong phạm vi EDA: **{total_boxes_checked:,}** (không phải full-raw total).
- Phạm vi bbox theo dataset: **{bbox_scope_summary}**.
- Annotation lỗi duy nhất: **{total_invalid:,}**; tổng issue: **{total_issues:,}**.
- Nhóm trùng/nghi gần trùng trên mẫu: **{duplicate_groups:,}**.
- Leakage mức CRITICAL: **{critical_leakage:,}**.
- Road type trong cross test proposal: **{cross_road_summary}**; `EMERGENCY_LANE_LIKE=0` nếu không xuất hiện.
- Điều kiện cross test đã review: **{cross_scene_summary}**.
- Quality gate: **{quality_gate_summary}**; bbox sample được sửa theo class mapping: **{class_mapping_corrections:,}**.
- UA `others` Data Lead review: **{others_review_summary}**; approved with mandatory exclusion of `MVI_40172 / track 79` (201 boxes).
- Kiểm tra split: **{split_check_summary}**; MIO không có sequence được giữ train-only.
- Điểm viewpoint trung bình cao nhất: **{most_relevant[0]} ({most_relevant[1]:.2f}/5, AUTOMATIC_ESTIMATE)**.

Không dataset nào có ground-truth “xe dừng trong làn khẩn cấp”. UA-DETRAC chỉ tạo `STATIONARY_CANDIDATE` từ track để con người review; không dùng làm nhãn train. Main test bắt buộc ưu tiên K230 tự quay.
"""
    (output / "executive_summary.md").write_text(executive, encoding="utf-8")
    findings = f"""# Research findings — EDA ba dataset bên ngoài

## 1. Mục tiêu EDA

Đánh giá dữ liệu bổ sung cho YOLOv8n 320×320 và camera K230 cố định trên cao, không thay thế test thực địa K230.

## 2. Mô tả dataset

{chr(10).join(f"- **{row['dataset_name']}**: {int(row['image_count']):,} ảnh/modality image, {int(row['video_count']):,} video, {int(row['sequence_count']):,} sequence, {int(row['bbox_count']):,} bbox reported." for row in inventory)}

## 3. Lý do không sử dụng RADIATE

`EXCLUDED_VIEWPOINT_MISMATCH`: camera phía trước phương tiện khác camera cố định trên cao. Dữ liệu không bị xóa.

## 4. Quy trình

Inventory archive, parse annotation gốc, validation bbox, image-quality sample streaming, letterbox 320, duplicate sample, leakage theo sequence, viewpoint estimate và subset proposal. Không copy/xóa dữ liệu gốc.

## 5. MIO-TCD Localization

Chỉ đọc TAR Localization; Classification bị chặn. MIO cung cấp localization ảnh tĩnh, không có Track ID và không chứng minh trạng thái dừng.

## 6. AAU RainSnow

Nhánh `aaurainsnow/` lặp được bỏ khỏi thống kê. Dataset có video RGB/thermal và COCO instance annotation; mưa/tuyết cụ thể theo sequence cần review vì metadata hiện có không tách rõ.

## 7. UA-DETRAC

Đọc toàn bộ XML train/test, Track ID, weather, camera state, occlusion/truncation. Stationary candidate là heuristic và luôn `manual_review_status=PENDING`.

## 8–10. Góc camera, điều kiện và class

Viewpoint cao nhất theo rule hiện tại: **{most_relevant[0]} {most_relevant[1]:.2f}/5**. AAU bổ sung adverse weather; UA hỗ trợ tracking; MIO/AAU có lớp xe máy, còn UA-DETRAC không quan sát thấy xe máy trong class XML đã đọc.

Cross-dataset test proposal theo road type: **{cross_road_summary}**. Chưa có sequence được xác nhận là `EMERGENCY_LANE_LIKE`; metadata cảnh vẫn chờ Data Lead duyệt.

Metadata cảnh cross-test: **{cross_scene_summary}**. `weather=UNKNOWN` được giữ lại khi XML chỉ ghi `night`, vì `night` là điều kiện ánh sáng chứ không phải thời tiết. Mật độ xe được tính bằng số bbox phương tiện trung bình trên mỗi ảnh có annotation.

Phân bố chi tiết theo class và kích thước bbox được xuất ở `class_distribution_by_scene.csv` và `bbox_distribution_by_scene.csv`; số đếm cấp sequence nằm trong `cross_test_sequence_statistics.csv`. Các tỷ lệ theo scene dùng đúng phạm vi bbox analysis sample và không được trình bày như toàn bộ dataset.

## 11–12. Bounding box và resize 320×320

Tỷ lệ box dưới 8 px theo dataset: {", ".join(f"{name}={rate}" for name, rate in tiny_rates.items())}. Đây là ngưỡng phân tích ban đầu, không phải ngưỡng ground truth.

## 13–15. Chất lượng ảnh, annotation, trùng và leakage

Đã kiểm tra {total_images_checked:,} ảnh/frame mẫu; ghi {total_invalid:,} annotation lỗi duy nhất ({total_issues:,} issue). Duplicate scan chỉ áp dụng trên mẫu đã đọc ảnh. Phát hiện {critical_leakage:,} leakage CRITICAL theo sequence metadata.

Quality gate hiện tại: **{quality_gate_summary}**. Pipeline đã sửa **{class_mapping_corrections:,}** bbox sample theo `vehicle_class_mapping.yaml`. Class policy đã chốt; Data Lead đã hoàn tất review `UA-DETRAC:others`. Hàng đợi hành động nằm tại `quality_review_queue.csv`.

UA `others` Data Lead review: **{others_review_summary}**. Review đã phủ đủ 74 unique track: giữ 73 track xe và loại toàn bộ 201 box của `MVI_40172 / track 79` theo `ua_others_track_exclusions.csv`.

## 16–17. Vehicle detection và giới hạn xe dừng

Ba bộ hỗ trợ nhận diện phương tiện. Không bộ nào có ground-truth xe dừng trong ROI. Không được kết luận xe dừng từ một ảnh.

## 18. Khoảng trống so với K230

Thiếu ROI làn khẩn cấp, thời gian dừng 2–3 giây/>5 giây, xe rời ROI, ngược sáng/đèn pha đã xác minh và domain camera K230 tại trường.

## 19–20. Subset

PILOT_500 và DATASET_V1_1500 được chia gần cân bằng giữa ba nguồn, theo sequence và chỉ là proposal. Xem `balanced_subset_plan.csv` và `selected_data_manifest.csv`.

## 21. Đề xuất thu K230

Quay theo sequence độc lập: ngày/đêm/mưa/đường ướt/ngược sáng/đèn pha; có xe chạy, chậm, dừng, rời ROI và negative. Main test khóa theo session/video.

## 22. Kết luận

Dùng dữ liệu ngoài cho train, validation sequence-level và cross-domain test. Main project test không được chỉ dùng ba dataset ngoài.

Split validation: **{split_check_summary}**. MIO-TCD không có sequence/session tin cậy nên chỉ được đề xuất vào `EXTERNAL_TRAIN`; test chính `MAIN_K230_TEST` vẫn là placeholder chờ thu và khóa manifest.

## 23. Nguồn

- MIO-TCD: https://tcd.miovision.com/
- AAU RainSnow: https://www.kaggle.com/datasets/aalborguniversity/aau-rainsnow
- UA-DETRAC original dataset name; Kaggle download mirror: https://www.kaggle.com/datasets/bratjay/ua-detrac-orig
"""
    (output / "research_findings.md").write_text(findings, encoding="utf-8")
    daily = f"""[SV1 – {datetime.now().strftime('%d/%m/%Y')}]

1. Hôm nay làm:
Thực hiện EDA cho MIO-TCD Localization, AAU RainSnow và UA-DETRAC nhằm đánh giá mức độ phù hợp với hệ thống camera K230 đặt cố định trên cao.

2. Kết quả/bằng chứng:
- Dataset đã tìm thấy: {", ".join(row["dataset_name"] for row in inventory if row["status"] == "ANALYZED")}
- Dataset chưa tìm thấy: {", ".join(row["dataset_name"] for row in inventory if row["status"] != "ANALYZED") or "KHÔNG CÓ"}
- Số ảnh đã kiểm tra: {total_images_checked:,}
- Số video/sequence đã kiểm tra: {sum(int(row["video_count"]) + int(row["sequence_count"]) for row in inventory):,}
- Số annotation đã đọc: {total_annotations:,}
- Tổng bounding box trong phạm vi EDA: {total_boxes_checked:,} (không phải full-raw total)
- Phạm vi bbox theo dataset: {bbox_scope_summary}
- Số annotation lỗi duy nhất: {total_invalid:,}
- Tổng số issue annotation: {total_issues:,}
- Số nhóm ảnh nghi ngờ trùng: {duplicate_groups:,}
- Tỷ lệ box dưới 8 px sau resize 320×320: {tiny_rates}
- Dataset phù hợp nhất về góc camera: {most_relevant[0]} ({most_relevant[1]:.2f}/5, cần review)
- Road type trong cross test proposal: {cross_road_summary}; chưa có `EMERGENCY_LANE_LIKE`.
- Điều kiện cross test đã review: {cross_scene_summary}.
- Quality gate: {quality_gate_summary}; class mapping corrections trong bbox sample: {class_mapping_corrections:,}.
- Split validation: {split_check_summary}; MIO train-only, K230 main test đang chờ thu.
- Điều kiện dữ liệu được bổ sung: mưa/tuyết, camera cố định, tracking sequence.
- Link báo cáo: reports/external_eda/research_findings.md
- Link biểu đồ: reports/external_eda/figures/
- Link commit/PR: commit hiện tại {git_commit()}

3. Vướng mắc/cần hỗ trợ:
- Class mapping đã chốt: CÓ; `UA-DETRAC:others` đã review đủ 74 track, giữ 73 track xe, loại 201 box của 1 track non-vehicle và luôn giữ original class.
- Dữ liệu chưa có nhãn detection: KHÔNG; AAU có COCO instance annotation, nhưng điều kiện theo sequence cần review.
- Dataset quá lớn: CÓ; image quality chạy theo sample/streaming.
- Thiếu dung lượng: KHÔNG XÁC NHẬN LÀ VƯỚNG MẮC.
- Thiếu dữ liệu K230 thực tế: CÓ.
- Cần giảng viên xác nhận: class xe máy/xe đạp, subset, vị trí K230 và protocol main test.

4. Ngày mai:
- Review các ảnh lỗi.
- Khi export nhãn train, áp dụng danh sách loại toàn bộ `MVI_40172 / track 79` gồm 201 box.
- Chốt subset cân bằng.
- Chuẩn bị dữ liệu gán nhãn còn thiếu.
- Tiếp tục khảo sát dữ liệu K230 thực tế.
"""
    (output / "daily_report_draft.md").write_text(daily, encoding="utf-8")


def _report_only(output: Path) -> int:
    inventory = read_csv(output / "dataset_inventory.csv")
    if not inventory:
        print("Không có kết quả EDA để tạo lại báo cáo.", file=sys.stderr)
        return 2
    lines = ["# Executive summary — report-only", "", f"- Dataset rows: {len(inventory)}"]
    for row in inventory:
        lines.append(
            f"- {row['dataset_name']}: images={row['image_count']}, bbox={row['bbox_count']}, scope={row['analysis_scope']}"
        )
    lines.append("\nSố liệu được đọc lại từ CSV kết quả; không chạy lại dataset.")
    (output / "executive_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output.resolve())
    return 0


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        return _report_only(output)
    mapping = load_yaml(ROOT / "configs" / "vehicle_class_mapping.yaml")
    road_type_config = load_yaml(ROOT / "configs" / "sequence_road_types.yaml")
    split_policy = load_yaml(ROOT / "configs" / "split_policy.yaml")
    paths = discover(ROOT, args.mio_path, args.aau_path, args.uadetrac_path)
    print("Dataset paths:")
    for key, value in paths.items():
        print(f"- {key}: {value or 'NOT_FOUND'}")

    inspectors = {
        "mio_tcd": ("MIO-TCD Localization", inspect_mio),
        "aau_rainsnow": ("AAU RainSnow", inspect_aau),
        "ua_detrac": ("UA-DETRAC Original", inspect_ua_detrac),
    }
    results: list[dict[str, Any]] = []
    serious_errors: list[str] = []
    cache_dir = output / "cache"
    log_lines: list[str] = []
    for key, (dataset_name, inspector) in inspectors.items():
        path = paths.get(key)
        if path is None or not path.exists():
            results.append(
                {
                    "dataset_name": dataset_name,
                    "path": str(path or ""),
                    "status": "NOT_FOUND",
                    "data_type": "",
                    "image_count": 0,
                    "video_count": 0,
                    "sequence_count": 0,
                    "annotation_status": "NOT_FOUND",
                    "annotation_file_count": 0,
                    "annotation_row_count": 0,
                    "bbox_count": 0,
                    "bbox_analyzed_count": 0,
                    "track_count": 0,
                    "class_counts": {},
                    "invalid_annotations": [],
                    "quality_rows": [],
                    "bbox_samples": [],
                    "image_records": [],
                    "sequences": [],
                    "conditions": [],
                    "files_processed_successfully": 0,
                    "files_failed": 0,
                    "elapsed_seconds": 0,
                    "analysis_scope": "NOT_FOUND",
                }
            )
            continue
        cache_path = cache_dir / f"{key}_{'full' if args.full_scan else f'sample_{args.sample_size}'}.json"
        cache_identity = _cache_identity(key, path, inspector, args)
        cached = _load_cache(cache_path, cache_identity) if args.resume else None
        if cached:
            print(f"Resume: dùng cache {cache_path}")
            results.append(cached)
            continue
        if args.resume and cache_path.exists():
            print(f"Resume: stale cache rejected (fingerprint mismatch) {cache_path}")
        try:
            result = inspector(
                path,
                sample_size=args.sample_size,
                full_scan=args.full_scan,
                skip_images=args.skip_images,
                progress=True,
            )
            results.append(result)
            _save_cache(cache_path, result, cache_identity)
        except Exception as error:  # tiếp tục bước độc lập theo prompt
            serious_errors.append(f"{dataset_name}: {error}")
            log_lines.append(traceback.format_exc())
            results.append(
                {
                    "dataset_name": dataset_name,
                    "path": str(path),
                    "status": "FAILED",
                    "data_type": "",
                    "image_count": 0,
                    "video_count": 0,
                    "sequence_count": 0,
                    "annotation_status": "FAILED",
                    "annotation_file_count": 0,
                    "annotation_row_count": 0,
                    "bbox_count": 0,
                    "bbox_analyzed_count": 0,
                    "track_count": 0,
                    "class_counts": {},
                    "invalid_annotations": [],
                    "quality_rows": [],
                    "bbox_samples": [],
                    "image_records": [],
                    "sequences": [],
                    "conditions": [],
                    "files_processed_successfully": 0,
                    "files_failed": 1,
                    "elapsed_seconds": 0,
                    "analysis_scope": "FAILED",
                }
            )
    if log_lines:
        (output / "pipeline_errors.log").write_text("\n".join(log_lines), encoding="utf-8")
    analyzed = [result for result in results if result["status"] == "ANALYZED"]
    _normalize_bbox_class_mapping(analyzed, mapping)
    inventory = _inventory(results)
    class_rows = _class_rows(analyzed, mapping)
    annotation_quality = _annotation_quality(results)
    quality_summary, quality_samples = _image_quality(analyzed)
    bbox_stats, bbox_samples = _bbox_statistics(analyzed)
    conditions = [row for result in analyzed for row in result.get("conditions", [])]
    viewpoints = assess_viewpoints(analyzed)
    plans, manifest, splits = create_plan(analyzed, split_policy, road_type_config)
    split_by_sequence: dict[tuple[str, str], str] = {}
    for row in splits:
        dataset_name = str(row["dataset_name"])
        proposed = str(row["proposed_split"])
        split_by_sequence[(dataset_name, str(row["sequence_id"]))] = proposed
        split_by_sequence[(dataset_name, str(row.get("source_sequence_id", "")))] = proposed
    duplicates = (
        []
        if args.skip_duplicates
        else detect_duplicates(analyzed, split_by_sequence)
    )
    leakage = detect_leakage(analyzed)
    road_type_distribution, scene_distribution = _apply_scene_metadata(
        splits, manifest, road_type_config
    )
    scene_assessments = _scene_assessment_rows(road_type_config)
    cross_sequence_stats, class_by_scene, bbox_by_scene = build_scene_slice_analysis(
        analyzed, road_type_config
    )
    quality_audit, label_consistency, quality_review_queue = build_quality_audit(
        analyzed, duplicates, mapping
    )
    split_validation, split_distribution, k230_holdout = build_split_audit(
        splits, manifest, split_policy
    )
    assert_sequence_split(splits)
    gaps = _gap_rows()
    comparison = _comparison(analyzed, bbox_stats, annotation_quality)
    invalid = [row for result in analyzed for row in result.get("invalid_annotations", [])]
    boundary_clip_samples = [
        row for result in analyzed for row in result.get("boundary_clip_samples", [])
    ]
    stationary = [row for result in analyzed for row in result.get("stationary_candidates", [])]
    boxes_per_image_distribution = _boxes_per_image_rows(analyzed)
    ua_annotation_attributes = _ua_annotation_attribute_rows(analyzed)

    write_csv(output / "dataset_inventory.csv", inventory, INVENTORY_FIELDS)
    write_csv(output / "dataset_comparison.csv", comparison, list(comparison[0]) if comparison else ["dataset_name"])
    write_csv(output / "annotation_quality.csv", annotation_quality, list(annotation_quality[0]) if annotation_quality else ["dataset_name"])
    write_csv(output / "image_quality.csv", quality_summary, list(quality_summary[0]) if quality_summary else ["dataset_name"])
    write_csv(output / "image_quality_samples.csv", quality_samples, QUALITY_SAMPLE_FIELDS)
    write_csv(output / "bbox_statistics.csv", bbox_stats, list(bbox_stats[0]) if bbox_stats else ["dataset_name"])
    write_csv(output / "bbox_samples.csv", bbox_samples, BBOX_SAMPLE_FIELDS)
    write_csv(output / "class_distribution.csv", class_rows, list(class_rows[0]) if class_rows else ["dataset_name"])
    write_csv(output / "condition_distribution.csv", conditions, ["dataset_name", "condition", "value", "count", "unit", "assessment_source"])
    write_csv(output / "viewpoint_suitability.csv", viewpoints, VIEWPOINT_FIELDS)
    write_csv(output / "duplicate_groups.csv", duplicates, DUPLICATE_FIELDS)
    write_csv(output / "sequence_leakage.csv", leakage, LEAKAGE_FIELDS)
    write_csv(output / "invalid_annotations.csv", invalid, INVALID_FIELDS)
    write_csv(
        output / "bbox_boundary_clip_samples.csv",
        boundary_clip_samples,
        [
            "dataset_name", "sequence_name", "source_file", "annotation_id",
            "original_class", "raw_bbox", "clipped_bbox_xyxy", "adjustments", "action",
        ],
    )
    write_csv(output / "dataset_gap_analysis.csv", gaps, list(gaps[0]))
    write_csv(output / "balanced_subset_plan.csv", plans, list(plans[0]) if plans else ["scenario"])
    write_csv(output / "selected_data_manifest.csv", manifest, MANIFEST_FIELDS)
    write_csv(output / "split_proposal.csv", splits, list(splits[0]) if splits else ["dataset_name"])
    write_csv(
        output / "road_type_distribution.csv",
        road_type_distribution,
        ["proposed_split", "dataset_name", "road_type", "sequence_count", "assessment_scope"],
    )
    write_csv(
        output / "scene_metadata_distribution.csv",
        scene_distribution,
        [
            "proposed_split", "dataset_name", "dimension", "value",
            "sequence_count", "assessment_scope",
        ],
    )
    write_csv(
        output / "sequence_scene_metadata.csv",
        scene_assessments,
        [
            "dataset_name", "sequence_id", "road_type", "weather", "lighting",
            "camera_view", "traffic_density", "mean_vehicles_per_image",
            "weather_source", "lighting_source", "camera_view_source",
            "traffic_density_source", "manual_review_status", "evidence",
        ],
    )
    write_csv(
        output / "cross_test_sequence_statistics.csv",
        cross_sequence_stats,
        list(cross_sequence_stats[0]) if cross_sequence_stats else ["dataset_name"],
    )
    write_csv(
        output / "class_distribution_by_scene.csv",
        class_by_scene,
        list(class_by_scene[0]) if class_by_scene else ["dimension"],
    )
    write_csv(
        output / "bbox_distribution_by_scene.csv",
        bbox_by_scene,
        list(bbox_by_scene[0]) if bbox_by_scene else ["dimension"],
    )
    write_csv(
        output / "quality_audit_summary.csv",
        quality_audit,
        list(quality_audit[0]) if quality_audit else ["dataset_name"],
    )
    write_csv(
        output / "label_consistency_audit.csv",
        label_consistency,
        list(label_consistency[0]) if label_consistency else ["dataset_name"],
    )
    write_csv(
        output / "quality_review_queue.csv",
        quality_review_queue,
        list(quality_review_queue[0]) if quality_review_queue else ["queue_id"],
    )
    write_csv(
        output / "split_validation_summary.csv",
        split_validation,
        list(split_validation[0]) if split_validation else ["check_id"],
    )
    write_csv(
        output / "split_distribution.csv",
        split_distribution,
        list(split_distribution[0]) if split_distribution else ["dataset_name"],
    )
    write_csv(
        output / "k230_holdout_plan.csv",
        k230_holdout,
        list(k230_holdout[0]) if k230_holdout else ["slice_id"],
    )
    write_csv(
        output / "stationary_candidates.csv",
        stationary,
        [
            "dataset_name", "sequence_name", "track_id", "original_class", "first_frame",
            "last_frame", "track_frame_count", "normalized_center_extent",
            "stationary_candidate", "confidence", "status", "ground_truth",
            "manual_review_status",
        ],
    )
    write_csv(
        output / "boxes_per_image_distribution.csv",
        boxes_per_image_distribution,
        ["dataset_name", "boxes_per_image", "image_count", "analysis_scope"],
    )
    write_csv(
        output / "ua_annotation_attribute_summary.csv",
        ua_annotation_attributes,
        ["metric", "count", "analysis_scope"],
    )
    figures = _figures(
        output,
        read_csv(output / "dataset_inventory.csv"),
        read_csv(output / "class_distribution.csv"),
        read_csv(output / "image_quality_samples.csv"),
        read_csv(output / "bbox_samples.csv"),
        read_csv(output / "bbox_statistics.csv"),
        read_csv(output / "condition_distribution.csv"),
        read_csv(output / "viewpoint_suitability.csv"),
        read_csv(output / "balanced_subset_plan.csv"),
        read_csv(output / "dataset_gap_analysis.csv"),
        read_csv(output / "boxes_per_image_distribution.csv"),
        read_csv(output / "ua_annotation_attribute_summary.csv"),
    )
    figures.extend(
        _audit_figures(
            output,
            read_csv(output / "cross_test_sequence_statistics.csv"),
            read_csv(output / "quality_audit_summary.csv"),
            read_csv(output / "split_distribution.csv"),
        )
    )
    _write_figure_provenance(output, figures)
    contact_paths: list[str] = []
    contact_root = ROOT / "storage_placeholders" / "online_data" / "contact_sheets" / "external_eda"
    if not args.skip_contact_sheets:
        for result in analyzed:
            contact = create_contact_sheet(result, contact_root)
            if contact:
                contact_paths.append(str(contact))
    else:
        contact_paths = [str(path) for path in sorted(contact_root.glob("*.jpg"))] if contact_root.exists() else []
    contact_status = [
        {
            "dataset_name": result["dataset_name"],
            "contact_sheet_type": "DATASET_REPRESENTATIVE",
            "status": "GENERATED_PIXELLATED_OUTSIDE_GIT"
            if any(result["dataset_name"].lower().replace(" ", "_").replace("-", "_") in Path(path).stem for path in contact_paths)
            else "NOT_GENERATED",
            "output_location": next(
                (
                    _portable_report_path(path)
                    for path in contact_paths
                    if result["dataset_name"].lower().replace(" ", "_").replace("-", "_") in Path(path).stem
                ),
                "",
            ),
            "privacy_status": "WHOLE_IMAGE_PIXELATED",
            "notes": "Không commit contact sheet; chỉ dùng review cục bộ.",
        }
        for result in analyzed
    ]
    contact_status.append(
        {
            "dataset_name": "ALL",
            "contact_sheet_type": "CONDITION_AND_ERROR_SPECIFIC",
            "status": "PENDING_MANUAL_PRIVACY_AND_CONDITION_REVIEW",
            "output_location": "",
            "privacy_status": "NOT_EXPORTED",
            "notes": "Không tự gán weather/day/night hoặc lỗi nhãn khi bằng chứng chưa đủ.",
        }
    )
    write_csv(
        output / "contact_sheet_status.csv",
        contact_status,
        ["dataset_name", "contact_sheet_type", "status", "output_location", "privacy_status", "notes"],
    )
    _write_reports(
        output, inventory, bbox_stats, viewpoints, duplicates, leakage, plans, figures,
        analyzed, road_type_distribution, scene_distribution, quality_audit, split_validation,
    )

    print("\nEDA output:")
    print(output)
    print(f"Analyzed datasets: {len(analyzed)}/3")
    print(f"Files/images checked: {sum(int(row['files_processed_successfully']) for row in inventory)}")
    print(f"Analysis-scope bounding boxes: {sum(int(row['bbox_analyzed_count']) for row in inventory)}")
    unique_invalid = {
        (str(row.get("dataset_name", "")), str(row.get("source_file", "")), str(row.get("annotation_id", "")))
        for row in invalid
    }
    print(f"Invalid annotations (unique): {len(unique_invalid)}; issues: {len(invalid)}")
    print(f"Duplicate groups: {len({row['duplicate_group_id'] for row in duplicates})}")
    print(f"Critical leakage: {sum(row.get('severity') == 'CRITICAL' for row in leakage)}")
    print(f"Contact sheets outside Git: {contact_paths or 'SKIPPED_OR_NOT_AVAILABLE'}")
    if serious_errors:
        print("Serious errors:", *serious_errors, sep="\n- ", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mio-path")
    parser.add_argument("--aau-path")
    parser.add_argument("--uadetrac-path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--full-scan", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-duplicates", action="store_true")
    parser.add_argument("--skip-contact-sheets", action="store_true")
    parser.add_argument("--apply-selection", action="store_true")
    parser.add_argument("--apply-split", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
