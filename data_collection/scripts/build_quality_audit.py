"""Tổng hợp quality gate, bất nhất nhãn và hàng đợi review từ EDA thật."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

DATASET_MAPPING_KEYS = {
    "MIO-TCD Localization": "mio_tcd",
    "AAU RainSnow": "aau_rainsnow",
    "UA-DETRAC Original": "ua_detrac",
}


def build_quality_audit(
    results: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    class_mapping: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    duplicate_groups: dict[str, defaultdict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    duplicate_items: dict[str, Counter[str]] = defaultdict(Counter)
    for row in duplicates:
        dataset = str(row.get("dataset_name", "UNKNOWN"))
        duplicate_type = str(row.get("duplicate_type", "UNKNOWN"))
        duplicate_groups[dataset][duplicate_type].add(str(row.get("duplicate_group_id", "")))
        duplicate_items[dataset][duplicate_type] += 1

    summaries: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    queue_index = 1

    def add_queue(
        dataset: str,
        category: str,
        count: int,
        severity: str,
        scope: str,
        action: str,
    ) -> None:
        nonlocal queue_index
        if count <= 0:
            return
        queue.append(
            {
                "queue_id": f"QREV_{queue_index:04d}",
                "dataset_name": dataset,
                "issue_category": category,
                "issue_count": count,
                "severity": severity,
                "scope": scope,
                "recommended_action": action,
                "manual_review_status": "PENDING",
            }
        )
        queue_index += 1

    for result in results:
        dataset = str(result["dataset_name"])
        quality = result.get("quality_rows", [])
        valid_quality = [row for row in quality if row.get("read_status") == "OK"]
        failed = len(quality) - len(valid_quality)
        underexposed = sum(bool(row.get("underexposed_suspect")) for row in valid_quality)
        overexposed = sum(bool(row.get("overexposed_suspect")) for row in valid_quality)
        blurred = sum(bool(row.get("blur_suspect")) for row in valid_quality)
        invalid_rows = result.get("invalid_annotations", [])
        boundary_clipped = int(result.get("boundary_clipped_bbox_count", 0))
        invalid_unique = len(
            {
                (str(row.get("source_file", "")), str(row.get("annotation_id", "")))
                for row in invalid_rows
            }
        )
        bbox_rows = result.get("bbox_samples", [])
        corrected_mapping = sum(
            row.get("class_mapping_status") == "CORRECTED_TO_CONFIG" for row in bbox_rows
        )
        unmapped_bbox = sum(row.get("class_mapping_status") == "UNMAPPED_CLASS" for row in bbox_rows)
        exact_groups = len(duplicate_groups[dataset]["EXACT_SHA256"])
        near_groups = len(duplicate_groups[dataset]["NEAR_DUPLICATE_PHASH_CONSECUTIVE"])
        basename_groups = len(duplicate_groups[dataset]["DUPLICATE_BASENAME"])
        gate = "BLOCKED" if failed else "REVIEW_REQUIRED" if any(
            (underexposed, overexposed, blurred, invalid_unique, exact_groups, near_groups,
             corrected_mapping, unmapped_bbox)
        ) else "PASS"
        summaries.append(
            {
                "dataset_name": dataset,
                "quality_images_checked": len(quality),
                "corrupt_or_unreadable": failed,
                "underexposed_suspects": underexposed,
                "overexposed_suspects": overexposed,
                "blur_suspects": blurred,
                "invalid_annotations_unique": invalid_unique,
                "invalid_annotation_issues": len(invalid_rows),
                "boundary_clipped_bbox_count": boundary_clipped,
                "exact_duplicate_groups": exact_groups,
                "near_duplicate_groups": near_groups,
                "duplicate_basename_groups": basename_groups,
                "class_mapping_corrections_in_bbox_sample": corrected_mapping,
                "unmapped_bbox_sample_count": unmapped_bbox,
                "quality_gate": gate,
                "assessment_scope": "EDA_SCAN_AND_ANALYSIS_SAMPLE",
                "notes": "Duplicate basename chỉ là tín hiệu cần kiểm tra, không phải bằng chứng nội dung trùng.",
            }
        )

        rules = class_mapping.get(DATASET_MAPPING_KEYS[dataset], {})
        sample_by_class: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in bbox_rows:
            sample_by_class[str(row.get("original_class", "UNKNOWN"))].append(row)
        for original_class, reported_count in sorted(result.get("class_counts", {}).items()):
            rule = rules.get(original_class)
            samples = sample_by_class.get(original_class, [])
            source_values = sorted(
                {str(row.get("source_mapped_class", "") or "EMPTY") for row in samples}
            )
            corrections = sum(
                row.get("class_mapping_status") == "CORRECTED_TO_CONFIG" for row in samples
            )
            if rule is None:
                status = "UNMAPPED_CLASS_REVIEW_REQUIRED"
            elif rule.get("review_required", True):
                status = "DEFINED_PENDING_DATA_LEAD_APPROVAL"
            else:
                status = "APPROVED"
            label_rows.append(
                {
                    "dataset_name": dataset,
                    "original_class": original_class,
                    "reported_annotation_count": reported_count,
                    "configured_mapped_class": (rule or {}).get("mapped_class", ""),
                    "configured_include": (rule or {}).get("include", ""),
                    "bbox_sample_count": len(samples),
                    "source_mapped_values": "|".join(source_values),
                    "corrected_bbox_sample_count": corrections,
                    "mapping_status": status,
                    "manual_review_required": (rule or {}).get("review_required", True),
                }
            )

        pending_label_classes = sum(
            row["dataset_name"] == dataset and row["manual_review_required"] for row in label_rows
        )
        add_queue(dataset, "CORRUPT_OR_UNREADABLE", failed, "CRITICAL", "IMAGE_SAMPLE", "QUARANTINE_AND_RECHECK_SOURCE")
        add_queue(dataset, "INVALID_ANNOTATION", invalid_unique, "HIGH", "ANNOTATION_SCAN", "REVIEW_MALFORMED_BOX_AND_EXCLUDE_ONLY_IF_NO_VISIBLE_AREA_AFTER_CLIP")
        add_queue(dataset, "BOUNDARY_BBOX_CLIPPED", boundary_clipped, "INFO", "ANNOTATION_SCAN", "KEEP_OBJECT_AND_USE_CLIPPED_COORDINATES")
        add_queue(dataset, "EXACT_DUPLICATE", exact_groups, "HIGH", "IMAGE_SAMPLE", "KEEP_ONE_PER_CONFIRMED_DUPLICATE_GROUP")
        add_queue(dataset, "NEAR_DUPLICATE", near_groups, "MEDIUM", "IMAGE_SAMPLE", "TEMPORAL_DOWNSAMPLE_AFTER_REVIEW")
        add_queue(dataset, "BLUR_SUSPECT", blurred, "MEDIUM", "IMAGE_SAMPLE", "VISUAL_REVIEW_THEN_REJECT_OR_KEEP")
        add_queue(dataset, "UNDEREXPOSED_SUSPECT", underexposed, "MEDIUM", "IMAGE_SAMPLE", "VISUAL_REVIEW_BY_LIGHTING_SLICE")
        add_queue(dataset, "OVEREXPOSED_SUSPECT", overexposed, "MEDIUM", "IMAGE_SAMPLE", "VISUAL_REVIEW_BY_LIGHTING_SLICE")
        add_queue(dataset, "CLASS_MAPPING_CORRECTED", corrected_mapping, "HIGH", "BBOX_SAMPLE", "USE_CONFIG_MAPPING_AND_INVALIDATE_OLD_MANIFEST")
        add_queue(dataset, "CLASS_POLICY_PENDING", pending_label_classes, "HIGH", "CLASS_TAXONOMY", "DATA_LEAD_APPROVE_INCLUDE_AND_MAPPING")

    return summaries, label_rows, queue


__all__ = ["build_quality_audit"]
