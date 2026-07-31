"""Tạo proposal/manifest, không copy ảnh nếu chưa có --apply-selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from external_eda_common import stable_split

ALLOCATIONS = {
    "PILOT_500": {
        "MIO-TCD Localization": 170,
        "AAU RainSnow": 165,
        "UA-DETRAC Original": 165,
    },
    "DATASET_V1_1500": {
        "MIO-TCD Localization": 500,
        "AAU RainSnow": 500,
        "UA-DETRAC Original": 500,
    },
    "EXTENDED": {
        "MIO-TCD Localization": 1000,
        "AAU RainSnow": 1000,
        "UA-DETRAC Original": 1000,
    },
}


def _unique_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for row in result.get("bbox_samples", []):
        source = str(row.get("source_file", "")).replace("\\", "/")
        if source and source not in candidates:
            candidates[source] = {**row, "source_file": source}
    for row in result.get("quality_rows", []):
        source = str(row.get("source_file", "")).replace("\\", "/")
        if source and source not in candidates:
            candidates[source] = {
                "dataset_name": result["dataset_name"],
                "sequence_name": row.get("sequence_name", "UNKNOWN"),
                "source_file": source,
                "original_class": "NO_VERIFIED_VEHICLE_BOX",
                "mapped_class": "",
                "distance": "UNKNOWN",
                "bbox_size_category": "UNKNOWN",
            }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates.values():
        grouped[str(row.get("sequence_name", "UNKNOWN"))].append(row)
    ordered: list[dict[str, Any]] = []
    while grouped:
        for sequence in sorted(list(grouped)):
            values = grouped[sequence]
            if values:
                ordered.append(values.pop(0))
            if not values:
                grouped.pop(sequence, None)
    return ordered


def create_plan(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    plans: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    selection_index = 1
    for scenario, allocation in ALLOCATIONS.items():
        for result in results:
            dataset = result["dataset_name"]
            target = allocation[dataset]
            candidates = _unique_candidates(result)
            proposed = min(target, len(candidates))
            plans.append(
                {
                    "scenario": scenario,
                    "dataset_name": dataset,
                    "target_images": target,
                    "available_candidates_in_analysis": len(candidates),
                    "proposed_images": proposed,
                    "shortfall": target - proposed,
                    "selection_unit": "sequence_then_file",
                    "apply_status": "PROPOSAL_ONLY",
                    "notes": "Ưu tiên sequence khác nhau; cần manual review điều kiện khó.",
                }
            )
            for row in candidates[:proposed]:
                sequence = str(row.get("sequence_name", "SEQUENCE_NOT_PROVIDED"))
                manifest.append(
                    {
                        "selection_id": f"SEL_{selection_index:06d}",
                        "dataset_name": dataset,
                        "sequence_id": sequence,
                        "source_file": row.get("source_file", ""),
                        "annotation_file": "FROM_SOURCE_DATASET",
                        "original_class": row.get("original_class", ""),
                        "mapped_class": row.get("mapped_class", ""),
                        "lighting": "NOT_VERIFIED",
                        "weather": "NOT_VERIFIED",
                        "distance": row.get("distance", "UNKNOWN"),
                        "bbox_size_category": row.get("bbox_size_category", "UNKNOWN"),
                        "camera_view": "FIXED_OR_ELEVATED_REVIEW_REQUIRED",
                        "selection_reason": "BALANCED_SEQUENCE_AND_DIFFICULTY_PROPOSAL",
                        "target_subset": scenario,
                        "selected": False,
                        "manual_review_status": "PENDING",
                        "notes": "Không copy file; chỉ proposal.",
                    }
                )
                selection_index += 1
            seen_sequences = sorted(
                {
                    str(row.get("sequence_name", "SEQUENCE_NOT_PROVIDED"))
                    for row in candidates[:proposed]
                }
            )
            for sequence in seen_sequences:
                splits.append(
                    {
                        "dataset_name": dataset,
                        "sequence_id": sequence,
                        "proposed_split": stable_split(f"{dataset}:{sequence}"),
                        "apply_status": "PROPOSAL_ONLY",
                        "main_project_test": False,
                        "notes": "Main project test phải dùng K230 tự quay.",
                    }
                )
    unique_splits = {
        (row["dataset_name"], row["sequence_id"], row["proposed_split"]): row for row in splits
    }
    return plans, manifest, list(unique_splits.values())


__all__ = ["ALLOCATIONS", "create_plan"]
