"""Chấm điểm rule-based, mọi kết quả đều cần review thủ công."""

from __future__ import annotations

from typing import Any


def _level(score: float) -> str:
    if score >= 4.3:
        return "HIGHLY_RELEVANT"
    if score >= 3.5:
        return "RELEVANT"
    if score >= 2.5:
        return "PARTIALLY_RELEVANT"
    if score >= 1.5:
        return "LOW_RELEVANCE"
    return "EXCLUDED"


def assess_viewpoints(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        dataset = result["dataset_name"]
        if dataset == "MIO-TCD Localization":
            score = 3.7
            rows.append(
                {
                    "dataset_name": dataset,
                    "sequence_name": "SEQUENCE_NOT_PROVIDED",
                    "camera_motion": "LIKELY_FIXED_UNVERIFIED",
                    "camera_height_estimate": "ELEVATED_UNVERIFIED",
                    "view_direction": "ROADSIDE_OBLIQUE",
                    "pitch_category": "DOWNWARD_UNVERIFIED",
                    "road_area_visibility": "MEDIUM",
                    "vehicle_scale_similarity": "MANUAL_REVIEW",
                    "fixed_camera_similarity": "HIGH",
                    "emergency_lane_similarity": "NOT_VERIFIED",
                    "night_suitability": "NOT_VERIFIED",
                    "rain_suitability": "NOT_VERIFIED",
                    "overall_score": score,
                    "relevance_level": _level(score),
                    "assessment_source": "DATASET_DOCUMENTATION_AND_AUTOMATIC_ESTIMATE",
                    "manual_review_status": "PENDING",
                    "notes": "Không có sequence/camera metadata trong TAR Localization.",
                }
            )
        elif dataset == "AAU RainSnow":
            for sequence in result.get("sequences", []):
                lighting = str(sequence.get("lighting", "UNKNOWN"))
                score = 4.2
                if "NIGHT" in lighting or "TWILIGHT" in lighting:
                    score += 0.2
                score = min(score, 5.0)
                rows.append(
                    {
                        "dataset_name": dataset,
                        "sequence_name": f"{sequence.get('sequence_name')}:{sequence.get('camera_name')}",
                        "camera_motion": sequence.get("camera_motion", "UNKNOWN"),
                        "camera_height_estimate": "ELEVATED_UNVERIFIED",
                        "view_direction": "FIXED_INTERSECTION_OR_ROAD",
                        "pitch_category": "DOWNWARD_UNVERIFIED",
                        "road_area_visibility": "HIGH_AUTOMATIC_ESTIMATE",
                        "vehicle_scale_similarity": "MANUAL_REVIEW",
                        "fixed_camera_similarity": "HIGH",
                        "emergency_lane_similarity": "PARTIAL",
                        "night_suitability": "HIGH" if "NIGHT" in lighting else "AVAILABLE_UNVERIFIED",
                        "rain_suitability": "HIGH_DATASET_LEVEL",
                        "overall_score": round(score, 2),
                        "relevance_level": _level(score),
                        "assessment_source": "VIDEO_METADATA_AND_AUTOMATIC_ESTIMATE",
                        "manual_review_status": "PENDING",
                        "notes": "Rain/snow cụ thể theo sequence chưa được metadata tách rõ.",
                    }
                )
        elif dataset == "UA-DETRAC Original":
            for sequence in result.get("sequences", []):
                stable = str(sequence.get("camera_state", "")).lower() == "stable"
                rainy = str(sequence.get("weather", "")).lower() == "rainy"
                score = 4.4 if stable else 3.9
                if rainy:
                    score = min(5.0, score + 0.2)
                rows.append(
                    {
                        "dataset_name": dataset,
                        "sequence_name": sequence.get("sequence_name", ""),
                        "camera_motion": sequence.get("camera_state", "UNKNOWN"),
                        "camera_height_estimate": "ELEVATED_UNVERIFIED",
                        "view_direction": "ROADSIDE_ELEVATED",
                        "pitch_category": "DOWNWARD",
                        "road_area_visibility": "HIGH_AUTOMATIC_ESTIMATE",
                        "vehicle_scale_similarity": "MANUAL_REVIEW",
                        "fixed_camera_similarity": "HIGH" if stable else "MEDIUM",
                        "emergency_lane_similarity": "PARTIAL",
                        "night_suitability": "NOT_VERIFIED",
                        "rain_suitability": "HIGH" if rainy else "LOW_OR_UNKNOWN",
                        "overall_score": round(score, 2),
                        "relevance_level": _level(score),
                        "assessment_source": "FROM_XML_METADATA_AND_AUTOMATIC_ESTIMATE",
                        "manual_review_status": "PENDING",
                        "notes": "Độ cao chính xác và độ giống camera K230 cần review contact sheet.",
                    }
                )
    rows.append(
        {
            "dataset_name": "RADIATE",
            "sequence_name": "NOT_ANALYZED",
            "camera_motion": "EGO_VEHICLE",
            "camera_height_estimate": "VEHICLE_MOUNTED",
            "view_direction": "FORWARD",
            "pitch_category": "FRONTAL",
            "road_area_visibility": "FORWARD_ONLY",
            "vehicle_scale_similarity": "LOW",
            "fixed_camera_similarity": "LOW",
            "emergency_lane_similarity": "LOW",
            "night_suitability": "NOT_ANALYZED",
            "rain_suitability": "NOT_ANALYZED",
            "overall_score": 1.0,
            "relevance_level": "EXCLUDED",
            "assessment_source": "PROJECT_SELECTION_POLICY",
            "manual_review_status": "EXCLUDED_VIEWPOINT_MISMATCH",
            "notes": "Góc camera phía trước phương tiện không phù hợp camera cố định trên cao; không chạy EDA.",
        }
    )
    return rows


__all__ = ["assess_viewpoints"]
