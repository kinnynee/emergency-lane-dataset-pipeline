from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_scene_slices import build_scene_slice_analysis
from build_quality_audit import build_quality_audit
from external_eda_common import load_yaml
from validate_split_policy import build_split_audit


def test_scene_slice_analysis_uses_sequence_metadata_and_bbox_sample() -> None:
    results = [
        {
            "dataset_name": "UA-DETRAC Original",
            "sequences": [
                {"sequence_name": "MVI_X", "image_count": 10, "bbox_count": 20}
            ],
            "quality_rows": [],
            "bbox_samples": [
                {
                    "sequence_name": "MVI_X",
                    "original_class": "car",
                    "mapped_class": "vehicle",
                    "bbox_size_category": "SMALL",
                    "box_320_category": "VERY_SMALL",
                }
            ],
        }
    ]
    config = {
        "assessments": {
            "UA-DETRAC Original": {
                "MVI_X": {
                    "road_type": "HIGHWAY",
                    "weather": "CLEAR",
                    "lighting": "DAY",
                    "camera_view": "ELEVATED_OBLIQUE",
                    "traffic_density": "MEDIUM",
                    "mean_vehicles_per_image": 2.0,
                }
            }
        }
    }
    sequences, classes, boxes = build_scene_slice_analysis(results, config)
    assert sequences[0]["image_count_in_scope"] == 10
    assert sequences[0]["bbox_count_in_scope"] == 20
    assert sequences[0]["very_small_320_count"] == 1
    assert any(row["dimension"] == "road_type" and row["value"] == "HIGHWAY" for row in classes)
    assert any(row["metric"] == "POST_RESIZE_320" for row in boxes)


def test_quality_audit_creates_actionable_review_queue() -> None:
    mapping = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "vehicle_class_mapping.yaml")
    results = [
        {
            "dataset_name": "MIO-TCD Localization",
            "quality_rows": [{"read_status": "OK", "blur_suspect": True}],
            "invalid_annotations": [],
            "bbox_samples": [],
            "class_counts": {"car": 1},
        }
    ]
    summary, labels, queue = build_quality_audit(results, [], mapping)
    assert summary[0]["quality_gate"] == "REVIEW_REQUIRED"
    assert labels[0]["mapping_status"] == "DEFINED_PENDING_DATA_LEAD_APPROVAL"
    assert {row["issue_category"] for row in queue} >= {"BLUR_SUSPECT", "CLASS_POLICY_PENDING"}


def test_split_audit_detects_source_file_leakage() -> None:
    policy = {
        "apply_status": "PROPOSAL_ONLY",
        "cross_test_requirements": {"minimum_datasets": 1},
        "main_test": {"status": "PLACEHOLDER_PENDING_COLLECTION", "locked": False},
    }
    splits = [
        {
            "dataset_name": "D",
            "sequence_id": "S1",
            "proposed_split": "EXTERNAL_TRAIN",
            "split_unit": "SEQUENCE_ID",
            "evaluation_eligible": True,
        },
        {
            "dataset_name": "D",
            "sequence_id": "S2",
            "proposed_split": "CROSS_DATASET_TEST",
            "split_unit": "SEQUENCE_ID",
            "evaluation_eligible": True,
        },
    ]
    manifest = [
        {"dataset_name": "D", "sequence_id": "S1", "source_file": "same.jpg"},
        {"dataset_name": "D", "sequence_id": "S2", "source_file": "same.jpg"},
    ]
    validations, _distribution, _holdout = build_split_audit(splits, manifest, policy)
    source_check = next(row for row in validations if row["check_id"] == "SOURCE_FILE_EXCLUSIVE")
    assert source_check["status"] == "FAIL"
