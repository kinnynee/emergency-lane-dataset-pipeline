from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from detect_sequence_leakage import assert_sequence_split
from create_balanced_subset_plan import create_plan
from external_eda_common import load_yaml
from run_external_dataset_eda import _apply_scene_metadata, _scene_assessment_rows


def test_same_sequence_must_not_have_multiple_splits() -> None:
    rows = [
        {"dataset_name": "UA", "sequence_id": "MVI_1", "proposed_split": "EXTERNAL_TRAIN"},
        {"dataset_name": "UA", "sequence_id": "MVI_1", "proposed_split": "CROSS_DATASET_TEST"},
    ]
    with pytest.raises(ValueError):
        assert_sequence_split(rows)


def test_different_sequences_can_have_different_splits() -> None:
    rows = [
        {"dataset_name": "UA", "sequence_id": "MVI_1", "proposed_split": "EXTERNAL_TRAIN"},
        {"dataset_name": "UA", "sequence_id": "MVI_2", "proposed_split": "CROSS_DATASET_TEST"},
    ]
    assert_sequence_split(rows)


def test_road_type_config_uses_allowed_values() -> None:
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "sequence_road_types.yaml")
    allowed = set(config["allowed_road_types"])
    values = {
        item["road_type"]
        for dataset in config["assessments"].values()
        for item in dataset.values()
    }
    assert values <= allowed


def test_scene_metadata_config_uses_allowed_values() -> None:
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "sequence_road_types.yaml")
    assessments = [
        item
        for dataset in config["assessments"].values()
        for item in dataset.values()
    ]
    field_to_allowed = {
        "weather": "allowed_weather",
        "lighting": "allowed_lighting",
        "camera_view": "allowed_camera_views",
        "traffic_density": "allowed_traffic_density",
    }
    for field, allowed_key in field_to_allowed.items():
        assert {item[field] for item in assessments} <= set(config[allowed_key])
    assert all(item["weather"] != "NIGHT" for item in assessments)


def test_traffic_density_matches_documented_thresholds() -> None:
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "sequence_road_types.yaml")
    for dataset in config["assessments"].values():
        for item in dataset.values():
            mean = float(item["mean_vehicles_per_image"])
            expected = "LOW" if mean <= 4.0 else "MEDIUM" if mean <= 10.0 else "HIGH"
            assert item["traffic_density"] == expected


def test_scene_metadata_enriches_split_and_manifest() -> None:
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "sequence_road_types.yaml")
    splits = [
        {
            "dataset_name": "UA-DETRAC Original",
            "sequence_id": "MVI_39851",
            "proposed_split": "CROSS_DATASET_TEST",
        }
    ]
    manifest = [
        {
            "dataset_name": "UA-DETRAC Original",
            "sequence_id": "MVI_39851",
        }
    ]
    road_distribution, scene_distribution = _apply_scene_metadata(splits, manifest, config)
    assert splits[0]["road_type"] == "URBAN_ROAD"
    assert splits[0]["lighting"] == "NIGHT"
    assert splits[0]["weather"] == "UNKNOWN"
    assert manifest[0]["traffic_density"] == "LOW"
    assert road_distribution[0]["sequence_count"] == 1
    assert len(scene_distribution) == 4


def test_scene_review_table_has_one_row_per_configured_sequence() -> None:
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "sequence_road_types.yaml")
    rows = _scene_assessment_rows(config)
    expected = sum(len(sequences) for sequences in config["assessments"].values())
    assert len(rows) == expected == 10
    assert len({(row["dataset_name"], row["sequence_id"]) for row in rows}) == expected


def test_split_policy_keeps_mio_train_only_and_fixed_cross_test() -> None:
    config_root = Path(__file__).resolve().parents[1] / "configs"
    policy = load_yaml(config_root / "split_policy.yaml")
    results = [
        {
            "dataset_name": "MIO-TCD Localization",
            "bbox_samples": [
                {
                    "sequence_name": "SEQUENCE_NOT_PROVIDED",
                    "source_file": "MIO/train/1.jpg",
                    "original_class": "car",
                    "mapped_class": "vehicle",
                }
            ],
            "quality_rows": [],
        },
        {
            "dataset_name": "AAU RainSnow",
            "bbox_samples": [
                {
                    "sequence_name": "Hjorringvej-4",
                    "source_file": "Hjorringvej-4/cam1-1.png",
                    "original_class": "car",
                    "mapped_class": "vehicle",
                }
            ],
            "quality_rows": [],
        },
    ]
    _plans, manifest, splits = create_plan(results, policy)
    mio = next(row for row in splits if row["dataset_name"] == "MIO-TCD Localization")
    aau = next(row for row in splits if row["dataset_name"] == "AAU RainSnow")
    assert mio["sequence_id"] == "MIO_NO_SEQUENCE_TRAIN_ONLY"
    assert mio["proposed_split"] == "EXTERNAL_TRAIN"
    assert mio["evaluation_eligible"] is False
    assert aau["proposed_split"] == "CROSS_DATASET_TEST"
    assert all(row["source_sequence_id"] for row in manifest)
