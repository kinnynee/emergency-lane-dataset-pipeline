from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from external_eda_common import load_yaml
from run_external_dataset_eda import _normalize_bbox_class_mapping


def test_car_only_policy_moves_two_wheel_classes_to_ignore_regions() -> None:
    mapping = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "vehicle_class_mapping.yaml")
    assert mapping["target_class"] == "vehicle"
    assert mapping["preserve_original_class"] is True
    assert mapping["mio_tcd"]["motorcycle"] == {
        "mapped_class": None,
        "include": False,
        "handling": "IGNORE_REGION",
        "review_required": False,
    }
    assert mapping["aau_rainsnow"]["motorbike"]["handling"] == "IGNORE_REGION"
    assert mapping["ua_detrac"]["others"]["include"] is True
    others = mapping["ua_detrac"]["others"]
    assert others["review_required"] is False
    assert "CAR_ONLY_POLICY_PENDING_REEXPORT" in mapping["review_status"]
    assert "all 74 unique others tracks" in others["review_note"]
    assert others["track_exclusions"] == [
        {
            "sequence_id": "MVI_40172",
            "track_id": "79",
            "action": "EXCLUDE_NON_VEHICLE_TRACK",
            "bbox_count": 201,
        }
    ]


def test_person_is_not_mapped_to_vehicle() -> None:
    mapping = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "vehicle_class_mapping.yaml")
    assert mapping["mio_tcd"]["pedestrian"]["mapped_class"] is None
    assert mapping["aau_rainsnow"]["person"]["mapped_class"] is None
    assert mapping["mio_tcd"]["bicycle"]["include"] is False
    assert mapping["mio_tcd"]["bicycle"]["handling"] == "IGNORE_REGION"
    assert mapping["aau_rainsnow"]["bicycle"]["include"] is False
    assert mapping["aau_rainsnow"]["bicycle"]["handling"] == "IGNORE_REGION"


def test_cached_bbox_mapping_is_corrected_to_config() -> None:
    mapping = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "vehicle_class_mapping.yaml")
    results = [
        {
            "dataset_name": "MIO-TCD Localization",
            "bbox_samples": [
                {"original_class": "pedestrian", "mapped_class": "vehicle"},
                {"original_class": "car", "mapped_class": "vehicle"},
            ],
        }
    ]
    _normalize_bbox_class_mapping(results, mapping)
    assert results[0]["bbox_samples"][0]["mapped_class"] == ""
    assert results[0]["bbox_samples"][0]["class_mapping_status"] == "CORRECTED_TO_CONFIG"
    assert results[0]["bbox_samples"][1]["class_mapping_status"] == "CONSISTENT"
