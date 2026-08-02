from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from external_eda_common import load_yaml
from run_external_dataset_eda import _normalize_bbox_class_mapping


def test_supervisor_vehicle_policy_is_applied() -> None:
    mapping = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "vehicle_class_mapping.yaml")
    assert mapping["target_class"] == "vehicle"
    assert mapping["preserve_original_class"] is True
    assert mapping["mio_tcd"]["motorcycle"]["include"] is True
    assert mapping["aau_rainsnow"]["motorbike"]["mapped_class"] == "vehicle"
    assert mapping["ua_detrac"]["others"]["include"] is True
    assert mapping["ua_detrac"]["others"]["review_required"] is True
    assert "CONDITIONAL_PENDING_DATA_LEAD_SIGNOFF" in mapping["review_status"]
    assert "60-sample stratified pre-review" in mapping["ua_detrac"]["others"]["review_note"]


def test_person_is_not_mapped_to_vehicle() -> None:
    mapping = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "vehicle_class_mapping.yaml")
    assert mapping["mio_tcd"]["pedestrian"]["mapped_class"] is None
    assert mapping["aau_rainsnow"]["person"]["mapped_class"] is None
    assert mapping["mio_tcd"]["bicycle"]["include"] is False
    assert mapping["aau_rainsnow"]["bicycle"]["include"] is False


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
