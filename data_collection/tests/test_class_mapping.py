from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from external_eda_common import load_yaml
from run_external_dataset_eda import _normalize_bbox_class_mapping


def test_all_defined_classes_require_review() -> None:
    mapping = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "vehicle_class_mapping.yaml")
    for dataset in ("mio_tcd", "aau_rainsnow", "ua_detrac"):
        assert mapping[dataset]
        assert all(rule["review_required"] is True for rule in mapping[dataset].values())


def test_person_is_not_mapped_to_vehicle() -> None:
    mapping = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "vehicle_class_mapping.yaml")
    assert mapping["mio_tcd"]["pedestrian"]["mapped_class"] is None
    assert mapping["aau_rainsnow"]["person"]["mapped_class"] is None


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
