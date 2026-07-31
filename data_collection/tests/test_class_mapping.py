from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from external_eda_common import load_yaml


def test_all_defined_classes_require_review() -> None:
    mapping = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "vehicle_class_mapping.yaml")
    for dataset in ("mio_tcd", "aau_rainsnow", "ua_detrac"):
        assert mapping[dataset]
        assert all(rule["review_required"] is True for rule in mapping[dataset].values())


def test_person_is_not_mapped_to_vehicle() -> None:
    mapping = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "vehicle_class_mapping.yaml")
    assert mapping["mio_tcd"]["pedestrian"]["mapped_class"] is None
    assert mapping["aau_rainsnow"]["person"]["mapped_class"] is None
