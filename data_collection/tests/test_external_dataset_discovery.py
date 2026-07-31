from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from discover_external_datasets import discover
from inspect_mio_tcd import inspect_mio


def test_explicit_classification_path_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "MIO-TCD-Classification.tar"
    path.touch()
    with pytest.raises(ValueError):
        discover(tmp_path, mio_path=str(path))


def test_mio_inspector_rejects_classification_even_before_reading(tmp_path: Path) -> None:
    path = tmp_path / "classification.tar"
    path.touch()
    with pytest.raises(ValueError):
        inspect_mio(path, sample_size=1, progress=False)


def test_discovery_never_returns_radiate_for_allowed_slots(tmp_path: Path) -> None:
    radiate = tmp_path / "RADIATE"
    radiate.mkdir()
    found = discover(tmp_path)
    assert all("radiate" not in str(value).lower() for value in found.values() if value)
