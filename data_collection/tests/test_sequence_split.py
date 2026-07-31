from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from detect_sequence_leakage import assert_sequence_split


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
