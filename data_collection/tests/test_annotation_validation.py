from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from external_eda_common import clip_bbox_to_image, letterbox_box_metrics, validate_bbox


def test_out_of_bounds_bbox_is_detected() -> None:
    issues = validate_bbox(10, 20, 110, 220, 100, 200)
    assert "X_OUT_OF_BOUNDS" in issues
    assert "Y_OUT_OF_BOUNDS" in issues


def test_non_positive_bbox_is_detected() -> None:
    assert "NON_POSITIVE_SIZE" in validate_bbox(10, 10, 10, 20, 100, 100)


def test_boundary_crossing_bbox_is_clipped_and_kept() -> None:
    clipped, adjustments = clip_bbox_to_image(-5, 10, 110, 95, 100, 100)
    assert clipped == (0.0, 10, 100.0, 95)
    assert adjustments == ["LEFT_BOUNDARY_CLIPPED", "RIGHT_BOUNDARY_CLIPPED"]
    assert validate_bbox(*clipped, 100, 100) == []


def test_bbox_fully_outside_has_no_visible_area_after_clip() -> None:
    clipped, _adjustments = clip_bbox_to_image(110, 10, 120, 20, 100, 100)
    assert "NON_POSITIVE_SIZE" in validate_bbox(*clipped, 100, 100)


def test_letterbox_categories_follow_prompt_thresholds() -> None:
    assert letterbox_box_metrics(3, 10, 320, 320)["box_320_category"] == "EXTREMELY_TINY"
    assert letterbox_box_metrics(7, 10, 320, 320)["box_320_category"] == "VERY_SMALL"
    assert letterbox_box_metrics(12, 20, 320, 320)["box_320_category"] == "SMALL"
    assert letterbox_box_metrics(20, 20, 320, 320)["box_320_category"] == "USABLE"
