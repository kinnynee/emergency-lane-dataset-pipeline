from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from generate_report_figures import (
    Box,
    EvaluationSample,
    _condition_metrics,
    _condition_overlap,
    average_precision,
    box_iou,
    map50_95,
    size_group,
)


def _sample(labels: tuple[Box, ...], predictions: tuple[Box, ...]) -> EvaluationSample:
    return EvaluationSample(
        image_id="sample",
        image_path=Path("sample.jpg"),
        width=640,
        height=480,
        lighting="DAY",
        weather="CLEAR",
        labels=labels,
        predictions=predictions,
    )


def test_iou_and_average_precision_use_confidence_ordered_one_to_one_matching() -> None:
    label = Box("sample", (0, 0, 20, 20))
    false_positive = Box("sample", (100, 100, 120, 120), 0.95)
    true_positive = Box("sample", (0, 0, 20, 20), 0.90)

    assert box_iou(label.xyxy, true_positive.xyxy) == pytest.approx(1.0)
    assert average_precision([_sample((label,), (false_positive, true_positive))], 0.50) == pytest.approx(0.5)
    assert map50_95([_sample((label,), (false_positive, true_positive))]) == pytest.approx(0.5)


def _condition_sample(image_id: str, lighting: str, weather: str) -> EvaluationSample:
    label = Box(image_id, (0, 0, 20, 20))
    return EvaluationSample(
        image_id=image_id,
        image_path=Path(f"{image_id}.jpg"),
        width=640,
        height=480,
        lighting=lighting,
        weather=weather,
        labels=(label,),
        predictions=(label,),
        dataset="AAU RainSnow",
        split="cross_test",
    )


def test_condition_metrics_marks_small_slices_na_and_reports_rain_night_overlap() -> None:
    samples = [_condition_sample(f"rain-night-{index}", "NIGHT", "RAIN") for index in range(29)]
    samples.append(_condition_sample("night-clear", "NIGHT", "CLEAR"))

    metrics = {row["condition"]: row for row in _condition_metrics(samples, "cross-test")}
    assert metrics["NIGHT"]["sample_status"] == "N/A_INSUFFICIENT_SAMPLE"
    assert metrics["RAIN"]["mAP50_95"] == ""

    overlap = _condition_overlap(samples)
    assert overlap["overlap_image_count"] == 29
    assert overlap["rain_images_also_night_ratio"] == pytest.approx(1.0)
    assert overlap["night_images_also_rain_ratio"] == pytest.approx(29 / 30)


def test_size_groups_keep_the_k230_25_pixel_guard_boundary_explicit() -> None:
    assert size_group(Box("sample", (0, 0, 24, 40)), 640, 480) == "<25 px"
    assert size_group(Box("sample", (0, 0, 25, 40)), 640, 480) == "25–49 px"
    assert size_group(Box("sample", (0, 0, 50, 60)), 640, 480) == "≥50 px"
