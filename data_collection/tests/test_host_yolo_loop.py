from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from host_yolo_loop import (
    GroundPlaneTransform,
    ROIConfig,
    TrackState,
    close_event,
    point_in_roi,
    update_speed_kmh,
    update_stop_alert,
)


def _calibration_payload() -> dict[str, object]:
    return {
        "frame_size_px": [100, 200],
        "image_points_px": [[0, 0], [100, 0], [100, 200], [0, 200]],
        "world_points_m": [[0, 0], [10, 0], [10, 20], [0, 20]],
        "camera_ground_point_m": [5, -2],
    }


def test_ground_plane_calibration_projects_pixels_to_metres(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_calibration_payload()), encoding="utf-8")

    transform = GroundPlaneTransform.from_file(path)

    assert transform.project((50, 100)) == pytest.approx((5.0, 10.0))
    assert transform.range_to_camera_m((50, 100)) == pytest.approx(12.0)
    transform.validate_frame_size(100, 200)
    with pytest.raises(ValueError, match="does not match"):
        transform.validate_frame_size(200, 100)


def test_speed_is_calculated_in_kmh_from_ground_distance() -> None:
    state = TrackState()

    assert update_speed_kmh(state, 0.0, (0.0, 0.0), 2.0) is None
    assert update_speed_kmh(state, 2.0, (10.0, 0.0), 2.0) == pytest.approx(18.0)


def test_static_vehicle_jitter_reliably_produces_one_alert() -> None:
    """Twenty noisy static tracks must meet the >=95% alert-rate release gate."""
    successes = 0
    for seed in range(20):
        randomizer = random.Random(seed)
        state = TrackState()
        events = []
        for frame in range(240):  # 8 s at 30 fps, 1--3 px-equivalent ground jitter.
            now_s = frame / 30.0
            point = (
                10.0 + randomizer.uniform(-0.045, 0.045),
                20.0 + randomizer.uniform(-0.045, 0.045),
            )
            speed = update_speed_kmh(state, now_s, point, 1.5)
            update_stop_alert(
                state, 7, now_s, speed, 10.0, events,
                stop_speed_kmh=2.0,
                stop_seconds=3.0,
                resume_speed_kmh=3.0,
                resume_seconds=1.0,
            )
        if len(events) == 1 and state.active_event is not None:
            successes += 1
    assert successes / 20 >= 0.95


def test_hysteresis_keeps_alert_open_until_sustained_motion() -> None:
    state = TrackState()
    events = []
    for frame in range(240):
        now_s = frame / 30.0
        speed = update_speed_kmh(state, now_s, (4.0, 8.0), 1.0)
        update_stop_alert(
            state, 3, now_s, speed, 5.0, events,
            stop_speed_kmh=2.0, stop_seconds=1.0, resume_speed_kmh=3.0, resume_seconds=0.5,
        )
    assert len(events) == 1
    assert state.active_event is not None

    # A short speed spike does not end the alert.
    for frame in range(240, 247):
        now_s = frame / 30.0
        speed = update_speed_kmh(state, now_s, (4.0 + (frame - 240) * 0.25, 8.0), 1.0)
        update_stop_alert(
            state, 3, now_s, speed, 5.0, events,
            stop_speed_kmh=2.0, stop_seconds=1.0, resume_speed_kmh=3.0, resume_seconds=0.5,
        )
    assert state.active_event is not None

    # Sustained movement beyond the higher threshold closes it once.
    for frame in range(247, 290):
        now_s = frame / 30.0
        speed = update_speed_kmh(state, now_s, (5.75 + (frame - 247) * 0.25, 8.0), 1.0)
        update_stop_alert(
            state, 3, now_s, speed, 5.0, events,
            stop_speed_kmh=2.0, stop_seconds=1.0, resume_speed_kmh=3.0, resume_seconds=0.5,
        )
    assert len(events) == 1
    assert state.active_event is None
    assert events[0].end_reason == "MOTION_RESUMED"
    assert state.ground_positions  # History is retained; the next estimate is not a one-frame sample.


def test_close_event_only_clears_history_when_track_is_discarded() -> None:
    state = TrackState()
    update_speed_kmh(state, 0.0, (0.0, 0.0), 1.0)
    update_speed_kmh(state, 1.0, (1.0, 0.0), 1.0)
    close_event(state, 1.0, "MOTION_RESUMED")
    assert state.ground_positions
    close_event(state, 1.0, "TRACK_LOST", clear_history=True)
    assert not state.ground_positions


def test_invalid_duplicate_calibration_points_are_rejected(tmp_path: Path) -> None:
    payload = _calibration_payload()
    payload["image_points_px"] = [[0, 0], [0, 0], [100, 200], [0, 200]]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="distinct"):
        GroundPlaneTransform.from_file(path)


def test_calibration_requires_a_measured_camera_ground_point(tmp_path: Path) -> None:
    payload = _calibration_payload()
    del payload["camera_ground_point_m"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="camera_ground_point_m"):
        GroundPlaneTransform.from_file(path)


def test_calibrated_polygon_roi_is_bound_to_the_camera_frame(tmp_path: Path) -> None:
    path = tmp_path / "roi.json"
    path.write_text(json.dumps({
        "camera_id": "K230_LANE_A",
        "frame_size_px": [100, 200],
        "status": "CALIBRATED",
        "normalized_polygon": [[0.1, 0.2], [0.9, 0.2], [0.8, 0.9], [0.2, 0.9]],
    }), encoding="utf-8")

    roi = ROIConfig.from_file(path)

    assert roi.to_pixel_polygon() == ((10, 40), (89, 40), (79, 179), (20, 179))
    assert point_in_roi(50, 100, roi.to_pixel_polygon())
    assert not point_in_roi(1, 1, roi.to_pixel_polygon())
    roi.validate_frame_size(100, 200)
    with pytest.raises(ValueError, match="does not match"):
        roi.validate_frame_size(200, 100)


def test_draft_roi_cannot_be_used_for_operational_detection(tmp_path: Path) -> None:
    path = tmp_path / "roi.json"
    path.write_text(json.dumps({
        "camera_id": "K230_LANE_A",
        "frame_size_px": [100, 200],
        "status": "DRAFT_UNCALIBRATED",
        "normalized_polygon": [[0.1, 0.2], [0.9, 0.2], [0.8, 0.9]],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="CALIBRATED"):
        ROIConfig.from_file(path)
