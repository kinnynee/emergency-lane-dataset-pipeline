from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from host_yolo_loop import GroundPlaneTransform, ROIConfig, TrackState, point_in_roi, update_speed_kmh


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
