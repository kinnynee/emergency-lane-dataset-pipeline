from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from host_yolo_loop import GroundPlaneTransform, TrackState, update_speed_kmh


def _calibration_payload() -> dict[str, object]:
    return {
        "frame_size_px": [100, 200],
        "image_points_px": [[0, 0], [100, 0], [100, 200], [0, 200]],
        "world_points_m": [[0, 0], [10, 0], [10, 20], [0, 20]],
    }


def test_ground_plane_calibration_projects_pixels_to_metres(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_calibration_payload()), encoding="utf-8")

    transform = GroundPlaneTransform.from_file(path)

    assert transform.project((50, 100)) == pytest.approx((5.0, 10.0))
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
