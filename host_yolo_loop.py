"""Shared, calibration-aware YOLO tracking loop for a fixed K230 camera.

Stopped-vehicle alerts are based on ground-plane speed (km/h), never image
pixel displacement. A reviewed polygon ROI can be loaded for a fixed camera;
temporary rectangular ROI selection remains available for exploration. A
verified four-point camera calibration is required for every run. The source
video, model, and calibration file are read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Iterable, Mapping, Sequence

import numpy as np


PixelPoint = tuple[float, float]
GroundPoint = tuple[float, float]
PixelROI = tuple[int, int, int, int]
PixelPolygon = tuple[PixelPoint, ...]

REPO_ROOT = Path(__file__).resolve().parent


def load_runtime_contract(path: Path) -> dict[str, object]:
    """Load the deployment values shared by host replay and CanMV."""
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read runtime contract: {path}") from exc
    expected = {
        "format": "k230-team-model-runtime-v1",
        "provenance": "TEAM_TRAINED_ONLY_NO_COCO_FALLBACK",
        "input_size": 320,
        "class_ids": [0],
        "class_names": {"0": "vehicle"},
        "confidence": 0.50,
    }
    mismatches = [key for key, value in expected.items() if contract.get(key) != value]
    if mismatches:
        raise ValueError("Runtime contract mismatch: " + ", ".join(mismatches))
    return contract


@dataclass(frozen=True)
class GroundPlaneTransform:
    """Project pixel locations onto a calibrated ground plane in metres."""

    coefficients: tuple[float, float, float, float, float, float, float, float]
    frame_size_px: tuple[int, int]
    camera_ground_point_m: GroundPoint
    source: Path

    @classmethod
    def from_file(cls, path: Path) -> "GroundPlaneTransform":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read speed calibration: {path}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("Speed calibration must be a JSON object")

        image_points = _points(payload.get("image_points_px"), "image_points_px")
        ground_points = _points(payload.get("world_points_m"), "world_points_m")
        camera_ground_point = _point(payload.get("camera_ground_point_m"), "camera_ground_point_m")
        frame_size = payload.get("frame_size_px")
        if not isinstance(frame_size, Sequence) or len(frame_size) != 2:
            raise ValueError("speed calibration requires frame_size_px: [width, height]")
        width, height = (int(value) for value in frame_size)
        if width <= 0 or height <= 0:
            raise ValueError("frame_size_px must contain positive dimensions")
        return cls(
            coefficients=_solve_homography(image_points, ground_points),
            frame_size_px=(width, height),
            camera_ground_point_m=camera_ground_point,
            source=path.resolve(),
        )

    def validate_frame_size(self, width: int, height: int) -> None:
        if (width, height) != self.frame_size_px:
            raise ValueError(
                "Calibration frame size does not match the video: "
                f"calibration={self.frame_size_px[0]}x{self.frame_size_px[1]}, "
                f"video={width}x{height}. Recalibrate or use the matching stream."
            )

    def project(self, point: PixelPoint) -> GroundPoint:
        """Map one image point to ground-plane metres using the homography."""
        u, v = point
        h11, h12, h13, h21, h22, h23, h31, h32 = self.coefficients
        denominator = h31 * u + h32 * v + 1.0
        if abs(denominator) < 1e-9:
            raise ValueError("Point projects to infinity; check the calibration quadrilateral")
        return (
            (h11 * u + h12 * v + h13) / denominator,
            (h21 * u + h22 * v + h23) / denominator,
        )

    def range_to_camera_m(self, point: PixelPoint) -> float:
        """Return ground distance from the camera's measured ground point.

        ``point`` should be the vehicle's ground-contact point (normally the
        bottom centre of its bounding box).  This is not a global px-to-metre
        scale: perspective makes the physical length of a pixel depend on its
        position in the image.
        """
        ground_x, ground_y = self.project(point)
        camera_x, camera_y = self.camera_ground_point_m
        return math.hypot(ground_x - camera_x, ground_y - camera_y)


@dataclass(frozen=True)
class ROIConfig:
    """A reviewed polygon ROI tied to one fixed-camera frame size."""

    camera_id: str
    frame_size_px: tuple[int, int]
    normalized_polygon: tuple[PixelPoint, ...]
    source: Path

    @classmethod
    def from_file(cls, path: Path) -> "ROIConfig":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read ROI config: {path}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("ROI config must be a JSON object")
        if str(payload.get("status", "")).upper() != "CALIBRATED":
            raise ValueError("ROI config status must be CALIBRATED before operational use")
        camera_id = str(payload.get("camera_id", "")).strip()
        if not camera_id:
            raise ValueError("ROI config requires a non-empty camera_id")
        frame_size = payload.get("frame_size_px")
        if not isinstance(frame_size, Sequence) or isinstance(frame_size, (str, bytes)) or len(frame_size) != 2:
            raise ValueError("ROI config requires frame_size_px: [width, height]")
        width, height = (int(value) for value in frame_size)
        if width <= 0 or height <= 0:
            raise ValueError("ROI config frame_size_px must contain positive dimensions")
        raw_polygon = payload.get("normalized_polygon")
        if not isinstance(raw_polygon, Sequence) or isinstance(raw_polygon, (str, bytes)) or len(raw_polygon) < 3:
            raise ValueError("ROI config normalized_polygon must contain at least three [x, y] points")
        polygon = tuple(_point(item, "normalized_polygon point") for item in raw_polygon)
        if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in polygon):
            raise ValueError("ROI config normalized_polygon coordinates must be in [0, 1]")
        if len(set(polygon)) != len(polygon):
            raise ValueError("ROI config normalized_polygon points must be distinct")
        return cls(camera_id, (width, height), polygon, path.resolve())

    def validate_frame_size(self, width: int, height: int) -> None:
        if (width, height) != self.frame_size_px:
            raise ValueError(
                "ROI config frame size does not match the video: "
                f"roi={self.frame_size_px[0]}x{self.frame_size_px[1]}, video={width}x{height}."
            )

    def to_pixel_polygon(self) -> PixelPolygon:
        width, height = self.frame_size_px
        return tuple((round(x * (width - 1)), round(y * (height - 1))) for x, y in self.normalized_polygon)


@dataclass
class StopEvent:
    event_id: int
    track_id: int
    started_at_s: float
    alerted_at_s: float
    speed_kmh_at_alert: float
    range_to_camera_m_at_alert: float
    last_seen_at_s: float
    ended_at_s: float | None = None
    end_reason: str = "EOF"


@dataclass
class TrackState:
    ground_positions: Deque[tuple[float, float, float]] = field(default_factory=deque)
    stopped_since_s: float | None = None
    last_seen_s: float = 0.0
    active_event: StopEvent | None = None


def _points(value: object, field_name: str) -> tuple[PixelPoint, PixelPoint, PixelPoint, PixelPoint]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{field_name} must contain exactly four [x, y] points")
    parsed: list[PixelPoint] = []
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise ValueError(f"Every {field_name} item must be [x, y]")
        x, y = (float(component) for component in item)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"{field_name} must contain finite coordinates")
        parsed.append((x, y))
    if len(set(parsed)) != 4:
        raise ValueError(f"{field_name} points must be distinct")
    return tuple(parsed)  # type: ignore[return-value]


def _point(value: object, field_name: str) -> PixelPoint:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field_name} must be [x, y]")
    x, y = (float(component) for component in value)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{field_name} must contain finite coordinates")
    return x, y


def _solve_linear_system(matrix: list[list[float]], values: list[float]) -> list[float]:
    """Solve a small square system with pivoted Gaussian elimination."""
    size = len(values)
    augmented = [row[:] + [value] for row, value in zip(matrix, values)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-9:
            raise ValueError("Calibration points are degenerate; choose four non-collinear road points")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            multiplier = augmented[row][column]
            augmented[row] = [
                current - multiplier * pivot_value
                for current, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [row[-1] for row in augmented]


def _solve_homography(
    image_points: tuple[PixelPoint, PixelPoint, PixelPoint, PixelPoint],
    ground_points: tuple[GroundPoint, GroundPoint, GroundPoint, GroundPoint],
) -> tuple[float, float, float, float, float, float, float, float]:
    matrix: list[list[float]] = []
    values: list[float] = []
    for (u, v), (x, y) in zip(image_points, ground_points):
        matrix.append([u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v])
        values.append(x)
        matrix.append([0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v])
        values.append(y)
    return tuple(_solve_linear_system(matrix, values))  # type: ignore[return-value]


def parse_classes(value: str) -> list[int]:
    try:
        classes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--classes must contain integer IDs, e.g. 2,3,5,7") from exc
    if not classes:
        raise argparse.ArgumentTypeError("--classes cannot be empty")
    return classes


def parse_roi(value: str) -> tuple[float, float, float, float]:
    try:
        x1, y1, x2, y2 = (float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--roi must be x1,y1,x2,y2, for example 0.15,0.45,0.85,0.95") from exc
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise argparse.ArgumentTypeError("ROI values must be normalized 0..1 and satisfy x1<x2, y1<y2")
    return x1, y1, x2, y2


def point_in_roi(x: float, y: float, roi: PixelPolygon) -> bool:
    """Return whether a point is inside a polygon, including its boundary."""
    inside = False
    previous_x, previous_y = roi[-1]
    for current_x, current_y in roi:
        cross = (x - previous_x) * (current_y - previous_y) - (y - previous_y) * (current_x - previous_x)
        if abs(cross) < 1e-9 and min(previous_x, current_x) <= x <= max(previous_x, current_x) and min(previous_y, current_y) <= y <= max(previous_y, current_y):
            return True
        intersects = (current_y > y) != (previous_y > y) and x < (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
        if intersects:
            inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def select_roi(cv2: object, frame: object) -> PixelPolygon:
    """Provide the shared OpenCV ROI picker used by every host loop."""
    selected = cv2.selectROI("Draw traffic ROI then press Enter", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Draw traffic ROI then press Enter")
    x, y, width, height = (int(value) for value in selected)
    if width <= 0 or height <= 0:
        raise ValueError("ROI selection was cancelled or empty")
    return _rectangle_to_polygon((x, y, x + width, y + height))


def update_speed_kmh(
    state: TrackState,
    now_s: float,
    ground_point: GroundPoint,
    window_s: float,
) -> float | None:
    """Return speed in km/h measured across the calibrated ground-plane window."""
    state.ground_positions.append((now_s, ground_point[0], ground_point[1]))
    while state.ground_positions and state.ground_positions[0][0] < now_s - window_s:
        state.ground_positions.popleft()
    if len(state.ground_positions) < 2:
        return None
    started_at_s, start_x, start_y = state.ground_positions[0]
    duration_s = now_s - started_at_s
    if duration_s <= 0:
        return None
    return 3.6 * math.hypot(ground_point[0] - start_x, ground_point[1] - start_y) / duration_s


def open_stop_event(
    state: TrackState,
    track_id: int,
    now_s: float,
    speed_kmh: float,
    range_to_camera_m: float,
    events: list[StopEvent],
) -> StopEvent:
    event = StopEvent(
        event_id=len(events) + 1,
        track_id=track_id,
        started_at_s=state.stopped_since_s if state.stopped_since_s is not None else now_s,
        alerted_at_s=now_s,
        speed_kmh_at_alert=speed_kmh,
        range_to_camera_m_at_alert=range_to_camera_m,
        last_seen_at_s=now_s,
    )
    events.append(event)
    state.active_event = event
    return event


def close_event(state: TrackState, now_s: float, reason: str) -> None:
    if state.active_event is not None:
        state.active_event.ended_at_s = now_s
        state.active_event.end_reason = reason
        state.active_event = None
    state.ground_positions.clear()
    state.stopped_since_s = None


def write_events(
    path: Path,
    events: Iterable[StopEvent],
    video: Path,
    roi: PixelPolygon,
    roi_source: str,
    calibration: GroundPlaneTransform,
    stop_speed_kmh: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_id", "track_id", "video_file", "roi_px_polygon", "roi_source", "speed_calibration", "camera_ground_point_m", "speed_threshold_kmh",
                "stopped_started_at_s", "alerted_at_s", "speed_kmh_at_alert", "last_seen_at_s", "ended_at_s",
                "range_to_camera_m_at_alert", "dwell_at_alert_s", "end_reason", "review_status",
            ],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "event_id": event.event_id,
                    "track_id": event.track_id,
                    "video_file": str(video),
                    "roi_px_polygon": ";".join(f"{round(x)},{round(y)}" for x, y in roi),
                    "roi_source": roi_source,
                    "speed_calibration": str(calibration.source),
                    "camera_ground_point_m": ",".join(f"{value:.3f}" for value in calibration.camera_ground_point_m),
                    "speed_threshold_kmh": f"{stop_speed_kmh:.2f}",
                    "stopped_started_at_s": f"{event.started_at_s:.2f}",
                    "alerted_at_s": f"{event.alerted_at_s:.2f}",
                    "speed_kmh_at_alert": f"{event.speed_kmh_at_alert:.3f}",
                    "range_to_camera_m_at_alert": f"{event.range_to_camera_m_at_alert:.3f}",
                    "last_seen_at_s": f"{event.last_seen_at_s:.2f}",
                    "ended_at_s": "" if event.ended_at_s is None else f"{event.ended_at_s:.2f}",
                    "dwell_at_alert_s": f"{event.alerted_at_s - event.started_at_s:.2f}",
                    "end_reason": event.end_reason,
                    "review_status": "NEEDS_MANUAL_REVIEW",
                }
            )


def load_dependencies() -> tuple[object, object]:
    try:
        import cv2
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing replay dependencies. Install Python 3.10+ then run:\n"
            "  py -m pip install -r requirements-replay.txt\n"
            f"Missing module: {exc.name}"
        ) from exc
    return cv2, YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", required=True, type=Path, help="Input video; it is never changed.")
    parser.add_argument("--model", required=True, type=Path, help="Local YOLO .pt model file.")
    parser.add_argument("--runtime-contract", type=Path, default=REPO_ROOT / "k230" / "runtime_contract.json", help="Shared Host/K230 model contract.")
    parser.add_argument("--speed-calibration", required=True, type=Path, help="Verified ground-plane calibration JSON.")
    roi_group = parser.add_mutually_exclusive_group(required=True)
    roi_group.add_argument("--roi-config", type=Path, help="Reviewed CALIBRATED polygon ROI JSON for this camera.")
    roi_group.add_argument("--roi", type=parse_roi, help="Temporary normalized rectangular ROI: x1,y1,x2,y2.")
    roi_group.add_argument("--select-roi", action="store_true", help="Draw a temporary rectangular ROI on the first frame with the mouse.")
    parser.add_argument("--output", type=Path, help="Annotated MP4 output path.")
    parser.add_argument("--events", type=Path, help="CSV alert log output path.")
    parser.add_argument("--classes", type=parse_classes, default=parse_classes("0"), help="Fixed to class 0: the project's one-class car/approved four-wheel-vehicle model.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker configuration.")
    parser.add_argument("--confidence", type=float, default=0.50, help="Minimum detection confidence. The operational value is locked at 0.50.")
    parser.add_argument("--iou", type=float, default=0.50, help="Tracker/detection IoU threshold.")
    parser.add_argument("--speed-window-sec", type=float, default=1.5, help="Ground-plane interval used to estimate km/h.")
    parser.add_argument("--stop-speed-kmh", type=float, default=2.0, help="Maximum calibrated speed that counts as stationary.")
    parser.add_argument("--stop-seconds", type=float, default=3.0, help="Seconds below stop speed inside ROI before alert.")
    parser.add_argument("--max-track-gap-sec", type=float, default=1.0, help="Close an alert after a missing track exceeds this gap.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame; keep 1 for final review.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many decoded frames; 0 processes all.")
    parser.add_argument("--display", action="store_true", help="Show the replay window; press Q to stop safely.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.video.is_file():
        raise ValueError(f"Video not found: {args.video}")
    if not args.model.is_file():
        raise ValueError(f"Model not found: {args.model}")
    contract = load_runtime_contract(args.runtime_contract)
    if args.classes != contract["class_ids"]:
        raise ValueError("--classes must match the shared Host/K230 runtime contract")
    if not math.isclose(args.confidence, float(contract["confidence"]), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("--confidence must match the shared Host/K230 runtime contract")
    if args.classes != [0]:
        raise ValueError("Only class 0 from the team one-class model is permitted; COCO class IDs are not supported")
    if not math.isclose(args.confidence, 0.50, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Operational replay is locked to --confidence 0.50")
    if not args.speed_calibration.is_file():
        raise ValueError(f"Speed calibration not found: {args.speed_calibration}")
    if args.roi_config is not None and not args.roi_config.is_file():
        raise ValueError(f"ROI config not found: {args.roi_config}")
    if not 0 < args.confidence <= 1 or not 0 < args.iou <= 1:
        raise ValueError("--confidence and --iou must be in (0, 1]")
    if min(args.speed_window_sec, args.stop_speed_kmh, args.stop_seconds, args.max_track_gap_sec) <= 0:
        raise ValueError("Speed and stop thresholds must be positive")
    if args.frame_stride < 1 or args.max_frames < 0:
        raise ValueError("--frame-stride must be >= 1 and --max-frames must be >= 0")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        calibration = GroundPlaneTransform.from_file(args.speed_calibration)
        roi_config = ROIConfig.from_file(args.roi_config) if args.roi_config else None
        cv2, YOLO = load_dependencies()
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    output = args.output or Path("runs") / "replay" / f"{args.video.stem}_stopped.mp4"
    events_path = args.events or output.with_suffix(".events.csv")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        parser.error(f"Cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        parser.error("Video has invalid frame dimensions")
    try:
        calibration.validate_frame_size(width, height)
        if roi_config is not None:
            roi_config.validate_frame_size(width, height)
    except ValueError as exc:
        capture.release()
        parser.error(str(exc))

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        parser.error("Cannot read first video frame")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    try:
        if roi_config is not None:
            roi = roi_config.to_pixel_polygon()
            roi_source = f"CALIBRATED:{roi_config.camera_id}:{roi_config.source}"
        elif args.select_roi:
            roi = select_roi(cv2, first_frame)
            roi_source = "TEMPORARY:interactive_rectangle"
        else:
            roi = _rectangle_to_polygon(_normalized_to_pixel_roi(args.roi, width, height))
            roi_source = "TEMPORARY:cli_rectangle"
    except ValueError as exc:
        capture.release()
        parser.error(str(exc))

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        parser.error(f"Cannot create annotated output: {output}")

    model = YOLO(str(args.model))
    states: dict[int, TrackState] = {}
    events: list[StopEvent] = []
    decoded_frames = written_frames = 0
    stopped_now: set[int] = set()
    user_quit = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded_frames += 1
            if args.max_frames and decoded_frames > args.max_frames:
                break
            if (decoded_frames - 1) % args.frame_stride:
                continue
            now_s = (decoded_frames - 1) / fps
            stopped_now.clear()
            seen_ids: set[int] = set()
            result = model.track(frame, persist=True, tracker=args.tracker, classes=args.classes, conf=args.confidence, iou=args.iou, verbose=False)[0]
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                coordinates = boxes.xyxy.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                for track_id, (x1, y1, x2, y2), confidence, class_id in zip(ids, coordinates, confidences, class_ids):
                    foot = ((x1 + x2) / 2.0, float(y2))
                    state = states.setdefault(track_id, TrackState())
                    seen_ids.add(track_id)
                    state.last_seen_s = now_s
                    in_roi = point_in_roi(*foot, roi)
                    speed_kmh: float | None = None
                    range_to_camera_m: float | None = None
                    is_stopped = False
                    if in_roi:
                        ground_point = calibration.project(foot)
                        range_to_camera_m = calibration.range_to_camera_m(foot)
                        speed_kmh = update_speed_kmh(state, now_s, ground_point, args.speed_window_sec)
                        if speed_kmh is not None and speed_kmh <= args.stop_speed_kmh:
                            if state.stopped_since_s is None:
                                state.stopped_since_s = state.ground_positions[0][0]
                            if now_s - state.stopped_since_s >= args.stop_seconds:
                                is_stopped = True
                                stopped_now.add(track_id)
                                if state.active_event is None:
                                    open_stop_event(state, track_id, now_s, speed_kmh, range_to_camera_m, events)
                                state.active_event.last_seen_at_s = now_s
                        elif speed_kmh is not None:
                            close_event(state, now_s, "MOTION_RESUMED")
                    else:
                        close_event(state, now_s, "LEFT_ROI")

                    color = (0, 0, 255) if is_stopped else ((0, 200, 0) if in_roi else (160, 160, 160))
                    speed_text = "calibrating" if speed_kmh is None else f"{speed_kmh:.1f} km/h"
                    range_text = "outside ROI" if range_to_camera_m is None else f"{range_to_camera_m:.1f} m"
                    dwell_text = "STOPPED" if is_stopped else ("IN_ROI" if in_roi else "OUTSIDE_ROI")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID {track_id} C{class_id} {confidence:.2f} {range_text} {speed_text} {dwell_text}", (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

            for track_id, state in list(states.items()):
                if track_id not in seen_ids and now_s - state.last_seen_s > args.max_track_gap_sec:
                    close_event(state, state.last_seen_s, "TRACK_LOST")
                    del states[track_id]

            roi_color = (0, 0, 255) if stopped_now else (0, 215, 255)
            roi_points = np.array([(round(x), round(y)) for x, y in roi], dtype=np.int32)
            cv2.polylines(frame, [roi_points], True, roi_color, 3, cv2.LINE_AA)
            cv2.putText(frame, f"STOPPED ALERTS: {len(stopped_now)} | limit: {args.stop_speed_kmh:.1f} km/h | t={now_s:.1f}s", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, roi_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "Range/speed are calibrated on the ground plane; manual review required", (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            written_frames += 1
            if args.display:
                cv2.imshow("Stopped-vehicle replay (Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF in {ord("q"), ord("Q")}:
                    user_quit = True
                    break
    finally:
        final_time_s = max(0.0, (decoded_frames - 1) / fps)
        for state in states.values():
            if state.active_event is not None:
                close_event(state, final_time_s, "USER_QUIT" if user_quit else "END_OF_VIDEO")
        capture.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()

    write_events(events_path, events, args.video, roi, roi_source, calibration, args.stop_speed_kmh)
    print(f"Annotated replay: {output}")
    print(f"Alert log: {events_path}")
    print(f"Frames written: {written_frames}; stopped alerts: {len(events)}")
    return 0


def _normalized_to_pixel_roi(roi: tuple[float, float, float, float], width: int, height: int) -> PixelROI:
    x1, y1, x2, y2 = roi
    return round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)


def _rectangle_to_polygon(roi: PixelROI) -> PixelPolygon:
    x1, y1, x2, y2 = roi
    return (float(x1), float(y1)), (float(x2), float(y1)), (float(x2), float(y2)), (float(x1), float(y2))


if __name__ == "__main__":
    raise SystemExit(main())
