"""Run stopped-vehicle replay on a video with YOLO tracking and an ROI.

This is an operator-assistance tool, not ground-truth annotation. It is best
used with a fixed/elevated traffic camera; dashboard cameras move with the
road and can produce false stopped-vehicle alerts.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque


@dataclass
class StopEvent:
    """One continuous stopped interval for a tracked vehicle."""

    event_id: int
    track_id: int
    started_at_s: float
    alerted_at_s: float
    last_seen_at_s: float
    ended_at_s: float | None = None
    end_reason: str = "EOF"


@dataclass
class TrackState:
    """Motion history and active alert state for one tracker ID."""

    positions: Deque[tuple[float, float, float]] = field(default_factory=deque)
    stopped_since_s: float | None = None
    last_seen_s: float = 0.0
    active_event: StopEvent | None = None


def parse_classes(value: str) -> list[int]:
    """Parse a comma-separated list of YOLO class IDs."""
    try:
        classes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--classes must contain integer IDs, e.g. 2,3,5,7") from exc
    if not classes:
        raise argparse.ArgumentTypeError("--classes cannot be empty")
    return classes


def parse_roi(value: str) -> tuple[float, float, float, float]:
    """Parse a normalized x1,y1,x2,y2 ROI and validate its bounds."""
    try:
        x1, y1, x2, y2 = (float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--roi must be x1,y1,x2,y2, for example 0.15,0.45,0.85,0.95") from exc
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise argparse.ArgumentTypeError("ROI values must be normalized 0..1 and satisfy x1<x2, y1<y2")
    return x1, y1, x2, y2


def point_in_roi(x: float, y: float, roi: tuple[int, int, int, int]) -> bool:
    """Return whether a point is inside the rectangular ROI."""
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


def stop_event(state: TrackState, track_id: int, now_s: float, events: list[StopEvent]) -> StopEvent:
    """Open a stop event exactly once for a track."""
    event = StopEvent(
        event_id=len(events) + 1,
        track_id=track_id,
        started_at_s=state.stopped_since_s if state.stopped_since_s is not None else now_s,
        alerted_at_s=now_s,
        last_seen_at_s=now_s,
    )
    events.append(event)
    state.active_event = event
    return event


def close_event(state: TrackState, now_s: float, reason: str) -> None:
    """Close an alert while retaining it for the output CSV."""
    if state.active_event is not None:
        state.active_event.ended_at_s = now_s
        state.active_event.end_reason = reason
        state.active_event = None
    state.positions.clear()
    state.stopped_since_s = None


def write_events(path: Path, events: list[StopEvent], video: Path, roi: tuple[int, int, int, int]) -> None:
    """Write an audit-friendly event log after replay ends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_id",
                "track_id",
                "video_file",
                "roi_px",
                "stopped_started_at_s",
                "alerted_at_s",
                "last_seen_at_s",
                "ended_at_s",
                "dwell_at_alert_s",
                "end_reason",
                "review_status",
            ],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "event_id": event.event_id,
                    "track_id": event.track_id,
                    "video_file": str(video),
                    "roi_px": ",".join(str(value) for value in roi),
                    "stopped_started_at_s": f"{event.started_at_s:.2f}",
                    "alerted_at_s": f"{event.alerted_at_s:.2f}",
                    "last_seen_at_s": f"{event.last_seen_at_s:.2f}",
                    "ended_at_s": "" if event.ended_at_s is None else f"{event.ended_at_s:.2f}",
                    "dwell_at_alert_s": f"{event.alerted_at_s - event.started_at_s:.2f}",
                    "end_reason": event.end_reason,
                    "review_status": "NEEDS_MANUAL_REVIEW",
                }
            )


def load_dependencies() -> tuple[object, object]:
    """Import optional runtime dependencies only when replay actually runs."""
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video; it is never changed.")
    parser.add_argument("--model", required=True, type=Path, help="Local YOLO .pt model file.")
    roi_group = parser.add_mutually_exclusive_group(required=True)
    roi_group.add_argument("--roi", type=parse_roi, help="Normalized ROI: x1,y1,x2,y2.")
    roi_group.add_argument("--select-roi", action="store_true", help="Draw the ROI on the first frame with the mouse.")
    parser.add_argument("--output", type=Path, help="Annotated MP4 output path.")
    parser.add_argument("--events", type=Path, help="CSV alert log output path.")
    parser.add_argument("--classes", type=parse_classes, default=parse_classes("2,3,5,7"), help="Vehicle class IDs. Use 0 for the project's one-class vehicle model.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker configuration.")
    parser.add_argument("--confidence", type=float, default=0.35, help="Minimum detection confidence.")
    parser.add_argument("--iou", type=float, default=0.50, help="Tracker/detection IoU threshold.")
    parser.add_argument("--motion-window-sec", type=float, default=1.5, help="History window used to decide whether a track is stationary.")
    parser.add_argument("--motion-threshold-px", type=float, default=12.0, help="Maximum center displacement across the motion window.")
    parser.add_argument("--stop-seconds", type=float, default=3.0, help="Seconds stationary inside ROI before STOPPED alert.")
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
    if not 0 < args.confidence <= 1 or not 0 < args.iou <= 1:
        raise ValueError("--confidence and --iou must be in (0, 1]")
    if min(args.motion_window_sec, args.motion_threshold_px, args.stop_seconds, args.max_track_gap_sec) <= 0:
        raise ValueError("Motion and stop thresholds must be positive")
    if args.frame_stride < 1 or args.max_frames < 0:
        raise ValueError("--frame-stride must be >= 1 and --max-frames must be >= 0")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
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

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        parser.error("Cannot read first video frame")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if args.select_roi:
        selected = cv2.selectROI("Draw traffic ROI then press Enter", first_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Draw traffic ROI then press Enter")
        x, y, roi_width, roi_height = (int(value) for value in selected)
        if roi_width <= 0 or roi_height <= 0:
            capture.release()
            parser.error("ROI selection was cancelled or empty")
        roi = x, y, x + roi_width, y + roi_height
    else:
        x1, y1, x2, y2 = args.roi
        roi = round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        parser.error(f"Cannot create annotated output: {output}")

    model = YOLO(str(args.model))
    states: dict[int, TrackState] = {}
    events: list[StopEvent] = []
    decoded_frames = 0
    written_frames = 0
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
            result = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                classes=args.classes,
                conf=args.confidence,
                iou=args.iou,
                verbose=False,
            )[0]
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                coordinates = boxes.xyxy.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                for track_id, (x1, y1, x2, y2), confidence, class_id in zip(ids, coordinates, confidences, class_ids):
                    # The bottom-centre approximates where the vehicle meets the road,
                    # which is more stable for a road/shoulder ROI than box centre.
                    center_x = (x1 + x2) / 2
                    center_y = y2
                    state = states.setdefault(track_id, TrackState())
                    seen_ids.add(track_id)
                    state.last_seen_s = now_s
                    in_roi = point_in_roi(center_x, center_y, roi)
                    is_stopped = False
                    if in_roi:
                        state.positions.append((now_s, center_x, center_y))
                        while state.positions and state.positions[0][0] < now_s - args.motion_window_sec:
                            state.positions.popleft()
                        if len(state.positions) >= 2:
                            _, oldest_x, oldest_y = state.positions[0]
                            displacement = math.hypot(center_x - oldest_x, center_y - oldest_y)
                            if displacement <= args.motion_threshold_px:
                                if state.stopped_since_s is None:
                                    state.stopped_since_s = state.positions[0][0]
                                dwell_s = now_s - state.stopped_since_s
                                if dwell_s >= args.stop_seconds:
                                    is_stopped = True
                                    stopped_now.add(track_id)
                                    if state.active_event is None:
                                        stop_event(state, track_id, now_s, events)
                                    state.active_event.last_seen_at_s = now_s
                            else:
                                close_event(state, now_s, "MOTION_RESUMED")
                    else:
                        close_event(state, now_s, "LEFT_ROI")

                    color = (0, 0, 255) if is_stopped else ((0, 200, 0) if in_roi else (160, 160, 160))
                    dwell_text = "STOPPED" if is_stopped else ("IN_ROI" if in_roi else "OUTSIDE_ROI")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID {track_id} C{class_id} {confidence:.2f} {dwell_text}",
                        (x1, max(22, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            for track_id, state in list(states.items()):
                if track_id not in seen_ids and now_s - state.last_seen_s > args.max_track_gap_sec:
                    close_event(state, state.last_seen_s, "TRACK_LOST")
                    del states[track_id]

            x1, y1, x2, y2 = roi
            roi_color = (0, 0, 255) if stopped_now else (0, 215, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 3)
            status = f"STOPPED ALERTS: {len(stopped_now)} | t={now_s:.1f}s"
            cv2.putText(frame, status, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, roi_color, 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                "Heuristic alert - manual review required",
                (18, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
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

    write_events(events_path, events, args.video, roi)
    print(f"Annotated replay: {output}")
    print(f"Alert log: {events_path}")
    print(f"Frames written: {written_frames}; stopped alerts: {len(events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
