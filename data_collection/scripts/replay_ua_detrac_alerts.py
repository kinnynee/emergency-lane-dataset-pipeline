"""Replay UA-DETRAC tracks through the same stopped-vehicle alert rule.

UA-DETRAC has track IDs and bounding boxes but no approved ground truth for a
stopped vehicle in an emergency lane.  This tool therefore reuses the runtime
state machine for a deliberately marked fake ROI, creates an optional overlay
video, and reports alert metrics as NOT_MEASURED until expected stop events are
labelled and approved.  It must never be used as K230 deployment evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from host_yolo_loop import (
    GroundPlaneTransform,
    StopEvent,
    TrackState,
    close_event,
    point_in_roi,
    update_stop_alert,
    update_speed_kmh,
)


EXPECTED_FIELDS = {
    "event_id", "sequence_id", "track_id", "expected_start_s", "expected_end_s", "approval_status"
}


def _parse_frame_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part.strip()) for part in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--frame-size must be WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("--frame-size must contain positive dimensions")
    return width, height


def _parse_roi(value: str) -> tuple[float, float, float, float]:
    try:
        x1, y1, x2, y2 = (float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--fake-roi must be x1,y1,x2,y2 in normalized coordinates") from exc
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise argparse.ArgumentTypeError("--fake-roi must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1")
    return x1, y1, x2, y2


def _roi_pixels(roi: tuple[float, float, float, float], size: tuple[int, int]) -> tuple[tuple[float, float], ...]:
    x1, y1, x2, y2 = roi
    width, height = size
    return ((x1 * width, y1 * height), (x2 * width, y1 * height), (x2 * width, y2 * height), (x1 * width, y2 * height))


def _tracks_from_xml(path: Path) -> tuple[str, dict[int, list[tuple[int, tuple[float, float, float, float]]]]]:
    root = ET.parse(path).getroot()
    sequence_id = root.attrib.get("name") or path.stem
    frames: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    for frame in root.findall("frame"):
        frame_id = int(frame.attrib.get("num", "0"))
        for target in frame.findall("./target_list/target"):
            box = target.find("box")
            if box is None:
                continue
            try:
                left = float(box.attrib["left"])
                top = float(box.attrib["top"])
                width = float(box.attrib["width"])
                height = float(box.attrib["height"])
                track_id = int(target.attrib["id"])
            except (KeyError, ValueError):
                continue
            if width > 0 and height > 0:
                frames[frame_id].append((track_id, (left, top, left + width, top + height)))
    if not frames:
        raise ValueError(f"No valid UA-DETRAC tracks in {path}")
    return sequence_id, dict(frames)


def _expected_events(path: Path | None, sequence_id: str) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not EXPECTED_FIELDS.issubset(set(rows[0])):
        raise ValueError("Expected-event CSV must contain " + ", ".join(sorted(EXPECTED_FIELDS)))
    return [row for row in rows if row.get("sequence_id") == sequence_id and row.get("approval_status") == "APPROVED"]


def _replay(
    frames: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    fps: float,
    roi: tuple[tuple[float, float], ...],
    calibration: GroundPlaneTransform,
    stop_speed_kmh: float,
    stop_seconds: float,
    resume_speed_kmh: float,
    resume_seconds: float,
    speed_window_s: float,
    max_track_gap_s: float,
) -> tuple[list[StopEvent], dict[int, list[tuple[int, tuple[float, float, float, float]]]]]:
    states: dict[int, TrackState] = {}
    events: list[StopEvent] = []
    last_frame = max(frames)
    for frame_id in range(min(frames), last_frame + 1):
        now_s = (frame_id - 1) / fps
        seen: set[int] = set()
        for track_id, (x1, y1, x2, y2) in frames.get(frame_id, []):
            seen.add(track_id)
            state = states.setdefault(track_id, TrackState())
            state.last_seen_s = now_s
            foot = ((x1 + x2) / 2.0, y2)
            if not point_in_roi(*foot, roi):
                close_event(state, now_s, "LEFT_ROI")
                continue
            ground_point = calibration.project(foot)
            speed_kmh = update_speed_kmh(state, now_s, ground_point, speed_window_s)
            update_stop_alert(
                state, track_id, now_s, speed_kmh, calibration.range_to_camera_m(foot), events,
                stop_speed_kmh=stop_speed_kmh,
                stop_seconds=stop_seconds,
                resume_speed_kmh=resume_speed_kmh,
                resume_seconds=resume_seconds,
            )
        for track_id, state in list(states.items()):
            if track_id not in seen and now_s - state.last_seen_s > max_track_gap_s:
                close_event(state, state.last_seen_s, "TRACK_LOST", clear_history=True)
                del states[track_id]
    end_s = (last_frame - 1) / fps
    for state in states.values():
        close_event(state, end_s, "END_OF_REPLAY")
    return events, frames


def _write_alerts(path: Path, sequence_id: str, events: Iterable[StopEvent], roi: tuple[tuple[float, float], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["event_id", "sequence_id", "track_id", "fake_roi_px_polygon", "stopped_started_at_s", "alerted_at_s", "ended_at_s", "end_reason", "review_status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            writer.writerow({
                "event_id": event.event_id, "sequence_id": sequence_id, "track_id": event.track_id,
                "fake_roi_px_polygon": ";".join(f"{round(x)},{round(y)}" for x, y in roi),
                "stopped_started_at_s": f"{event.started_at_s:.3f}", "alerted_at_s": f"{event.alerted_at_s:.3f}",
                "ended_at_s": "" if event.ended_at_s is None else f"{event.ended_at_s:.3f}",
                "end_reason": event.end_reason, "review_status": "NEEDS_MANUAL_REVIEW",
            })


def _metrics(events: list[StopEvent], expected: list[dict[str, str]], duration_s: float) -> dict[str, Any]:
    base = {
        "duration_hours": duration_s / 3600.0,
        "alert_count": len(events),
        "false_alerts_per_hour": "NOT_MEASURED_MISSING_APPROVED_STOP_GROUND_TRUTH",
        "missed_stopped_vehicles": "NOT_MEASURED_MISSING_APPROVED_STOP_GROUND_TRUTH",
        "detection_to_alert_latency_s": "NOT_MEASURED_MISSING_APPROVED_STOP_GROUND_TRUTH",
        "status": "NOT_MEASURED_MISSING_APPROVED_STOP_GROUND_TRUTH",
    }
    if not expected:
        return base
    unmatched_alerts = set(range(len(events)))
    unmatched_expected = 0
    latencies: list[float] = []
    for target in expected:
        track_id = int(target["track_id"])
        start = float(target["expected_start_s"])
        end = float(target["expected_end_s"])
        candidates = [index for index in unmatched_alerts if events[index].track_id == track_id and start <= events[index].alerted_at_s <= end]
        if not candidates:
            unmatched_expected += 1
            continue
        best = min(candidates, key=lambda index: events[index].alerted_at_s)
        unmatched_alerts.remove(best)
        latencies.append(events[best].alerted_at_s - start)
    return {
        "duration_hours": duration_s / 3600.0,
        "alert_count": len(events),
        "false_alerts_per_hour": len(unmatched_alerts) / (duration_s / 3600.0) if duration_s > 0 else "NOT_MEASURABLE_ZERO_DURATION",
        "missed_stopped_vehicles": unmatched_expected,
        "detection_to_alert_latency_s": sum(latencies) / len(latencies) if latencies else "NOT_MEASURED_NO_MATCHED_APPROVED_EVENTS",
        "status": "MEASURED_AGAINST_APPROVED_UA_REPLAY_EVENTS",
    }


def _render_video(
    images_dir: Path, frames: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    roi: tuple[tuple[float, float], ...], fps: float, output: Path,
) -> None:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for --annotated-video") from exc
    candidates = sorted(images_dir.glob("img*.*"))
    if not candidates:
        raise FileNotFoundError(f"No UA-DETRAC img* files in {images_dir}")
    first = cv2.imread(str(candidates[0]))
    if first is None:
        raise ValueError(f"Cannot read {candidates[0]}")
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video {output}")
    try:
        for frame_path in candidates:
            frame_id = int(frame_path.stem.lower().replace("img", ""))
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            for track_id, (x1, y1, x2, y2) in frames.get(frame_id, []):
                cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), (0, 200, 0), 2)
                cv2.putText(frame, f"GT ID {track_id}", (round(x1), max(18, round(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.polylines(frame, [np.array(roi, dtype=np.int32)], True, (0, 215, 255), 2)
            cv2.putText(frame, "FAKE ROI - UA REPLAY ONLY", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)
            writer.write(frame)
    finally:
        writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ua-xml", required=True, type=Path)
    parser.add_argument("--speed-calibration", required=True, type=Path)
    parser.add_argument("--frame-size", required=True, type=_parse_frame_size)
    parser.add_argument("--fake-roi", required=True, type=_parse_roi)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--stop-speed-kmh", type=float, default=2.0)
    parser.add_argument("--stop-seconds", type=float, default=3.0)
    parser.add_argument("--resume-speed-kmh", type=float, default=3.0)
    parser.add_argument("--resume-seconds", type=float, default=1.0)
    parser.add_argument("--speed-window-sec", type=float, default=1.5)
    parser.add_argument("--max-track-gap-sec", type=float, default=1.0)
    parser.add_argument("--expected-events", type=Path)
    parser.add_argument("--alerts", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--annotated-video", type=Path)
    args = parser.parse_args()
    if args.fps <= 0 or min(args.stop_speed_kmh, args.stop_seconds, args.resume_speed_kmh, args.resume_seconds, args.speed_window_sec, args.max_track_gap_sec) <= 0:
        parser.error("fps and all alert thresholds must be positive")
    if args.resume_speed_kmh <= args.stop_speed_kmh:
        parser.error("--resume-speed-kmh must be greater than --stop-speed-kmh")
    if (args.images_dir is None) != (args.annotated_video is None):
        parser.error("--images-dir and --annotated-video must be used together")
    sequence_id, frames = _tracks_from_xml(args.ua_xml)
    calibration = GroundPlaneTransform.from_file(args.speed_calibration)
    calibration.validate_frame_size(*args.frame_size)
    roi = _roi_pixels(args.fake_roi, args.frame_size)
    events, replay_frames = _replay(
        frames, args.fps, roi, calibration,
        args.stop_speed_kmh, args.stop_seconds, args.resume_speed_kmh, args.resume_seconds,
        args.speed_window_sec, args.max_track_gap_sec,
    )
    expected = _expected_events(args.expected_events, sequence_id)
    duration_s = (max(frames) - min(frames)) / args.fps
    metrics = {
        "sequence_id": sequence_id,
        "replay_source": "UA_DETRAC_GT_TRACK_IDS",
        "roi_status": "REPLAY_FAKE_ROI_NOT_DEPLOYMENT_CALIBRATION",
        "rule_logic": "host_yolo_loop.TrackState/update_speed_kmh(median)/update_stop_alert(hysteresis)",
        **_metrics(events, expected, duration_s),
    }
    _write_alerts(args.alerts, sequence_id, events, roi)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.images_dir and args.annotated_video:
        _render_video(args.images_dir, replay_frames, roi, args.fps, args.annotated_video)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
