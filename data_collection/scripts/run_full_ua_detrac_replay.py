"""Replay every available UA-DETRAC XML sequence with reproducible evidence.

UA-DETRAC track annotations are replayed through the Host's stopped-vehicle
state machine, never treated as K230 evidence.  The command writes the model
hash (provenance only), repository commit, locked replay configuration,
per-sequence results, and an aggregate summary.  Approved stop-event labels
are still required before FAR, misses, or latency become measured metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from host_yolo_loop import GroundPlaneTransform
from replay_ua_detrac_alerts import (
    _expected_events,
    _metrics,
    _parse_frame_size,
    _parse_roi,
    _replay,
    _roi_pixels,
    _tracks_from_xml,
    _write_alerts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "NOT_AVAILABLE"
    return result.stdout.strip() if result.returncode == 0 else "NOT_AVAILABLE"


def discover_xmls(annotation_roots: list[Path]) -> list[Path]:
    """Return the complete, unique XML set; empty inputs are a hard failure."""
    paths: dict[Path, Path] = {}
    for root in annotation_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"UA-DETRAC annotation root not found: {root}")
        for path in root.rglob("*.xml"):
            paths[path.resolve()] = path.resolve()
    result = sorted(paths.values(), key=lambda path: (path.name.lower(), path.as_posix().lower()))
    if not result:
        raise ValueError("No UA-DETRAC XML files were found")
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_full_replay(
    annotation_roots: list[Path],
    model: Path,
    calibration_path: Path,
    frame_size: tuple[int, int],
    fake_roi: tuple[float, float, float, float],
    output: Path,
    *,
    fps: float,
    stop_speed_kmh: float,
    stop_seconds: float,
    resume_speed_kmh: float,
    resume_seconds: float,
    speed_window_s: float,
    max_track_gap_s: float,
    expected_events_path: Path | None,
) -> dict[str, Any]:
    """Run every XML and persist only newly generated replay evidence."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite replay output: {output}")
    if not model.is_file():
        raise FileNotFoundError(f"Model required for replay provenance was not found: {model}")
    xml_paths = discover_xmls(annotation_roots)
    calibration = GroundPlaneTransform.from_file(calibration_path)
    calibration.validate_frame_size(*frame_size)
    roi = _roi_pixels(fake_roi, frame_size)
    output.mkdir(parents=True)
    config = {
        "format": "ua-detrac-full-replay-v1",
        "repo_commit": _git_commit(),
        "model": {"path": str(model.resolve()), "sha256": _sha256(model), "role": "PROVENANCE_ONLY_GT_TRACK_REPLAY"},
        "annotation_roots": [str(path.resolve()) for path in annotation_roots],
        "annotation_xml_count": len(xml_paths),
        "speed_calibration": str(calibration_path.resolve()),
        "frame_size": list(frame_size),
        "fake_roi": list(fake_roi),
        "fps": fps,
        "stop_speed_kmh": stop_speed_kmh,
        "stop_seconds": stop_seconds,
        "resume_speed_kmh": resume_speed_kmh,
        "resume_seconds": resume_seconds,
        "speed_window_sec": speed_window_s,
        "max_track_gap_sec": max_track_gap_s,
        "expected_events": None if expected_events_path is None else str(expected_events_path.resolve()),
    }
    _write_json(output / "replay_config.json", config)
    results: list[dict[str, Any]] = []
    seen_sequences: set[str] = set()
    for xml_path in xml_paths:
        sequence_id, frames = _tracks_from_xml(xml_path)
        if sequence_id in seen_sequences:
            raise ValueError(f"Duplicate sequence ID across annotation roots: {sequence_id}")
        seen_sequences.add(sequence_id)
        events, _ = _replay(
            frames,
            fps,
            roi,
            calibration,
            stop_speed_kmh,
            stop_seconds,
            resume_speed_kmh,
            resume_seconds,
            speed_window_s,
            max_track_gap_s,
        )
        duration_s = (max(frames) - min(frames)) / fps
        expected = _expected_events(expected_events_path, sequence_id)
        metrics = _metrics(events, expected, duration_s)
        alerts = output / "sequence_alerts" / f"{sequence_id}.csv"
        _write_alerts(alerts, sequence_id, events, roi)
        results.append(
            {
                "sequence_id": sequence_id,
                "xml": str(xml_path),
                "alerts": str(alerts),
                "frame_count": len(frames),
                "approved_expected_events": len(expected),
                **metrics,
            }
        )
    fields = ["sequence_id", "xml", "alerts", "frame_count", "approved_expected_events", "duration_hours", "alert_count", "false_alerts_per_hour", "missed_stopped_vehicles", "detection_to_alert_latency_s", "status"]
    with (output / "sequence_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    status_counts = Counter(str(result["status"]) for result in results)
    summary = {
        "format": "ua-detrac-full-replay-summary-v1",
        "replay_config": "replay_config.json",
        "sequence_results": "sequence_results.csv",
        "sequence_count": len(results),
        "total_alert_count": sum(int(result["alert_count"]) for result in results),
        "metric_status_counts": dict(sorted(status_counts.items())),
        "status": "COMPLETE" if all(status == "MEASURED_AGAINST_APPROVED_UA_REPLAY_EVENTS" for status in status_counts) else "COMPLETE_METRICS_NOT_MEASURED_WITHOUT_APPROVED_GROUND_TRUTH",
    }
    _write_json(output / "replay_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ua-annotation-root", required=True, type=Path, action="append")
    parser.add_argument("--model", required=True, type=Path, help="Model hash is recorded as replay provenance only.")
    parser.add_argument("--speed-calibration", required=True, type=Path)
    parser.add_argument("--frame-size", required=True, type=_parse_frame_size)
    parser.add_argument("--fake-roi", required=True, type=_parse_roi)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--stop-speed-kmh", type=float, default=2.0)
    parser.add_argument("--stop-seconds", type=float, default=3.0)
    parser.add_argument("--resume-speed-kmh", type=float, default=3.0)
    parser.add_argument("--resume-seconds", type=float, default=1.0)
    parser.add_argument("--speed-window-sec", type=float, default=1.5)
    parser.add_argument("--max-track-gap-sec", type=float, default=1.0)
    parser.add_argument("--expected-events", type=Path)
    args = parser.parse_args()
    values = (args.fps, args.stop_speed_kmh, args.stop_seconds, args.resume_speed_kmh, args.resume_seconds, args.speed_window_sec, args.max_track_gap_sec)
    if min(values) <= 0:
        parser.error("fps and every replay threshold must be positive")
    if args.resume_speed_kmh <= args.stop_speed_kmh:
        parser.error("--resume-speed-kmh must be greater than --stop-speed-kmh")
    summary = run_full_replay(
        [path.resolve() for path in args.ua_annotation_root],
        args.model.resolve(),
        args.speed_calibration.resolve(),
        args.frame_size,
        args.fake_roi,
        args.output.resolve(),
        fps=args.fps,
        stop_speed_kmh=args.stop_speed_kmh,
        stop_seconds=args.stop_seconds,
        resume_speed_kmh=args.resume_speed_kmh,
        resume_seconds=args.resume_seconds,
        speed_window_s=args.speed_window_sec,
        max_track_gap_s=args.max_track_gap_sec,
        expected_events_path=None if args.expected_events is None else args.expected_events.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
