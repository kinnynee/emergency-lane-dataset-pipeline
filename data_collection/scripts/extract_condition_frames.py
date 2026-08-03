"""Extract review-approved video frames into DAY/NIGHT/RAIN/BACKLIT folders.

This tool categorizes only from reviewed manifest fields; it never infers night
or backlight from pixel brightness and never synthesizes rain.  A video may be
placed in more than one condition folder (for example, NIGHT and RAIN), with a
manifest preserving the original video/session identifiers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


TARGETS = ("DAY", "NIGHT", "RAIN", "BACKLIT")
APPROVED_STATUSES = {"APPROVED", "DONE", "PROCESSED"}
BACKLIT_APPROVAL = {"APPROVED", "REVIEWED", "DATA_LEAD_APPROVED"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FRAME_FIELDS = [
    "frame_id", "condition", "video_id", "session_id", "source_id", "source_video",
    "frame_index", "timestamp_ms", "width", "height", "storage_path", "review_basis",
]


def _value(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def _status(row: dict[str, str]) -> str:
    return _value(row, "final_status", "status").upper()


def _conditions(row: dict[str, str]) -> tuple[list[str], str]:
    """Return condition folders and the review fields that justify them."""
    lighting = _value(row, "lighting_condition", "lighting", "condition").upper().replace("BACKLIGHT", "BACKLIT")
    weather = _value(row, "weather_condition", "weather", "condition").upper()
    found: list[str] = []
    bases: list[str] = []
    if lighting == "DAY":
        found.append("DAY")
        bases.append("lighting_condition=DAY")
    if lighting == "NIGHT":
        found.append("NIGHT")
        bases.append("lighting_condition=NIGHT")
    if weather in {"RAIN", "RAIN_OR_WET_ROAD", "RAIN_OR_SNOW"}:
        found.append("RAIN")
        bases.append(f"weather_condition={weather}")
    if lighting == "BACKLIT":
        review = _value(row, "backlit_review_status", "backlit_review", "lighting_review_status").upper()
        if review not in BACKLIT_APPROVAL:
            raise ValueError("BACKLIT requires backlit_review_status=APPROVED/REVIEWED; bright DAY footage is not a substitute")
        found.append("BACKLIT")
        bases.append(f"lighting_condition=BACKLIT;backlit_review_status={review}")
    return found, ";".join(bases)


def _source_path(row: dict[str, str], input_root: Path) -> Path:
    raw = _value(row, "processed_path", "storage_path", "processed_file_name", "file_name", "original_file_name")
    if not raw:
        raise ValueError("Video row has no source path or file name")
    path = Path(raw)
    return path if path.is_absolute() else input_root / path


def _eligible_rows(manifest: Path, input_root: Path) -> list[tuple[dict[str, str], Path, list[str], str]]:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible: list[tuple[dict[str, str], Path, list[str], str]] = []
    for line_number, row in enumerate(rows, start=2):
        video_id = _value(row, "video_id")
        if not video_id:
            raise ValueError(f"{manifest}:{line_number} is missing video_id")
        if _status(row) not in APPROVED_STATUSES:
            continue
        source = _source_path(row, input_root)
        if source.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"{manifest}:{line_number} has unsupported video type: {source.name}")
        conditions, basis = _conditions(row)
        if conditions:
            eligible.append((row, source, conditions, basis))
    return eligible


def extract_frames(manifest: Path, input_root: Path, output: Path, fps: float, execute: bool) -> dict[str, Any]:
    """Extract one non-overlapping output dataset and return its summary."""
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    rows = _eligible_rows(manifest, input_root)
    planned = {
        "eligible_videos": len(rows),
        "conditions_by_video": {str(_value(row, "video_id")): conditions for row, _source, conditions, _basis in rows},
        "fps": fps,
    }
    if not execute:
        return {"status": "DRY_RUN", **planned}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}. Use a new extraction version; no files were changed.")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required; install requirements-data.txt before extraction") from exc
    output.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    by_condition: Counter[str] = Counter()
    for row, source, conditions, basis in rows:
        if not source.is_file():
            raise FileNotFoundError(f"Approved video is missing: {source}")
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {source}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            capture.release()
            raise RuntimeError(f"Video has no readable FPS: {source}")
        interval = max(1, round(source_fps / fps))
        video_id = _value(row, "video_id")
        session_id = _value(row, "session_id")
        source_id = _value(row, "source_id")
        frame_index = 0
        saved_index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % interval != 0:
                frame_index += 1
                continue
            saved_index += 1
            timestamp_ms = round(frame_index * 1000.0 / source_fps)
            height, width = frame.shape[:2]
            stem = f"{video_id}_f{saved_index:06d}_t{timestamp_ms:09d}"
            for condition in conditions:
                image_path = output / condition.lower() / video_id / f"{stem}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if image_path.exists():
                    capture.release()
                    raise FileExistsError(f"Refusing to overwrite frame: {image_path}")
                if not cv2.imwrite(str(image_path), frame):
                    capture.release()
                    raise RuntimeError(f"Failed to write frame: {image_path}")
                frame_rows.append({
                    "frame_id": f"{condition}_{stem}", "condition": condition, "video_id": video_id,
                    "session_id": session_id, "source_id": source_id, "source_video": str(source),
                    "frame_index": frame_index, "timestamp_ms": timestamp_ms, "width": width, "height": height,
                    "storage_path": image_path.relative_to(output).as_posix(), "review_basis": basis,
                })
                by_condition[condition] += 1
            frame_index += 1
        capture.release()
    with (output / "frame_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(frame_rows)
    summary = {"status": "COMPLETE", **planned, "frames_by_condition": dict(sorted(by_condition.items())), "frame_records": len(frame_rows)}
    (output / "extraction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with approved video metadata and review fields")
    parser.add_argument("--input-root", type=Path, default=Path("."), help="Base path for relative video paths")
    parser.add_argument("--output", type=Path, required=True, help="New output directory for condition-organized frames")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true", help="Perform extraction; omit for a non-writing dry run")
    args = parser.parse_args()
    try:
        summary = extract_frames(args.manifest.resolve(), args.input_root.resolve(), args.output.resolve(), args.fps, args.execute)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
