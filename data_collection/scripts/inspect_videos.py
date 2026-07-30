"""Kiểm tra kỹ thuật video raw/processed và thêm kết quả vào CSV review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from online_common import ONLINE_ROOT, ffprobe, read_csv, sha256, write_csv

FIELDS = ["video_id", "source_id", "original_file_name", "processed_file_name", "file_size_mb", "duration_seconds", "width", "height", "fps", "codec", "container", "frame_count", "rotation", "has_audio", "read_first_frame", "read_middle_frame", "read_last_frame", "checksum_sha256", "camera_type", "lighting_condition", "weather_condition", "vehicle_types", "contains_stopped_vehicle", "contains_moving_vehicle", "contains_negative_segment", "privacy_review", "license_review", "technical_status", "content_status", "final_status", "rejection_reason", "reviewed_by", "evidence_link", "notes"]


def inspect(path: Path) -> dict[str, str]:
    """Đọc metadata ffprobe; nội dung nghiệp vụ luôn cần review thủ công."""
    data = ffprobe(path)
    video = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"), {})
    audio = any(stream.get("codec_type") == "audio" for stream in data.get("streams", []))
    rate = video.get("avg_frame_rate", "0/1")
    numerator, denominator = (rate.split("/") + ["1"])[:2]
    fps = float(numerator) / float(denominator or 1)
    return {"original_file_name": path.name, "file_size_mb": f"{path.stat().st_size / 1048576:.3f}", "duration_seconds": str(data.get("format", {}).get("duration", "")), "width": str(video.get("width", "")), "height": str(video.get("height", "")), "fps": f"{fps:.3f}", "codec": str(video.get("codec_name", "")), "container": str(data.get("format", {}).get("format_name", "")), "frame_count": str(video.get("nb_frames", "")), "rotation": str((video.get("tags") or {}).get("rotate", "0")), "has_audio": str(audio).upper(), "read_first_frame": "NEEDS_MANUAL_REVIEW", "read_middle_frame": "NEEDS_MANUAL_REVIEW", "read_last_frame": "NEEDS_MANUAL_REVIEW", "checksum_sha256": sha256(path), "camera_type": "NEEDS_MANUAL_REVIEW", "lighting_condition": "NEEDS_MANUAL_REVIEW", "weather_condition": "NEEDS_MANUAL_REVIEW", "vehicle_types": "NEEDS_MANUAL_REVIEW", "contains_stopped_vehicle": "NEEDS_MANUAL_REVIEW", "contains_moving_vehicle": "NEEDS_MANUAL_REVIEW", "contains_negative_segment": "NEEDS_MANUAL_REVIEW", "privacy_review": "NEEDS_MANUAL_REVIEW", "license_review": "NEEDS_MANUAL_REVIEW", "technical_status": "NEEDS_MANUAL_REVIEW", "content_status": "NEEDS_MANUAL_REVIEW", "final_status": "NEEDS_MANUAL_REVIEW", "rejection_reason": "", "reviewed_by": "", "evidence_link": "", "notes": "Đọc metadata tự động; nội dung cần review"}


def main() -> int:
    """Inspect tất cả file video được yêu cầu."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ONLINE_ROOT / "raw")
    parser.add_argument("--source-id", default="")
    args = parser.parse_args()
    paths = [path for path in args.input.rglob("*") if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}]
    if not paths:
        print("CHƯA CÓ VIDEO ĐỂ KIỂM TRA.")
        return 0
    rows = read_csv("planning/video_quality_review.csv")
    try:
        for path in paths:
            result = inspect(path)
            result.update({"video_id": f"VID_{path.stem}", "source_id": args.source_id})
            rows.append(result)
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"LỖI inspect: {exc}", file=sys.stderr)
        return 1
    write_csv("planning/video_quality_review.csv", rows, FIELDS)
    print(f"Đã inspect {len(paths)} video.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
