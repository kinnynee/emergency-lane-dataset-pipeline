"""Chuẩn hóa video đã APPROVED sang MP4 H.264 mà không sửa file gốc."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from online_common import ONLINE_ROOT, read_csv


def main() -> int:
    """Chạy FFmpeg khi có execute và video đủ trạng thái duyệt bên ngoài."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ONLINE_ROOT / "raw")
    parser.add_argument("--output", type=Path, default=ONLINE_ROOT / "processed")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    sources = {row.get("source_id"): row for row in read_csv("planning/online_source_candidates.csv")}
    approved = {row.get("original_file_name"): row for row in read_csv("planning/video_quality_review.csv") if row.get("final_status") == "APPROVED"}
    videos = []
    for path in args.input.rglob("*"):
        row = approved.get(path.name)
        source = sources.get((row or {}).get("source_id"), {})
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"} and row and source.get("review_status") in {"APPROVED_FOR_DOWNLOAD", "DOWNLOADED", "PROCESSED"}:
            videos.append(path)
    if not videos:
        print("Không có video raw vừa APPROVED vừa có source được duyệt để chuẩn hóa.")
        return 0
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("Không tìm thấy ffmpeg. Hãy cài FFmpeg từ nguồn tin cậy rồi thêm PATH.", file=sys.stderr)
        return 1
    for source in videos:
        target = args.output / f"{source.stem}.mp4"
        if target.exists():
            print(f"SKIPPED tồn tại: {target}")
            continue
        command = [ffmpeg, "-i", str(source), "-map", "0:v:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(target)]
        if args.dry_run and not args.execute:
            print("[DRY-RUN] " + " ".join(command))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(command, check=False, timeout=3600)
            if result.returncode:
                print(f"LỖI chuẩn hóa {source}", file=sys.stderr)
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
