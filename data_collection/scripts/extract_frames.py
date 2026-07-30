"""Tách frame có khoảng cách thời gian từ video APPROVED; không ghi đè."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from online_common import ONLINE_ROOT, read_csv


def main() -> int:
    """Tạo lệnh ffmpeg từ review CSV; metadata ảnh chỉ thêm khi execute thành công."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        print("--fps phải lớn hơn 0", file=sys.stderr)
        return 2
    rows = [r for r in read_csv("planning/video_quality_review.csv") if r.get("final_status") == "APPROVED"]
    if args.video_id:
        rows = [r for r in rows if r.get("video_id") == args.video_id]
    if not rows:
        print("Không có video final_status=APPROVED để tách frame.")
        return 0
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("Không tìm thấy ffmpeg. Hãy cài FFmpeg từ nguồn tin cậy rồi thêm PATH.", file=sys.stderr)
        return 1
    for row in rows:
        source_name = row.get("processed_file_name") or row.get("original_file_name")
        source = ONLINE_ROOT / "processed" / source_name
        target = ONLINE_ROOT / "frames" / row["video_id"]
        command = [ffmpeg, "-i", str(source), "-vf", f"fps={args.fps}", "-q:v", "2", str(target / f"{Path(source_name).stem}_f%06d.jpg")]
        if args.dry_run and not args.execute:
            print("[DRY-RUN] " + " ".join(command))
        else:
            if not source.exists():
                print(f"LỖI thiếu file processed: {source}", file=sys.stderr)
                return 1
            target.mkdir(parents=True, exist_ok=False)
            if subprocess.run(command, check=False).returncode:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
