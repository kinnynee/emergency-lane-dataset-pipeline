"""Điều phối pipeline online; mặc định dry-run và dừng trước tải nếu license lỗi."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from online_common import ROOT


def run(script: str, args: list[str] | None = None, required: bool = True) -> int:
    """Chạy script cùng thư mục và in lệnh để tạo audit trail."""
    command = [sys.executable, str(ROOT / "scripts" / script), *(args or [])]
    print("$ " + " ".join(command))
    result = subprocess.run(command, check=False)
    if required and result.returncode:
        print(f"DỪNG PIPELINE: {script} trả {result.returncode}", file=sys.stderr)
    return result.returncode


def main() -> int:
    """Chạy các bước an toàn; execute không vượt qua gate license."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-items", type=int, default=1)
    parser.add_argument("--max-total-size-mb", type=float, default=500)
    parser.add_argument("--source-id")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--skip-frame-extraction", action="store_true")
    parser.add_argument("--generate-report-only", action="store_true")
    args = parser.parse_args()
    required_files = ["configs/online_source_search.yaml", "configs/license_rules.yaml", "planning/online_source_candidates.csv", "planning/video_download_queue.csv"]
    missing = [relative for relative in required_files if not (ROOT / relative).exists()]
    if missing:
        print("Thiếu cấu hình: " + ", ".join(missing), file=sys.stderr)
        return 2
    if args.generate_report_only:
        return run("generate_online_data_report.py")
    if run("validate_source_license.py"):
        return 1
    if not args.skip_download:
        command = ["--max-items", str(args.max_items), "--max-total-size-mb", str(args.max_total_size_mb)]
        command += ["--execute"] if args.execute else ["--dry-run"]
        if run("download_approved_sources.py", command):
            return 1
    if not args.skip_normalize:
        command = ["--execute"] if args.execute else ["--dry-run"]
        if run("normalize_videos.py", command, required=False):
            print("Bỏ qua normalize: không có video hoặc thiếu ffmpeg.")
    if not args.skip_frame_extraction:
        command = ["--execute"] if args.execute else ["--dry-run"]
        if run("extract_frames.py", command, required=False):
            print("Bỏ qua extract: không có video APPROVED hoặc thiếu ffmpeg.")
    run("detect_duplicate_videos.py", required=False)
    run("detect_near_duplicate_frames.py", ["--dry-run"], required=False)
    run("generate_contact_sheet.py", required=False)
    return run("generate_online_data_report.py")


if __name__ == "__main__":
    raise SystemExit(main())
