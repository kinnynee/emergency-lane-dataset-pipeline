"""Tải có giới hạn các URL trực tiếp đã được duyệt; mặc định chỉ dry-run."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

from online_common import ONLINE_ROOT, read_csv, sha256

ALLOWED_PERMISSION = {"APPROVED", "NOT_REQUIRED"}


def eligible(row: dict[str, str]) -> bool:
    """Kiểm tra điều kiện tối thiểu của một hàng queue."""
    return bool(row.get("direct_download_url")) and row.get("download_status") == "APPROVED" and row.get("license_verified") == "TRUE" and row.get("permission_status") in ALLOWED_PERMISSION


def main() -> int:
    """Chạy download an toàn, không dùng cookie/đăng nhập/bypass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-items", type=int, default=1)
    parser.add_argument("--max-total-size-mb", type=float, default=500.0)
    args = parser.parse_args()
    dry_run = args.dry_run and not args.execute
    queue = [row for row in read_csv("planning/video_download_queue.csv") if eligible(row)][: args.max_items]
    if not queue:
        print("Không có hàng APPROVED đủ điều kiện tải.")
        return 0
    target_root = ONLINE_ROOT / "raw"
    used_mb = 0.0
    for row in queue:
        name = row.get("downloaded_file_name") or Path(row["direct_download_url"]).name or f"src_{row['source_id']}_{row['download_id']}.bin"
        target = target_root / name
        if target.exists():
            print(f"SKIPPED tồn tại: {target}")
            continue
        expected = float(row.get("expected_size_mb") or 0)
        if expected and used_mb + expected > args.max_total_size_mb:
            print(f"SKIPPED vượt giới hạn dung lượng: {name}")
            continue
        if dry_run:
            print(f"[DRY-RUN] Sẽ tải trực tiếp: {row['direct_download_url']} -> {target}")
            continue
        partial = target.with_suffix(target.suffix + ".part")
        try:
            request = urllib.request.Request(row["direct_download_url"], headers={"User-Agent": "UMT-K230-Dataset-Research/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            size_mb = partial.stat().st_size / (1024 * 1024)
            if used_mb + size_mb > args.max_total_size_mb:
                raise RuntimeError("vượt --max-total-size-mb; giữ file .part để audit")
            partial.replace(target)
            used_mb += size_mb
            print(f"DOWNLOADED {target.name} sha256={sha256(target)}")
        except (OSError, ValueError, RuntimeError) as exc:
            quarantine = ONLINE_ROOT / "quarantine" / partial.name
            if partial.exists():
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                partial.replace(quarantine)
                print(f"Đã chuyển file tải lỗi vào quarantine: {quarantine}", file=sys.stderr)
            print(f"FAILED {name}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
