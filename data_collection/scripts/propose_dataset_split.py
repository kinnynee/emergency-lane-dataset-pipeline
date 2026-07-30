"""Đề xuất split theo video_id, không ghi đè nếu thiếu --apply."""

from __future__ import annotations

import argparse
import hashlib

from online_common import read_csv


def split_for(video_id: str) -> str:
    """Chia xác định theo video để mọi frame cùng video cùng split."""
    bucket = int(hashlib.sha256(video_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def main() -> int:
    """In đề xuất split để reviewer quyết định trước khi áp dụng."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Chưa được triển khai tự ghi để tránh ghi đè metadata.")
    args = parser.parse_args()
    groups = sorted({row.get("video_id", "") for row in read_csv("metadata/images.csv") if row.get("video_id")})
    print("ĐỀ XUẤT SPLIT (theo video_id):")
    for video_id in groups:
        print(f"{video_id},{split_for(video_id)}")
    if args.apply:
        print("--apply được nhận nhưng chưa tự ghi: cần reviewer khóa manifest trước khi thay đổi metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
