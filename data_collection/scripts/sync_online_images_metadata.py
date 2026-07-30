"""Đồng bộ ảnh Commons vào metadata/images.csv mà không gán nhãn nội dung giả."""

from __future__ import annotations

import csv
from pathlib import Path

from online_common import ROOT

FIELDS = [
    "image_id", "video_id", "source_id", "session_id", "file_name",
    "frame_index", "timestamp_seconds", "width", "height", "condition",
    "source_type", "camera_type", "contains_vehicle", "vehicle_count", "split",
    "duplicate_status", "quality_status", "storage_path", "notes",
]


def main() -> int:
    """Thêm các ảnh chưa có vào metadata chính với trạng thái chưa review."""
    source_path = ROOT / "planning/online_image_metadata.csv"
    target_path = ROOT / "metadata/images.csv"
    with source_path.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    with target_path.open(encoding="utf-8-sig", newline="") as handle:
        target_rows = list(csv.DictReader(handle))
    known = {row.get("image_id", "") for row in target_rows}
    added = 0
    for row in source_rows:
        if row["image_id"] in known:
            continue
        target_rows.append({
            "image_id": row["image_id"],
            "video_id": "",
            "source_id": "SRC_ONL007",
            "session_id": "",
            "file_name": row["file_name"],
            "frame_index": "",
            "timestamp_seconds": "",
            "width": row["width"],
            "height": row["height"],
            "condition": row["condition"],
            "source_type": "OPEN_DATASET",
            "camera_type": "UNKNOWN",
            "contains_vehicle": "UNVERIFIED",
            "vehicle_count": "",
            "split": "UNASSIGNED",
            "duplicate_status": "NOT_CHECKED",
            "quality_status": "NEEDS_MANUAL_REVIEW",
            "storage_path": row["storage_path"],
            "notes": "Ảnh demo từ Wikimedia Commons; condition dựa trên truy vấn, chưa phải ground truth.",
        })
        known.add(row["image_id"])
        added += 1
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(target_rows)
    print(f"Đã thêm {added} ảnh vào metadata/images.csv; tổng {len(target_rows)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
