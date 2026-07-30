"""Nhóm video có checksum giống nhau; không xóa hoặc tự từ chối file."""

from __future__ import annotations

from collections import defaultdict

from online_common import ROOT, read_csv


def main() -> int:
    """Tạo duplicate_video_report.md từ video_quality_review.csv."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv("planning/video_quality_review.csv"):
        if row.get("checksum_sha256"):
            groups[row["checksum_sha256"]].append(row)
    duplicates = [items for items in groups.values() if len(items) > 1]
    lines = ["# Báo cáo video trùng", "", "Không có file nào bị xóa; người dùng phải duyệt nhóm trùng."]
    if not duplicates:
        lines += ["", "CHƯA PHÁT HIỆN VIDEO TRÙNG TỪ METADATA HIỆN CÓ."]
    for index, items in enumerate(duplicates, 1):
        lines += ["", f"## Nhóm {index}"] + [f"- {item.get('video_id')} - {item.get('original_file_name')}" for item in items]
    (ROOT / "reports/duplicate_video_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã ghi duplicate report; nhóm trùng: {len(duplicates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
