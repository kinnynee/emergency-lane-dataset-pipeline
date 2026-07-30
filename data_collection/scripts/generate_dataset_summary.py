"""Sinh báo cáo Markdown từ metadata ảnh và nhãn, không tạo số liệu giả."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    """Đọc CSV UTF-8."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bullets(counter: Counter[str]) -> list[str]:
    """Định dạng bộ đếm thành bullet."""
    return [f"- `{key or 'CHƯA GHI'}`: {value}" for key, value in sorted(counter.items())]


def main() -> int:
    """Tạo reports/dataset_summary.md."""
    try:
        images = read_rows(ROOT / "metadata/images.csv")
        labels = read_rows(ROOT / "metadata/labels.csv")
        output = ROOT / "reports/dataset_summary.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Tóm tắt dataset", "", "## Trạng thái"]
        if not images:
            lines += ["", "CHƯA CÓ DỮ LIỆU ẢNH TRONG METADATA. Không có thống kê thực tế để báo cáo."]
        else:
            conditions = Counter(row.get("condition", "") for row in images)
            splits = Counter(row.get("split", "") for row in images)
            vehicle = Counter(row.get("contains_vehicle", "") for row in images)
            lines += [
                "",
                f"- Tổng ảnh: {len(images)}",
                f"- Tổng bounding box theo metadata nhãn: {sum(int(row.get('bbox_count') or 0) for row in labels)}",
                "",
                "## Theo condition",
                *bullets(conditions),
                "",
                "## Theo split",
                *bullets(splits),
                "",
                "## Có xe/không xe",
                *bullets(vehicle),
            ]
        if not labels:
            lines += ["", "## Nhãn", "", "CHƯA CÓ DỮ LIỆU NHÃN TRONG METADATA."]
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Đã ghi báo cáo: {output}")
        return 0
    except (OSError, csv.Error, ValueError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

