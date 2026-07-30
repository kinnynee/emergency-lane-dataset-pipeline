"""Đối chiếu ảnh và nhãn YOLO theo tên file, rồi xuất báo cáo CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def stems_by_relative_parent(root: Path, suffixes: set[str]) -> dict[tuple[Path, str], Path]:
    """Lập chỉ mục bằng (thư mục tương đối, stem) để hỗ trợ các split."""
    result: dict[tuple[Path, str], Path] = {}
    if not root.exists():
        return result
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            result[(path.parent.relative_to(root), path.stem)] = path
    return result


def main() -> int:
    """Chạy kiểm tra cặp file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "dataset_output/images")
    parser.add_argument("--labels", type=Path, default=ROOT / "dataset_output/labels")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/image_label_pair_report.csv")
    args = parser.parse_args()
    try:
        images = stems_by_relative_parent(args.images, IMAGE_SUFFIXES)
        labels = stems_by_relative_parent(args.labels, {".txt"})
        rows: list[tuple[str, str, str]] = []
        for key in sorted(images.keys() - labels.keys(), key=str):
            rows.append(("MISSING_LABEL", str(images[key]), ""))
        for key in sorted(labels.keys() - images.keys(), key=str):
            rows.append(("MISSING_IMAGE", "", str(labels[key])))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["issue_type", "image_path", "label_path"])
            writer.writerows(rows)
        print(f"Đã ghi {args.output}; số vấn đề: {len(rows)}")
        return 1 if rows else 0
    except (OSError, csv.Error) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

