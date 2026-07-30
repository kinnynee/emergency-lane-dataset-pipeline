"""Tạo cấu trúc YOLO train/val/test mà không xóa dữ liệu hiện có."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    """Đọc các trường YAML cấp cao cần cho script bằng thư viện chuẩn."""
    try:
        with path.open(encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError as exc:
        raise RuntimeError(f"Không đọc được cấu hình {path}: {exc}") from exc
    data: dict[str, Any] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if raw_line[:1].isspace():
            continue
        cleaned = value.strip().strip("\"'")
        data[key.strip()] = cleaned
    if "project_name" not in data:
        raise ValueError("Cấu hình thiếu trường project_name.")
    return data


def planned_directories(output_root: Path) -> list[Path]:
    """Trả về các thư mục dataset cần có."""
    return [
        output_root / kind / split
        for kind in ("images", "labels")
        for split in ("train", "val", "test")
    ]


def main() -> int:
    """Chạy CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dataset_config.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        print(f"Project: {config.get('project_name', 'CHƯA CẤU HÌNH')}")
        for directory in planned_directories(args.output):
            if args.dry_run:
                print(f"[DRY-RUN] Sẽ tạo nếu chưa có: {directory}")
            else:
                existed = directory.exists()
                directory.mkdir(parents=True, exist_ok=True)
                print(f"[{'ĐÃ CÓ' if existed else 'ĐÃ TẠO'}] {directory}")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
