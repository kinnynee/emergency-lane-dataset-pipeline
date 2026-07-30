"""Kiểm tra schema, ID và trạng thái trong các CSV quản lý dữ liệu."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALID_STATUSES = {"NOT_STARTED", "IN_PROGRESS", "BLOCKED", "REVIEW", "DONE", "CANCELLED"}
SCHEMAS: dict[str, tuple[str, list[str], str | None]] = {
    "metadata/sessions.csv": ("session_id", ["session_id", "date", "location_id"], None),
    "metadata/videos.csv": ("video_id", ["video_id", "session_id", "file_name", "status"], "status"),
    "metadata/images.csv": ("image_id", ["image_id", "video_id", "session_id", "file_name", "split"], None),
    "metadata/labels.csv": ("label_id", ["label_id", "image_id", "file_name", "annotation_status"], "annotation_status"),
    "planning/data_collection_master_plan.csv": ("task_id", ["task_id", "task_name", "owner", "status"], "status"),
    "planning/labeling_progress.csv": ("batch_id", ["date", "dataset_version", "batch_id", "status"], "status"),
    "planning/quality_issue_log.csv": ("issue_id", ["issue_id", "date_found", "status"], "status"),
}


def validate_file(path: Path, id_column: str, required: list[str], status_column: str | None) -> list[str]:
    """Kiểm tra một CSV và trả danh sách lỗi."""
    errors: list[str] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            missing = [column for column in required if column not in headers]
            if missing:
                return [f"{path}: thiếu cột {', '.join(missing)}"]
            seen: set[str] = set()
            for line, row in enumerate(reader, start=2):
                item_id = (row.get(id_column) or "").strip()
                if not item_id:
                    errors.append(f"{path}:{line}: ID {id_column} rỗng")
                elif item_id in seen:
                    errors.append(f"{path}:{line}: ID trùng {item_id}")
                seen.add(item_id)
                if status_column:
                    status = (row.get(status_column) or "").strip()
                    if status and status not in VALID_STATUSES:
                        errors.append(f"{path}:{line}: trạng thái không hợp lệ {status}")
    except (OSError, csv.Error) as exc:
        errors.append(f"{path}: không đọc được CSV: {exc}")
    return errors


def main() -> int:
    """Chạy toàn bộ kiểm tra."""
    errors: list[str] = []
    for relative, (id_column, required, status_column) in SCHEMAS.items():
        errors.extend(validate_file(ROOT / relative, id_column, required, status_column))
    if errors:
        print("\n".join(f"LỖI: {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"HỢP LỆ: đã kiểm tra {len(SCHEMAS)} file CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

