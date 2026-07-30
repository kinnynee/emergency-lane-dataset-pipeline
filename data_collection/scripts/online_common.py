"""Hàm dùng chung cho pipeline dữ liệu Internet, chỉ dùng thư viện chuẩn."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
ONLINE_ROOT = ROOT / "storage_placeholders/online_data"


def read_csv(relative: str) -> list[dict[str, str]]:
    """Đọc CSV UTF-8, trả danh sách rỗng nếu chỉ có header."""
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(relative: str, rows: Iterable[dict[str, str]], fields: list[str]) -> None:
    """Ghi CSV UTF-8 mà không thay đổi schema được gọi."""
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    """Tính SHA-256 theo khối để không nạp video vào RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    """Lấy metadata video bằng ffprobe hoặc nêu lỗi có hướng dẫn."""
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("Không tìm thấy ffprobe. Hãy cài FFmpeg từ nguồn tin cậy rồi thêm vào PATH.")
    command = [executable, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe không đọc được video")
    return json.loads(result.stdout)


def safe_name(value: str) -> str:
    """Chuẩn hóa tên file ASCII đơn giản, tránh khoảng trắng/ký tự đặc biệt."""
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return re.sub(r"_+", "_", normalized).strip("_")


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    """Tạo bảng Markdown nhỏ từ CSV."""
    if not rows:
        return ["CHƯA CÓ DỮ LIỆU."]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join((row.get(col) or "").replace("|", "/") for col in columns) + " |")
    return lines
