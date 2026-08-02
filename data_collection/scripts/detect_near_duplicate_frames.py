"""Đánh dấu cặp ảnh gần trùng bằng average-hash; không xóa ảnh."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from online_common import ONLINE_ROOT, ROOT


def average_hash(path: Path) -> int:
    """Tính aHash 8x8 bằng Pillow; chỉ dùng cho sàng lọc review."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Thiếu Pillow; cài requirements-online-data.txt") from exc
    with Image.open(path) as image:
        gray = image.convert("L").resize((8, 8))
        pixels = list(gray.get_flattened_data())
    average = sum(pixels) / len(pixels)
    return sum((1 << index) for index, pixel in enumerate(pixels) if pixel >= average)


def main() -> int:
    """Tạo báo cáo các cặp có Hamming distance thấp."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ONLINE_ROOT / "frames")
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    images = [p for p in args.input.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        print("CHƯA CÓ FRAME ĐỂ KIỂM TRA GẦN TRÙNG.")
        return 0
    try:
        hashes = {path: average_hash(path) for path in images}
    except (RuntimeError, OSError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1
    pairs = [(a, b, (hashes[a] ^ hashes[b]).bit_count()) for index, a in enumerate(images) for b in images[index + 1 :] if (hashes[a] ^ hashes[b]).bit_count() <= args.threshold]
    lines = ["# Báo cáo frame gần trùng", "", f"Ngưỡng Hamming: {args.threshold}. Không ảnh nào bị xóa tự động."]
    def portable(path: Path) -> str:
        try:
            return path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return f"<EXTERNAL_FRAME_ROOT>/{path.name}"

    lines += [f"- Giữ đề xuất: `{portable(a)}`; gần trùng: `{portable(b)}`; distance={distance}" for a, b, distance in pairs] or ["", "Không có cặp gần trùng được phát hiện."]
    (ROOT / "reports/near_duplicate_frame_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Near-duplicate report written; candidate pairs: {len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
