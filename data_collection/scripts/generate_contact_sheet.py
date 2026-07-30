"""Tạo contact sheet để review thủ công; không che hay sửa frame gốc."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from online_common import ONLINE_ROOT


def main() -> int:
    """Tạo một contact sheet tối đa 40 ảnh với tên file làm timestamp/evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ONLINE_ROOT / "frames")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ONLINE_ROOT / "contact_sheets/contact_sheet.jpg")
    args = parser.parse_args()
    frames = [p for p in args.input.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}][: args.max_frames]
    if not frames:
        print("CHƯA CÓ FRAME ĐỂ TẠO CONTACT SHEET.")
        return 0
    try:
        from PIL import Image, ImageDraw
        thumbs = []
        for frame in frames:
            image = Image.open(frame).convert("RGB")
            image.thumbnail((320, 180))
            cell = Image.new("RGB", (320, 210), "white")
            cell.paste(image, ((320 - image.width) // 2, 0))
            ImageDraw.Draw(cell).text((4, 184), frame.name[:48], fill="black")
            thumbs.append(cell)
        cols = 5
        sheet = Image.new("RGB", (cols * 320, math.ceil(len(thumbs) / cols) * 210), "white")
        for index, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((index % cols) * 320, (index // cols) * 210))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.output)
        print(f"Đã tạo contact sheet: {args.output}")
        return 0
    except (ImportError, OSError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
