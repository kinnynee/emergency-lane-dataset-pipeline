"""Render evenly spaced YOLO-label samples into one contact sheet for visual QC."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2


def labels_for(path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        class_id, cx, cy, bw, bh = line.split()
        if class_id != "0":
            raise ValueError(f"Unexpected class {class_id} in {path}")
        cx, cy, bw, bh = (float(value) for value in (cx, cy, bw, bh))
        x1 = round((cx - bw / 2) * width)
        y1 = round((cy - bh / 2) * height)
        x2 = round((cx + bw / 2) * width)
        y2 = round((cy + bh / 2) * height)
        boxes.append((x1, y1, x2, y2))
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    images = sorted(args.images.glob("*.jpg"))
    if not images:
        raise ValueError("No JPG images found")
    positions = sorted({round(index * (len(images) - 1) / max(1, args.samples - 1)) for index in range(args.samples)})
    sample_images = [images[position] for position in positions]
    tile_width, tile_height = 400, 250
    tiles = []
    for image_path in sample_images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Unreadable image: {image_path}")
        height, width = image.shape[:2]
        boxes = labels_for(args.labels / f"{image_path.stem}.txt", width, height)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), max(2, width // 500))
        image = cv2.resize(image, (tile_width, tile_height - 28), interpolation=cv2.INTER_AREA)
        tile = cv2.copyMakeBorder(image, 0, 28, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(tile, f"{image_path.stem} | {len(boxes)} vehicles", (8, tile_height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    rows = math.ceil(len(tiles) / args.columns)
    blank = tiles[0] * 0
    while len(tiles) < rows * args.columns:
        tiles.append(blank.copy())
    contact_rows = [cv2.hconcat(tiles[index:index + args.columns]) for index in range(0, len(tiles), args.columns)]
    sheet = cv2.vconcat(contact_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"Could not write {args.output}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
