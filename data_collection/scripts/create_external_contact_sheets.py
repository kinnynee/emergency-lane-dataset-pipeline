"""Contact sheet pixel hóa toàn ảnh và lưu ngoài Git để review góc camera."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from external_eda_common import image_from_bytes


def _load(result: dict[str, Any], source_file: str) -> np.ndarray | None:
    path = Path(result["path"])
    if result["dataset_name"] == "MIO-TCD Localization":
        with tarfile.open(path, "r:") as archive:
            handle = archive.extractfile(source_file)
            return image_from_bytes(handle.read()) if handle else None
    if result["dataset_name"] == "UA-DETRAC Original":
        with zipfile.ZipFile(path) as archive:
            return image_from_bytes(archive.read(source_file))
    return cv2.imread(str(path / source_file), cv2.IMREAD_COLOR)


def create_contact_sheet(
    result: dict[str, Any],
    output_dir: Path,
    max_images: int = 12,
) -> Path | None:
    candidates = [
        row for row in result.get("quality_rows", []) if row.get("read_status") == "OK"
    ][:max_images]
    if not candidates:
        return None
    tiles: list[np.ndarray] = []
    for row in candidates:
        image = _load(result, str(row["source_file"]))
        if image is None:
            continue
        # Pixel hóa toàn ảnh để giảm khả năng đọc biển số/khuôn mặt.
        tiny = cv2.resize(image, (80, 45), interpolation=cv2.INTER_AREA)
        tile = cv2.resize(tiny, (320, 180), interpolation=cv2.INTER_NEAREST)
        label = f"{result['dataset_name']} | {row.get('sequence_name')} | {Path(str(row['source_file'])).name}"
        cv2.rectangle(tile, (0, 150), (320, 180), (0, 0, 0), -1)
        cv2.putText(tile, label[:54], (5, 169), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        return None
    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    canvas = np.zeros((rows * 180, columns * 320, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y, x = divmod(index, columns)
        canvas[y * 180 : (y + 1) * 180, x * 320 : (x + 1) * 320] = tile
    output_dir.mkdir(parents=True, exist_ok=True)
    name = result["dataset_name"].lower().replace(" ", "_").replace("-", "_") + "_pixelated.jpg"
    destination = output_dir / name
    cv2.imwrite(str(destination), canvas)
    return destination


__all__ = ["create_contact_sheet"]
