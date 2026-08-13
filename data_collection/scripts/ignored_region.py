"""Safe handling of UA-DETRAC ``ignored_region`` boxes.

UA-DETRAC marks areas that must not teach the detector or count as false
positives.  These utilities deliberately keep the original annotations
unchanged: they create a black-masked training copy and filter *predictions*
by bbox centre before metrics are calculated.
"""

from __future__ import annotations

import io
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


IgnoredRegion = tuple[float, float, float, float]


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_ua_ignored_regions(root: ET.Element) -> list[IgnoredRegion]:
    """Return valid ``(xmin, ymin, xmax, ymax)`` UA ignored regions."""
    regions: list[IgnoredRegion] = []
    for box in root.findall("./ignored_region/box"):
        left, top, width, height = (
            _number(box.attrib.get(field, "")) for field in ("left", "top", "width", "height")
        )
        if left is None or top is None or width is None or height is None or width <= 0 or height <= 0:
            continue
        regions.append((left, top, left + width, top + height))
    return regions


def mask_image_bytes(image_bytes: bytes, regions: Iterable[IgnoredRegion], source_name: str) -> bytes:
    """Return an RGB image with valid regions filled black.

    The function changes only the derived exported image.  It never writes to
    an input image or annotation file.
    """
    with Image.open(io.BytesIO(image_bytes)) as original:
        image = original.convert("RGB")
        image_format = original.format or ("PNG" if Path(source_name).suffix.lower() == ".png" else "JPEG")
    drawer = ImageDraw.Draw(image)
    drawn = False
    for xmin, ymin, xmax, ymax in regions:
        left = max(0, min(image.width, math.floor(xmin)))
        top = max(0, min(image.height, math.floor(ymin)))
        right = max(0, min(image.width, math.ceil(xmax)))
        bottom = max(0, min(image.height, math.ceil(ymax)))
        if left < right and top < bottom:
            drawer.rectangle((left, top, right - 1, bottom - 1), fill=(0, 0, 0))
            drawn = True
    if not drawn:
        return image_bytes
    destination = io.BytesIO()
    options: dict[str, Any] = {"format": image_format}
    if image_format.upper() in {"JPEG", "JPG"}:
        options.update(quality=95, subsampling=0)
    image.save(destination, **options)
    return destination.getvalue()


def bbox_center_is_ignored(box: IgnoredRegion, regions: Iterable[IgnoredRegion]) -> bool:
    """Whether the box centre belongs to a UA ignored region (boundary included)."""
    xmin, ymin, xmax, ymax = box
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    return any(left <= center_x <= right and top <= center_y <= bottom for left, top, right, bottom in regions)


def filter_boxes_by_ignored_region(
    boxes: Iterable[IgnoredRegion], regions: Iterable[IgnoredRegion]
) -> tuple[list[IgnoredRegion], list[IgnoredRegion]]:
    """Split boxes into metric-eligible and ignored lists using bbox centres."""
    region_list = list(regions)
    kept: list[IgnoredRegion] = []
    ignored: list[IgnoredRegion] = []
    for box in boxes:
        (ignored if bbox_center_is_ignored(box, region_list) else kept).append(box)
    return kept, ignored


def region_map_from_ua_annotations(path: Path) -> dict[str, list[IgnoredRegion]]:
    """Load ignored regions keyed by UA sequence from an XML directory."""
    if not path.is_dir():
        raise FileNotFoundError(f"UA annotation directory not found: {path}")
    result: dict[str, list[IgnoredRegion]] = {}
    for xml_path in sorted(path.rglob("*.xml")):
        root = ET.parse(xml_path).getroot()
        sequence_id = root.attrib.get("name") or xml_path.stem
        result[sequence_id] = parse_ua_ignored_regions(root)
    if not result:
        raise ValueError(f"No UA XML annotations were found in {path}")
    return result


def box_from_row(row: Mapping[str, object]) -> IgnoredRegion | None:
    values = tuple(_number(row.get(field, "")) for field in ("xmin", "ymin", "xmax", "ymax"))
    if any(value is None for value in values):
        return None
    xmin, ymin, xmax, ymax = values  # type: ignore[misc]
    return (xmin, ymin, xmax, ymax) if xmin < xmax and ymin < ymax else None
