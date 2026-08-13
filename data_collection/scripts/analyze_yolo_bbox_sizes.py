"""Create a 320x320 letterbox bbox-size EDA report for a YOLO release.

The report uses the exported annotation manifest, so it covers every retained
box rather than an EDA sample.  Size is measured after the same aspect-ratio
preserving resize (letterbox scale) used by YOLO.  A box is counted as
``under_threshold`` only when its *longest* side is below the requested pixel
threshold; this is deliberately stricter and less ambiguous than looking at
one side only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLITS = ("train", "val", "cross_test")
DEFAULT_BINS = (0, 5, 10, 15, 20, 25, 32, 48, 64, 96, 128, 160, 224, 320)


def _float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"{field} is not finite")
    return value


def _image_dimensions(path: Path) -> dict[str, tuple[int, int, str]]:
    dimensions: dict[str, tuple[int, int, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = str(row.get("image_id", "")).strip()
            split = str(row.get("split", "")).strip()
            try:
                width = int(float(str(row.get("width", ""))))
                height = int(float(str(row.get("height", ""))))
            except ValueError:
                continue
            if image_id and split in SPLITS and width > 0 and height > 0:
                dimensions[image_id] = (width, height, split)
    return dimensions


def _histogram(values: list[float], bins: tuple[int, ...]) -> list[dict[str, Any]]:
    counts = [0] * len(bins)
    for value in values:
        index = len(bins) - 1
        for candidate in range(len(bins) - 1):
            if bins[candidate] <= value < bins[candidate + 1]:
                index = candidate
                break
        counts[index] += 1
    total = len(values)
    rows: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        upper = bins[index + 1] if index + 1 < len(bins) else "INF"
        rows.append(
            {
                "min_px_inclusive": bins[index],
                "max_px_exclusive": upper,
                "count": count,
                "ratio": round(count / total, 8) if total else 0.0,
            }
        )
    return rows


def analyze_bbox_sizes(
    dataset: Path,
    output: Path | None = None,
    target_size: int = 320,
    threshold_px: float = 25.0,
) -> dict[str, Any]:
    """Write a CSV, JSON and PNG report and return the JSON payload."""
    if target_size <= 0 or threshold_px <= 0:
        raise ValueError("target_size and threshold_px must be positive")
    metadata = dataset / "metadata"
    images_path = metadata / "images.csv"
    annotations_path = metadata / "annotations.csv"
    if not images_path.is_file() or not annotations_path.is_file():
        raise FileNotFoundError("Expected metadata/images.csv and metadata/annotations.csv")
    destination = output or metadata / "eda"
    destination.mkdir(parents=True, exist_ok=True)

    dimensions = _image_dimensions(images_path)
    longest_sides: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    under_by_split: Counter[str] = Counter()
    boxes_by_split: Counter[str] = Counter()
    skipped = Counter()
    with annotations_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = str(row.get("image_id", "")).strip()
            image = dimensions.get(image_id)
            if image is None:
                skipped["MISSING_IMAGE_DIMENSIONS"] += 1
                continue
            try:
                xmin = _float(row, "clipped_xmin")
                ymin = _float(row, "clipped_ymin")
                xmax = _float(row, "clipped_xmax")
                ymax = _float(row, "clipped_ymax")
            except (KeyError, TypeError, ValueError):
                skipped["INVALID_BOX"] += 1
                continue
            raw_width, raw_height = xmax - xmin, ymax - ymin
            if raw_width <= 0 or raw_height <= 0:
                skipped["NON_POSITIVE_BOX"] += 1
                continue
            image_width, image_height, split = image
            scale = min(target_size / image_width, target_size / image_height)
            width_320, height_320 = raw_width * scale, raw_height * scale
            longest = max(width_320, height_320)
            widths.append(width_320)
            heights.append(height_320)
            longest_sides.append(longest)
            boxes_by_split[split] += 1
            if longest < threshold_px:
                under_by_split[split] += 1

    total = len(longest_sides)
    under_total = sum(under_by_split.values())
    histogram = _histogram(longest_sides, DEFAULT_BINS)
    by_split = {
        split: {
            "box_count": boxes_by_split[split],
            "under_threshold_count": under_by_split[split],
            "under_threshold_ratio": round(under_by_split[split] / boxes_by_split[split], 8)
            if boxes_by_split[split]
            else 0.0,
        }
        for split in SPLITS
    }
    report = {
        "dataset": str(dataset.resolve()),
        "resize": {"width": target_size, "height": target_size, "method": "letterbox_aspect_ratio_preserved"},
        "threshold": {
            "pixels": threshold_px,
            "definition": "longest_bbox_side_px_after_letterbox_resize < threshold",
        },
        "box_count": total,
        "under_threshold_count": under_total,
        "under_threshold_ratio": round(under_total / total, 8) if total else 0.0,
        "by_split": by_split,
        "skipped_annotations": dict(sorted(skipped.items())),
        "histogram_measure": "longest_bbox_side_px_after_letterbox_resize",
        "histogram": histogram,
    }
    (destination / "bbox_size_320_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (destination / "bbox_size_320_histogram.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["min_px_inclusive", "max_px_exclusive", "count", "ratio"])
        writer.writeheader()
        writer.writerows(histogram)

    ratio_text = f"{under_total / total:.2%}" if total else "0.00%"
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.hist(longest_sides, bins=list(DEFAULT_BINS) + [max(321, int(max(longest_sides, default=320)) + 1)], color="#376f9f", edgecolor="white")
    axis.axvline(threshold_px, color="#c23b22", linewidth=2, linestyle="--", label=f"Threshold {threshold_px:g}px")
    axis.set_xlabel("BBox longest side after letterbox resize to 320×320 (px)")
    axis.set_ylabel("Box count")
    axis.set_title(f"BBox-size distribution — {under_total:,}/{total:,} ({ratio_text}) below {threshold_px:g}px")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "bbox_size_320_histogram.png", dpi=180)
    plt.close(figure)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-size", type=int, default=320)
    parser.add_argument("--threshold-px", type=float, default=25.0)
    args = parser.parse_args()
    report = analyze_bbox_sizes(args.dataset.resolve(), args.output.resolve() if args.output else None, args.target_size, args.threshold_px)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
