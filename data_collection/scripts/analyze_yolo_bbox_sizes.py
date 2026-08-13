"""Analyze exported YOLO bounding-box sizes at the 320 px model input scale."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze_bbox_sizes(dataset: Path, input_size: int = 320, threshold_px: float = 25.0) -> dict[str, object]:
    """Write bbox-size EDA using the uniform letterbox scale."""
    dataset = Path(dataset).resolve()
    metadata = dataset / "metadata"
    images = {row["image_id"]: row for row in _read_rows(metadata / "images.csv")}
    annotations = _read_rows(metadata / "annotations.csv")
    longest_sides: list[float] = []
    rows: list[dict[str, object]] = []
    for annotation in annotations:
        image_id = annotation["image_id"]
        image = images.get(image_id)
        if image is None:
            raise ValueError(f"Annotation references unknown image_id: {image_id}")
        width, height = float(image["width"]), float(image["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Image has invalid dimensions: {image_id}")
        scale = input_size / max(width, height)
        box_width = (float(annotation["clipped_xmax"]) - float(annotation["clipped_xmin"])) * scale
        box_height = (float(annotation["clipped_ymax"]) - float(annotation["clipped_ymin"])) * scale
        longest = max(box_width, box_height)
        longest_sides.append(longest)
        rows.append({"image_id": image_id, "split": image.get("split", ""), "bbox_width_320": round(box_width, 6), "bbox_height_320": round(box_height, 6), "longest_side_320": round(longest, 6), "under_25px": longest < threshold_px})

    output = metadata / "eda"
    output.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "split", "bbox_width_320", "bbox_height_320", "longest_side_320", "under_25px"]
    with (output / "bbox_size_320.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(longest_sides, bins=min(30, max(1, len(longest_sides))), color="#3977b8", edgecolor="white")
    axis.axvline(threshold_px, color="#c43c39", linestyle="--", label=f"{threshold_px:g} px guard")
    axis.set(xlabel=f"Longest bbox side at {input_size}px input (px)", ylabel="Boxes", title="YOLO bbox-size distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "bbox_size_320_histogram.png", dpi=160)
    plt.close(figure)
    under = sum(value < threshold_px for value in longest_sides)
    report: dict[str, object] = {"input_size": input_size, "threshold_px": threshold_px, "box_count": len(longest_sides), "under_threshold_count": under, "under_threshold_ratio": under / len(longest_sides) if longest_sides else 0.0}
    (output / "bbox_size_320_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--threshold-px", type=float, default=25.0)
    args = parser.parse_args()
    print(json.dumps(analyze_bbox_sizes(args.dataset, args.input_size, args.threshold_px), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
