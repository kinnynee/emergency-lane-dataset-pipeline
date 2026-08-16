"""Build a review queue for YOLO boxes that fully contain several other boxes.

This is a triage heuristic, not an automatic label deletion tool.  A large bus
box can legitimately contain detections visible through its windows.  The
queue therefore records the evidence needed for a human reviewer and never
changes a label file.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    class_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def area(self) -> float:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


def _read_boxes(path: Path) -> list[Box]:
    boxes: list[Box] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Malformed YOLO label at {path}:{line_number}")
        class_id = int(values[0])
        x_center, y_center, width, height = (float(value) for value in values[1:])
        if width <= 0 or height <= 0:
            raise ValueError(f"Non-positive YOLO box at {path}:{line_number}")
        boxes.append(Box(class_id, x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2))
    return boxes


def _contains(outer: Box, inner: Box, tolerance: float) -> bool:
    return (
        outer.class_id == inner.class_id
        and outer.xmin <= inner.xmin + tolerance
        and outer.ymin <= inner.ymin + tolerance
        and outer.xmax >= inner.xmax - tolerance
        and outer.ymax >= inner.ymax - tolerance
        and outer.area > inner.area
    )


def find_outliers(boxes: list[Box], minimum_contained: int = 4, tolerance: float = 0.0) -> list[tuple[int, list[int]]]:
    """Return ``(outer_index, contained_indices)`` candidates for review."""
    candidates = []
    for outer_index, outer in enumerate(boxes):
        contained = [
            inner_index
            for inner_index, inner in enumerate(boxes)
            if inner_index != outer_index and _contains(outer, inner, tolerance)
        ]
        if len(contained) >= minimum_contained:
            candidates.append((outer_index, contained))
    return candidates


def build_review_queue(
    dataset: Path,
    output: Path,
    splits: tuple[str, ...] = ("train", "val", "cross_test"),
    prefix: str = "",
    minimum_contained: int = 4,
    tolerance: float = 0.0,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in splits:
        labels = dataset / "labels" / split
        if not labels.is_dir():
            raise FileNotFoundError(f"Missing label split directory: {labels}")
        for label in sorted(labels.glob(f"{prefix}*.txt"), key=lambda path: path.name.casefold()):
            boxes = _read_boxes(label)
            image_candidates = [
                dataset / "images" / split / f"{label.stem}{suffix}"
                for suffix in (".jpg", ".jpeg", ".png")
            ]
            image = next((candidate for candidate in image_candidates if candidate.is_file()), image_candidates[0])
            for outer_index, contained in find_outliers(boxes, minimum_contained, tolerance):
                outer = boxes[outer_index]
                rows.append({
                    "split": split,
                    "image_id": label.stem,
                    "image_path": image.relative_to(dataset).as_posix(),
                    "label_path": label.relative_to(dataset).as_posix(),
                    "outer_box_index": str(outer_index),
                    "outer_area_normalized": f"{outer.area:.8f}",
                    "contained_box_count": str(len(contained)),
                    "contained_box_indices": ";".join(str(index) for index in contained),
                    "review_action": "VISUAL_REVIEW_ONLY_DO_NOT_AUTO_REMOVE",
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "image_id", "image_path", "label_path", "outer_box_index", "outer_area_normalized",
        "contained_box_count", "contained_box_indices", "review_action",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "cross_test"])
    parser.add_argument("--prefix", default="", help="Optional image/label prefix, e.g. AAU_")
    parser.add_argument("--minimum-contained", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()
    if args.minimum_contained < 1 or args.tolerance < 0:
        parser.error("--minimum-contained must be positive and --tolerance cannot be negative")
    rows = build_review_queue(
        args.dataset.resolve(), args.output.resolve(), tuple(args.splits), args.prefix,
        args.minimum_contained, args.tolerance,
    )
    print(f"Wrote {len(rows)} review candidates to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
