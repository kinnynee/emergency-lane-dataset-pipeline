from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from qc_outlier_boxes import Box, build_review_queue, find_outliers  # noqa: E402


def test_finds_outer_box_containing_four_boxes() -> None:
    outer = Box(0, 0.1, 0.1, 0.9, 0.9)
    inners = [Box(0, 0.2 + index * 0.1, 0.2, 0.25 + index * 0.1, 0.25) for index in range(4)]
    assert find_outliers([outer, *inners]) == [(0, [1, 2, 3, 4])]


def test_writes_review_queue_without_changing_labels(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    labels = dataset / "labels" / "val"
    labels.mkdir(parents=True)
    label = labels / "AAU_1949.txt"
    original = "\n".join([
        "0 0.5 0.5 0.8 0.8",
        "0 0.3 0.3 0.1 0.1",
        "0 0.4 0.3 0.1 0.1",
        "0 0.5 0.3 0.1 0.1",
        "0 0.6 0.3 0.1 0.1",
    ]) + "\n"
    label.write_text(original, encoding="utf-8")

    rows = build_review_queue(dataset, tmp_path / "queue.csv", splits=("val",), prefix="AAU_")

    assert rows[0]["image_id"] == "AAU_1949"
    assert rows[0]["review_action"] == "VISUAL_REVIEW_ONLY_DO_NOT_AUTO_REMOVE"
    assert label.read_text(encoding="utf-8") == original
