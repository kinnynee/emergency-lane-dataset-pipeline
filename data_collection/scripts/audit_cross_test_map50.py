"""Recompute one-class AP@50 after UA ignore regions from saved ONNX boxes.

The supplied pickle holds score>=0.01, NMS=0.5, top-100 detections per
image, sorted by score. AP has no 25-pixel operating guard. We greedily match
each image at IoU>=0.5, exclude unmatched ign/bo and UA-region detections,
then integrate the global precision envelope over recall for each sequence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from audit_cross_test_raw_detections import (
    EXPECTED,
    RestrictedUnpickler,
    file_hash,
    match_flags,
    overlap,
    transformed_ua_regions,
)


def average_precision(detections: list[tuple[float, int]], gt_count: int) -> float:
    """All-point interpolated precision-recall area, as a percent."""
    if gt_count <= 0:
        raise ValueError("AP requires a positive GT count")
    detections.sort(key=lambda item: -item[0])
    tp = np.cumsum([item[1] for item in detections])
    fp = np.cumsum([1 - item[1] for item in detections])
    recall = tp / gt_count
    precision = tp / (tp + fp)
    rec = np.concatenate(([0], recall, [1]))
    prec = np.concatenate(([1], precision, [0]))
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(100 * np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def audit(archive: Path, ua_ignore_csv: Path, output_dir: Path) -> None:
    ua_regions: dict[str, list[np.ndarray]] = defaultdict(list)
    with ua_ignore_csv.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            ua_regions[row["sequence"]].append(
                np.asarray([float(row[k]) for k in ("left", "top", "xmax", "ymax")], dtype=np.float32)
            )
    if sum(map(len, ua_regions.values())) != 29:
        raise ValueError("Expected 29 UA source ignore rectangles")

    with ZipFile(archive) as z:
        members = {Path(name).name: name for name in z.namelist() if name.endswith(".pkl")}
        if set(members) != set(EXPECTED):
            raise ValueError(f"Unexpected pickle members: {members}")
        name = members["detections_ba_model.pkl"]
        with z.open(name) as stream:
            digest = file_hash(stream)
        if digest != EXPECTED["detections_ba_model.pkl"]:
            raise ValueError("SHA-256 mismatch for detections_ba_model.pkl")
        with z.open(name) as stream:
            rows = RestrictedUnpickler(stream).load()
    if not isinstance(rows, list) or len(rows) != 12522:
        raise ValueError("Expected 12,522 image rows")
    stems = [row["stem"] for row in rows]
    if len(set(stems)) != len(stems):
        raise ValueError("Duplicate image ID")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["seq"]].append(row)
    if len(grouped) != 10:
        raise ValueError("Expected 10 cross-test sequences")

    metrics = []
    for sequence, sequence_rows in sorted(grouped.items()):
        combinations = [("H1", "full"), ("H3", "full")]
        if sequence.startswith("MVI_"):
            combinations += [("H1", "squash"), ("H1", "letterbox")]
        for geometry, model in combinations:
            detections: list[tuple[float, int]] = []
            gt_count = 0
            for row in sequence_rows:
                item = row["hh"][geometry]
                gt = np.asarray(item["gt"], dtype=np.float32)
                gt_count += len(gt)
                boxes, scores = item["det"][model]
                boxes = np.asarray(boxes, dtype=np.float32)
                scores = np.asarray(scores, dtype=np.float32)
                if boxes.shape != (len(scores), 4):
                    raise ValueError("Invalid detection shape")
                if len(scores) and (np.min(scores) < 0.01 or np.any(scores[:-1] < scores[1:])):
                    raise ValueError("Pickle detections must be score>=0.01 and score-sorted")
                matched = match_flags(gt, boxes)
                local_ignore = np.concatenate(
                    (
                        np.asarray(item["ign"], dtype=np.float32).reshape(-1, 4),
                        np.asarray(item["bo"], dtype=np.float32).reshape(-1, 4),
                    )
                ) * 320
                source_ignore = transformed_ua_regions(
                    ua_regions.get(sequence, []), int(row["W"]), int(row["H"]), geometry
                )
                for box, score, is_tp in zip(boxes, scores, matched):
                    if not is_tp:
                        if overlap(box, local_ignore, "iou") > 0.3:
                            continue
                        if overlap(box, source_ignore, "ioa") >= 0.5:
                            continue
                    detections.append((float(score), int(is_tp)))
            ap = average_precision(detections, gt_count)
            metrics.append({
                "sequence": sequence,
                "images": len(sequence_rows),
                "geometry": geometry,
                "model": model,
                "gt": gt_count,
                "detections_in_ap": len(detections),
                "ap50_pct": ap,
            })

    by_key = {(row["sequence"], row["geometry"], row["model"]): row["ap50_pct"] for row in metrics}
    ua_sequences = sorted(sequence for sequence in grouped if sequence.startswith("MVI_"))
    all_sequences = sorted(grouped)
    if len(ua_sequences) != 8:
        raise ValueError("Expected 8 UA sequences")
    means = {}
    for label, sequences, geometry, model in (
        ("ua_8_full_H1", ua_sequences, "H1", "full"),
        ("ua_8_full_H3", ua_sequences, "H3", "full"),
        ("all_10_full_H1", all_sequences, "H1", "full"),
        ("all_10_full_H3", all_sequences, "H3", "full"),
        ("ua_8_squash_H1", ua_sequences, "H1", "squash"),
        ("ua_8_letterbox_H1", ua_sequences, "H1", "letterbox"),
    ):
        means[label] = float(np.mean([by_key[(sequence, geometry, model)] for sequence in sequences]))

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "map50_after_ua_regions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sequence", "images", "geometry", "model", "gt", "detections_in_ap", "ap50_pct"))
        writer.writeheader()
        for row in metrics:
            writer.writerow({**row, "ap50_pct": f"{row['ap50_pct']:.6f}"})
    (output_dir / "map50_after_ua_regions_summary.json").write_text(
        json.dumps({
            "source_pickle_sha256": digest,
            "ua_ignore_regions": 29,
            "method": "score>=0.01; NMS 0.5 and top 100 preapplied in pickle; per-image greedy IoU>=0.5; unmatched local ign/bo IoU>0.3 and UA source IoA>=0.5 excluded; no guard; all-point precision-envelope AP@50; unweighted mean of sequence AP",
            "sequence_count": len(grouped),
            "image_count": len(rows),
            "mean_ap50_pct": means,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("ua_ignore_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    audit(args.archive, args.ua_ignore_csv, args.output_dir)
