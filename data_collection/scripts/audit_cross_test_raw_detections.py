"""Read-only, restricted audit of the 10 October cross-test prediction archive.

Run with the repository venv's Python. This script never calls pickle.load and
allows only the two NumPy reconstruction globals observed in the supplied files.
It reports independent recall and two explicitly scoped precision calculations;
the official evaluator script is not included in the supplied archive.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import pickle
from zipfile import ZipFile

import numpy as np


EXPECTED = {
    "detections_ba_model.pkl": "0822f98bdb44a235900a4fa7602e07c23a9b86712395890207b311c3be234a6a",
    "detections.pkl": "04603f35b036aaa4b81f0d1ae68f059fb86ce56478645d561f486ac42221028b",
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        allowed = {
            ("numpy._core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
            ("numpy", "dtype"): np.dtype,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"blocked global: {module}.{name}")
        return allowed[(module, name)]


def file_hash(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def match_flags(gt: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Greedy, descending-score IoU matching; boxes are pre-sorted in pickle."""
    flags = np.zeros(len(boxes), dtype=bool)
    if not len(gt) or not len(boxes):
        return flags
    gt_px = gt * 320.0
    used = np.zeros(len(gt_px), dtype=bool)
    for index, box in enumerate(boxes):
        left = np.maximum(gt_px[:, :2], box[:2])
        right = np.minimum(gt_px[:, 2:], box[2:])
        inter_size = np.maximum(right - left, 0)
        intersection = inter_size[:, 0] * inter_size[:, 1]
        a = np.maximum(gt_px[:, 2:] - gt_px[:, :2], 0)
        b = np.maximum(box[2:] - box[:2], 0)
        union = a[:, 0] * a[:, 1] + b[0] * b[1] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        iou[used] = -1
        best = int(np.argmax(iou))
        if iou[best] >= 0.5:
            used[best] = True
            flags[index] = True
    return flags


def overlap(candidate: np.ndarray, references: np.ndarray, denominator: str) -> float:
    if not len(references):
        return 0.0
    left = np.maximum(references[:, :2], candidate[:2])
    right = np.minimum(references[:, 2:], candidate[2:])
    size = np.maximum(right - left, 0)
    intersection = size[:, 0] * size[:, 1]
    candidate_size = np.maximum(candidate[2:] - candidate[:2], 0)
    candidate_area = candidate_size[0] * candidate_size[1]
    if denominator == "ioa":
        divisor = np.full(len(references), candidate_area)
    else:
        ref_size = np.maximum(references[:, 2:] - references[:, :2], 0)
        divisor = candidate_area + ref_size[:, 0] * ref_size[:, 1] - intersection
    values = np.divide(intersection, divisor, out=np.zeros_like(intersection), where=divisor > 0)
    return float(np.max(values))


def transformed_ua_regions(regions: list[np.ndarray], width: int, height: int, geometry: str) -> np.ndarray:
    if not regions:
        return np.empty((0, 4), dtype=np.float32)
    box = np.asarray(regions, dtype=np.float32).copy()
    if geometry == "H3":
        crop_width = min(width, height * 4 / 3)
        x_offset = (width - crop_width) / 2
        box[:, [0, 2]] = (box[:, [0, 2]] - x_offset) * (320 / crop_width)
    else:
        box[:, [0, 2]] *= 320 / width
    box[:, [1, 3]] *= 320 / height
    return np.clip(box, 0, 320)


def audit(archive: Path, export_images: Path, ua_ignore_csv: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    expected_stems: set[str] = set()
    with export_images.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["split"] == "cross_test":
                expected_stems.add(row["image_id"])
    if len(expected_stems) != 12522:
        raise ValueError(f"Expected 12,522 unique export image IDs; got {len(expected_stems)}")
    ua_regions = defaultdict(list)
    with ua_ignore_csv.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            ua_regions[row["sequence"]].append(np.asarray([float(row[k]) for k in ("left", "top", "xmax", "ymax")], dtype=np.float32))

    totals = defaultdict(lambda: Counter())
    sequence_counts = Counter()
    checks = {}
    with ZipFile(archive) as z:
        archive_names = {Path(name).name: name for name in z.namelist() if name.endswith(".pkl")}
        if set(archive_names) != set(EXPECTED):
            raise ValueError(f"Unexpected pickle members: {archive_names}")
        for short_name, expected_hash in EXPECTED.items():
            member = archive_names[short_name]
            with z.open(member) as stream:
                digest = file_hash(stream)
            if digest != expected_hash:
                raise ValueError(f"SHA-256 mismatch: {short_name}: {digest}")
            with z.open(member) as stream:
                rows = RestrictedUnpickler(stream).load()
            if not isinstance(rows, list) or len(rows) != 12522:
                raise ValueError(f"Unexpected row count: {short_name}")
            stems = [r["stem"] for r in rows]
            if len(set(stems)) != len(stems) or set(stems) != expected_stems:
                raise ValueError(f"Image IDs differ from export: {short_name}")
            checks[short_name] = {"sha256": digest, "rows": len(rows), "unique_stems": len(set(stems))}
            for row in rows:
                if short_name == "detections_ba_model.pkl":
                    sequence_counts[row["seq"]] += 1
                for geometry, item in row["hh"].items():
                    gt = np.asarray(item["gt"], dtype=np.float32)
                    if gt.ndim != 2 or gt.shape[1] != 4:
                        raise ValueError("Invalid GT shape")
                    for model, (box, score) in item["det"].items():
                        key = (short_name, row["seq"], geometry, model)
                        counter = totals[key]
                        counter["images"] += 1
                        counter["gt"] += len(gt)
                        box = np.asarray(box, dtype=np.float32)
                        score = np.asarray(score, dtype=np.float32)
                        if box.shape != (len(score), 4):
                            raise ValueError("Invalid detection shape")
                        operational = box[score > 0.5][:20]
                        counter["detections_before_guard"] += len(operational)
                        # H1/H3: 320x320 model coordinates mapped to 640x480 display.
                        width = (operational[:, 2] - operational[:, 0]) * 2.0
                        height = (operational[:, 3] - operational[:, 1]) * 1.5
                        guarded = operational[(width >= 25) & (height >= 25)]
                        counter["detections_after_guard"] += len(guarded)
                        counter["tp_no_guard"] += int(match_flags(gt, operational).sum())
                        matched = match_flags(gt, guarded)
                        counter["tp_guard"] += int(matched.sum())
                        if geometry in ("H1", "H3"):
                            ignored = np.concatenate((np.asarray(item["ign"], dtype=np.float32).reshape(-1, 4), np.asarray(item["bo"], dtype=np.float32).reshape(-1, 4))) * 320
                            source_ignored = transformed_ua_regions(ua_regions[row["seq"]], int(row["W"]), int(row["H"]), geometry)
                            for candidate in guarded[~matched]:
                                if overlap(candidate, ignored, "iou") > 0.3:
                                    counter["ignored_local"] += 1
                                elif overlap(candidate, source_ignored, "ioa") >= 0.5:
                                    counter["ignored_ua"] += 1
                                else:
                                    counter["fp_guard"] += 1
            del rows

    fields = ["file", "sequence", "geometry", "model", "images", "gt", "detections_before_guard", "detections_after_guard", "tp_no_guard", "tp_guard", "fn_guard", "recall_guard_pct", "fp_without_ua_regions", "precision_without_ua_regions_pct", "fp_after_ua_regions", "precision_after_ua_regions_pct", "ignored_local", "ignored_ua"]
    with (output / "raw_detection_recall.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for (filename, seq, geom, model), c in sorted(totals.items()):
            fp_without_ua = c["fp_guard"] + c["ignored_ua"]
            writer.writerow({"file": filename, "sequence": seq, "geometry": geom, "model": model, **{field: c[field] for field in ("images", "gt", "detections_before_guard", "detections_after_guard", "tp_no_guard", "tp_guard")}, "fn_guard": c["gt"] - c["tp_guard"], "recall_guard_pct": f"{100*c['tp_guard']/c['gt']:.4f}" if c["gt"] else "MISSING", "fp_without_ua_regions": fp_without_ua if geom in ("H1", "H3") else "MISSING", "precision_without_ua_regions_pct": f"{100*c['tp_guard']/(c['tp_guard']+fp_without_ua):.4f}" if geom in ("H1", "H3") and c["tp_guard"]+fp_without_ua else "MISSING", "fp_after_ua_regions": c["fp_guard"] if geom in ("H1", "H3") else "MISSING", "precision_after_ua_regions_pct": f"{100*c['tp_guard']/(c['tp_guard']+c['fp_guard']):.4f}" if geom in ("H1", "H3") and c["tp_guard"]+c["fp_guard"] else "MISSING", "ignored_local": c["ignored_local"] if geom in ("H1", "H3") else "MISSING", "ignored_ua": c["ignored_ua"] if geom in ("H1", "H3") else "MISSING"})
    summary = {"source_zip": str(archive), "export_images": str(export_images), "ua_ignore_csv": str(ua_ignore_csv), "ua_ignore_regions": sum(map(len, ua_regions.values())), "files": checks, "sequence_images": dict(sorted(sequence_counts.items())), "method": "Independent greedy IoU>=0.5; score>0.5; top 20; H1/H3 guard maps 320x320 to 640x480 and requires both sides>=25px. Local ign/bo IoU>0.3 always removed from FP. UA source rectangles IoA>=0.5 removed from FP. The after-UA-regions columns are the official precision basis, as confirmed by the experiment owner on 2026-10-10; the without-UA-regions columns are historical diagnostics. Original evaluator script unavailable."}
    (output / "raw_detection_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("export_images", type=Path)
    parser.add_argument("ua_ignore_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.archive, args.export_images, args.ua_ignore_csv, args.output)
