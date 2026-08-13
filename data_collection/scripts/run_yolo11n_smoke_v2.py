"""Run the team-model 500-image / 25-epoch YOLO11n smoke test.

The source release has non-reconciled online additions, so this runner selects
only MIO, AAU RainSnow and UA-DETRAC rows.  It creates a derived dataset in a
new run directory, black-masks UA ignored regions in *training* images, and
does not alter source images or labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw

from ignored_region import (
    IgnoredRegion,
    bbox_center_is_ignored,
    mask_image_bytes,
    parse_ua_ignored_regions,
)
from validate_yolo_dataset import validate_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "cross_test")
CORE_DATASETS = ("MIO-TCD Localization", "AAU RainSnow", "UA-DETRAC Original")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
UA_DATASET = "UA-DETRAC Original"
TARGETS = {
    "train": {"MIO-TCD Localization": 166, "AAU RainSnow": 167, "UA-DETRAC Original": 167},
    "val": {"AAU RainSnow": 90, "UA-DETRAC Original": 90},
    "cross_test": {"AAU RainSnow": 90, "UA-DETRAC Original": 90},
}


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {
        "model", "model_variant", "imgsz", "epochs", "batch", "seed", "deterministic", "workers",
        "optimizer", "pretrained", "plots", "prediction_confidence", "train_images", "validation_images",
        "cross_test_images", "source_dataset", "ua_annotation_roots", "base_weights", "training_stage",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"V2 config missing fields: {', '.join(missing)}")
    if config["model_variant"] != "YOLO11n" or Path(str(config["model"])).suffix.lower() != ".pt":
        raise ValueError("The smoke run must fine-tune a local YOLO11n .pt base checkpoint")
    if not bool(config["pretrained"]):
        raise ValueError("pretrained must be true: this smoke run validates transfer-learning provenance")
    if config.get("model_provenance_policy") != "FINETUNED_TEAM_MODEL_REQUIRED":
        raise ValueError("Missing FINETUNED_TEAM_MODEL_REQUIRED provenance policy")
    if config["training_stage"] != "SMOKE_PIPELINE_ONLY":
        raise ValueError("This runner accepts only training_stage=SMOKE_PIPELINE_ONLY")
    if int(config["imgsz"]) != 320 or not 20 <= int(config["epochs"]) <= 30:
        raise ValueError("V2 requires imgsz=320 and 20–30 epochs")
    if int(config["train_images"]) != 500:
        raise ValueError("The initial team-model run requires the deterministic 500-train-image smoke selection")
    return config


def _resolve_config_path(config_path: Path, configured_path: str | Path) -> Path:
    """Resolve a local configurable path without embedding a machine drive."""
    candidate = Path(configured_path)
    return candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def _load_ua_ignored_regions(roots: Iterable[Path]) -> dict[str, list[IgnoredRegion]]:
    result: dict[str, list[IgnoredRegion]] = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"UA annotation directory not found: {root}")
        for xml_path in root.rglob("*.xml"):
            sequence = ET.parse(xml_path).getroot()
            sequence_id = sequence.attrib.get("name") or xml_path.stem
            regions = parse_ua_ignored_regions(sequence)
            previous = result.get(sequence_id)
            if previous is not None and previous != regions:
                raise ValueError(f"UA sequence has conflicting ignored regions: {sequence_id}")
            result[sequence_id] = regions
    if not result:
        raise ValueError("No UA XML files were discovered for ignored_region handling")
    return result


def _select_rows(source: Path, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    images, image_fields = _read_csv(source / "metadata" / "images.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in images:
        dataset, split = row.get("dataset", ""), row.get("split", "")
        if dataset not in CORE_DATASETS or split not in TARGETS or dataset not in TARGETS[split]:
            continue
        if int(row.get("vehicle_box_count", "0") or 0) <= 0:
            continue  # Empty labels remain in the human-review queue, never silently used for C.
        image = source / str(row.get("exported_image", ""))
        label = source / str(row.get("exported_label", ""))
        if image.is_file() and label.is_file():
            grouped[(split, dataset)].append(row)
    selected: list[dict[str, str]] = []
    for split, requirements in TARGETS.items():
        for dataset, desired in requirements.items():
            candidates = grouped[(split, dataset)]
            if len(candidates) < desired:
                raise ValueError(f"Need {desired} {dataset}/{split} labeled images, but only {len(candidates)} are eligible")
            chooser = random.Random(f"{seed}:{split}:{dataset}")
            selected.extend(chooser.sample(candidates, desired))
    selected.sort(key=lambda row: (row["split"], row["dataset"], row["image_id"]))
    return selected, image_fields


def _build_subset(
    source: Path,
    destination: Path,
    seed: int,
    ignored_by_sequence: dict[str, list[IgnoredRegion]],
) -> dict[str, Any]:
    selected, image_fields = _select_rows(source, seed)
    if destination.exists():
        raise FileExistsError(f"Run directory already exists: {destination}")
    for split in SPLITS:
        (destination / "dataset" / "images" / split).mkdir(parents=True)
        (destination / "dataset" / "labels" / split).mkdir(parents=True)
    annotation_rows, annotation_fields = _read_csv(source / "metadata" / "annotations.csv")
    selected_ids = {row["image_id"] for row in selected}
    selected_annotations = [row for row in annotation_rows if row.get("image_id") in selected_ids]
    masked_images = 0
    masked_regions = 0
    if "ignored_region_mask_applied" not in image_fields:
        image_fields.extend(["ignored_region_mask_applied", "ignored_region_count"])
    for row in selected:
        split = row["split"]
        source_image = source / row["exported_image"]
        source_label = source / row["exported_label"]
        target_image = destination / "dataset" / "images" / split / source_image.name
        target_label = destination / "dataset" / "labels" / split / source_label.name
        regions = ignored_by_sequence.get(row["sequence_id"], []) if row["dataset"] == UA_DATASET else []
        should_mask = bool(split == "train" and row["dataset"] == UA_DATASET and regions)
        if should_mask:
            target_image.write_bytes(mask_image_bytes(source_image.read_bytes(), regions, source_image.name))
            masked_images += 1
            masked_regions += len(regions)
        else:
            shutil.copy2(source_image, target_image)
        shutil.copy2(source_label, target_label)
        row["exported_image"] = target_image.relative_to(destination / "dataset").as_posix()
        row["exported_label"] = target_label.relative_to(destination / "dataset").as_posix()
        row["ignored_region_mask_applied"] = should_mask
        row["ignored_region_count"] = len(regions)
    metadata = destination / "dataset" / "metadata"
    _write_csv(metadata / "images.csv", image_fields, selected)
    _write_csv(metadata / "annotations.csv", annotation_fields, selected_annotations)
    sequence_rows = [
        {"dataset": dataset, "sequence_id": sequence, "split": split, "split_source": "parent_dataset_fixed_sequence_split"}
        for dataset, sequence, split in sorted({(row["dataset"], row["sequence_id"], row["split"]) for row in selected})
    ]
    _write_csv(metadata / "sequence_splits.csv", ["dataset", "sequence_id", "split", "split_source"], sequence_rows)
    image_counts = Counter(row["split"] for row in selected)
    box_counts = Counter(row["split"] for row in selected_annotations)
    summary = {
        "format": "smoke-test-v2-derived-yolo",
        "parent_dataset": str(source.resolve()),
        "scope": {"included_datasets": list(CORE_DATASETS), "excluded_online_data": True, "empty_labels_excluded_pending_review": True},
        "counts": {
            "input_images": len(selected), "exported_images": len(selected), "input_annotations": len(selected_annotations),
            "exported_boxes": len(selected_annotations), "rejected_annotations": 0,
            "ignored_region_masked_train_images": masked_images, "ignored_region_count_applied": masked_regions,
        },
        "images_by_split": dict(image_counts), "boxes_by_split": dict(box_counts), "sequence_count": len(sequence_rows),
    }
    (metadata / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data_yaml = {"path": (destination / "dataset").resolve().as_posix(), "train": "images/train", "val": "images/val", "test": "images/cross_test", "names": {0: "vehicle"}}
    (destination / "dataset" / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return summary


def _yolo_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            _, x, y, width, height = line.split()
            boxes.append((float(x), float(y), float(width), float(height)))
    return boxes


def _draw_boxes(draw: ImageDraw.ImageDraw, origin: tuple[int, int], size: tuple[int, int], boxes: Iterable[tuple[float, float, float, float]], color: str) -> None:
    left, top = origin
    width, height = size
    for x, y, box_width, box_height in boxes:
        draw.rectangle((left + (x - box_width / 2) * width, top + (y - box_height / 2) * height, left + (x + box_width / 2) * width, top + (y + box_height / 2) * height), outline=color, width=2)


def _render_train_batch(dataset: Path, output: Path) -> None:
    paths = sorted((dataset / "images" / "train").iterdir())[:16]
    canvas = Image.new("RGB", (4 * 300, math.ceil(len(paths) / 4) * 220), "#202020")
    draw = ImageDraw.Draw(canvas)
    for index, image_path in enumerate(paths):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image.thumbnail((288, 180))
        x = (index % 4) * 300 + (300 - image.width) // 2
        y = (index // 4) * 220 + 4
        canvas.paste(image, (x, y))
        boxes = _yolo_boxes(dataset / "labels" / "train" / f"{image_path.stem}.txt")
        _draw_boxes(draw, (x, y), image.size, boxes, "#33dd77")
        draw.text((index % 4 * 300 + 4, (index // 4 + 1) * 220 - 28), f"{image_path.stem} | labels={len(boxes)}", fill="white")
    canvas.save(output, quality=94)


def _plot_losses(results: Path, output: Path) -> dict[str, list[float]]:
    rows, _ = _read_csv(results)
    fields = ("train/box_loss", "train/cls_loss", "train/dfl_loss")
    result = {field: [float(row[field]) for row in rows] for field in fields}
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for field, label in zip(fields, ("box loss", "cls loss", "dfl loss")):
        axis.plot(range(1, len(result[field]) + 1), result[field], marker="o", label=label)
    axis.set(xlabel="Epoch", ylabel="Train loss", title="YOLO11n 500-image fine-tuning smoke: train losses")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return result


def _render_validation_predictions(
    weights: Path,
    dataset: Path,
    metadata: dict[str, dict[str, str]],
    ignored_by_sequence: dict[str, list[IgnoredRegion]],
    output: Path,
    table: Path,
    confidence: float,
    device: str,
) -> dict[str, int]:
    from ultralytics import YOLO

    paths = sorted(path for path in (dataset / "images" / "val").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    predictions = YOLO(str(weights)).predict([str(path) for path in paths], imgsz=320, conf=confidence, device=device, verbose=False)
    canvas = Image.new("RGB", (4 * 300, math.ceil(min(20, len(paths)) / 4) * 220), "#202020")
    draw = ImageDraw.Draw(canvas)
    rows: list[dict[str, Any]] = []
    totals = Counter()
    for index, (path, prediction) in enumerate(zip(paths, predictions)):
        meta = metadata[path.stem]
        regions = ignored_by_sequence.get(meta["sequence_id"], []) if meta["dataset"] == UA_DATASET else []
        raw = prediction.boxes.xyxy.cpu().numpy().tolist() if prediction.boxes is not None else []
        eligible = [box for box in raw if not bbox_center_is_ignored(tuple(map(float, box)), regions)]
        totals["raw_predictions"] += len(raw)
        totals["ignored_region_predictions"] += len(raw) - len(eligible)
        totals["eligible_predictions"] += len(eligible)
        labels = _yolo_boxes(dataset / "labels" / "val" / f"{path.stem}.txt")
        totals["labels"] += len(labels)
        rows.append({"image_id": path.stem, "dataset": meta["dataset"], "sequence_id": meta["sequence_id"], "label_count": len(labels), "raw_prediction_count": len(raw), "ignored_region_prediction_count": len(raw) - len(eligible), "eligible_prediction_count": len(eligible), "confidence": confidence})
        if index < 20:
            with Image.open(path) as source:
                image = source.convert("RGB")
            image.thumbnail((288, 180))
            x = (index % 4) * 300 + (300 - image.width) // 2
            y = (index // 4) * 220 + 4
            canvas.paste(image, (x, y))
            _draw_boxes(draw, (x, y), image.size, labels, "#33dd77")
            width, height = prediction.orig_shape[1], prediction.orig_shape[0]
            normalized = [((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height) for x1, y1, x2, y2 in eligible]
            _draw_boxes(draw, (x, y), image.size, normalized, "#ff5c5c")
            draw.text((index % 4 * 300 + 4, (index // 4 + 1) * 220 - 28), f"pred={len(eligible)} label={len(labels)}", fill="white")
    canvas.save(output, quality=94)
    _write_csv(table, ["image_id", "dataset", "sequence_id", "label_count", "raw_prediction_count", "ignored_region_prediction_count", "eligible_prediction_count", "confidence"], rows)
    return dict(totals)


def _write_report(run_dir: Path, config: dict[str, Any], subset: dict[str, Any], validation: dict[str, Any], losses: dict[str, list[float]], predictions: dict[str, int]) -> None:
    summary = f"""# Smoke Test V2 — YOLO11n 320, 500 train images, 25 epochs

This is a pipeline smoke run only. It does not report or interpret mAP because UA ignored regions are filtered in the custom prediction-count review, not in Ultralytics' default metric loop.

| Item | Result |
|---|---:|
| Train / val / cross-test | {validation['images_by_split']['train']} / {validation['images_by_split']['val']} / {validation['images_by_split']['cross_test']} images |
| Epochs / input | {config['epochs']} / {config['imgsz']}×{config['imgsz']} |
| Train images with UA ignored region black mask | {subset['counts']['ignored_region_masked_train_images']} |
| Train loss start → end (box) | {losses['train/box_loss'][0]:.4f} → {losses['train/box_loss'][-1]:.4f} |
| Validation labels | {predictions.get('labels', 0)} |
| Raw / ignored / eligible validation predictions | {predictions.get('raw_predictions', 0)} / {predictions.get('ignored_region_predictions', 0)} / {predictions.get('eligible_predictions', 0)} |

![Loss](loss_by_epoch.png)

![First training batch](train_batch0.jpg)

![Validation predictions versus labels](val_predictions_vs_labels.png)

Green boxes are labels; red boxes are metric-eligible predictions.  Predictions whose bbox centers fall in a UA `ignored_region` are counted separately and excluded from the eligible total.

Artifacts: `dataset/metadata/images.csv`, `dataset/metadata/annotations.csv`, `dataset_validation.json`, `results.csv`, `val_prediction_counts.csv`, and `weights/last.pt`.
"""
    (run_dir / "smoke_test_v2_report.md").write_text(summary, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_team_model_manifest(
    run_dir: Path,
    config: dict[str, Any],
    weights: Path,
    base_weights: Path,
    config_path: Path,
    source_dataset: Path,
) -> None:
    """Bind a fine-tuned checkpoint to its COCO base and training evidence."""
    checkpoint_sha256 = _sha256(weights)
    base_sha256 = _sha256(base_weights)
    if checkpoint_sha256 == base_sha256:
        raise RuntimeError("Final checkpoint equals base weights; rejecting model release")
    manifest = {
        "format": "team-yolo-release-v2",
        "provenance": "FINETUNED_TEAM_MODEL_REQUIRED",
        "architecture": "yolo11n",
        "pretrained": bool(config["pretrained"]),
        "class_names": {"0": "vehicle"},
        "positive_policy": "CAR_AND_APPROVED_FOUR_WHEEL_ONLY",
        "runtime": {"classes": [0], "confidence": 0.50},
        "base_weights": {
            "name": base_weights.name,
            "path": str(base_weights.resolve()),
            "sha256": base_sha256,
            "bytes": base_weights.stat().st_size,
        },
        "checkpoint": {
            "path": str(weights.resolve()),
            "sha256": checkpoint_sha256,
            "bytes": weights.stat().st_size,
        },
        "finetuned_on": {
            "dataset_path": str(source_dataset.resolve()),
            "target_class": "vehicle",
            "class_mapping": "CAR_AND_APPROVED_FOUR_WHEEL_ONLY",
            "train_images": int(config["train_images"]),
            "validation_images": int(config["validation_images"]),
            "cross_test_images": int(config["cross_test_images"]),
        },
        "training_run": {
            "run_directory": str(run_dir.resolve()),
            "config": config_path.name,
            "stage": str(config["training_stage"]),
            "imgsz": int(config["imgsz"]),
            "epochs": int(config["epochs"]),
            "seed": int(config["seed"]),
        },
    }
    (run_dir / "team_model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "data_collection" / "configs" / "yolo11n_320_smoke_v2_500.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-dataset", type=Path, help="Override the relative source_dataset configured in YAML.")
    parser.add_argument("--ua-annotation-root", type=Path, action="append", help="Override/append a UA XML root; repeat for train and test XML roots.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    source = args.source_dataset.resolve() if args.source_dataset else _resolve_config_path(config_path, str(config["source_dataset"]))
    run_dir = args.run_dir.resolve()
    annotation_roots = args.ua_annotation_root or [_resolve_config_path(config_path, str(item)) for item in config["ua_annotation_roots"]]
    ignored_by_sequence = _load_ua_ignored_regions(path.resolve() for path in annotation_roots)
    base_weights = _resolve_config_path(config_path, str(config["base_weights"]))
    if not base_weights.is_file():
        raise FileNotFoundError(f"Fine-tuning base weights not found: {base_weights}")
    started = datetime.now(timezone.utc)
    subset = _build_subset(source, run_dir, int(config["seed"]), ignored_by_sequence)
    validation = validate_dataset(run_dir / "dataset")
    (run_dir / "dataset_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if validation["status"] != "PASS":
        raise RuntimeError(f"Derived V2 subset failed QC: {validation['errors']}")
    # Persist the resolved inputs with the run.  The source YAML remains
    # portable, while report generation from another working directory must
    # not reinterpret its relative paths against that new directory.
    run_config = dict(config)
    run_config["source_dataset"] = str(source)
    run_config["base_weights"] = str(base_weights)
    run_config["ua_annotation_roots"] = [str(path.resolve()) for path in annotation_roots]
    (run_dir / "smoke_config.yaml").write_text(
        yaml.safe_dump(run_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _render_train_batch(run_dir / "dataset", run_dir / "train_batch0.jpg")
    from ultralytics import YOLO

    YOLO(str(base_weights)).train(
        data=str((run_dir / "dataset" / "data.yaml").resolve()), imgsz=int(config["imgsz"]), epochs=int(config["epochs"]),
        batch=int(config["batch"]), seed=int(config["seed"]), deterministic=bool(config["deterministic"]), workers=int(config["workers"]),
        optimizer=config["optimizer"], pretrained=bool(config["pretrained"]), plots=False, val=False, save=True,
        project=str(run_dir.parent), name=run_dir.name, exist_ok=True, device=args.device,
    )
    losses = _plot_losses(run_dir / "results.csv", run_dir / "loss_by_epoch.png")
    weights = run_dir / "weights" / "last.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Expected trained weights were not written: {weights}")
    _write_team_model_manifest(run_dir, config, weights, base_weights, config_path, source)
    images, _ = _read_csv(run_dir / "dataset" / "metadata" / "images.csv")
    metadata = {row["image_id"]: row for row in images}
    predictions = _render_validation_predictions(weights, run_dir / "dataset", metadata, ignored_by_sequence, run_dir / "val_predictions_vs_labels.png", run_dir / "val_prediction_counts.csv", float(config["prediction_confidence"]), args.device)
    ended = datetime.now(timezone.utc)
    run_parameters = {"started_at_utc": started.isoformat(), "finished_at_utc": ended.isoformat(), "wall_seconds": (ended - started).total_seconds(), "config": config, "subset": subset, "validation": validation, "prediction_counts": predictions}
    (run_dir / "run_parameters.json").write_text(json.dumps(run_parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(run_dir, config, subset, validation, losses, predictions)
    print(json.dumps(run_parameters, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
