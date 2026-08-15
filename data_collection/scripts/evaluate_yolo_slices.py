"""Evaluate a release checkpoint separately on AAU and UA-DETRAC.

The final release must not report a combined headline score.  This module
creates one immutable image list and Ultralytics data file per source slice,
then records one result for each requested confidence threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


REQUIRED_DATASETS = ("AAU RainSnow", "UA-DETRAC Original")
REQUIRED_CONFIDENCES = (0.0, 0.50)
VALID_SPLITS = ("val", "cross_test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_slice_manifest(source_dataset: Path, destination: Path, dataset_name: str, split: str) -> tuple[Path, Path, int]:
    """Create a source-specific Ultralytics validation input without copying data."""
    if dataset_name not in REQUIRED_DATASETS:
        raise ValueError(f"Unsupported release evaluation dataset: {dataset_name}")
    if split not in VALID_SPLITS:
        raise ValueError(f"Release evaluation split must be one of {VALID_SPLITS}")
    rows = _read_rows(source_dataset / "metadata" / "images.csv")
    selected = [row for row in rows if row.get("dataset") == dataset_name and row.get("split") == split]
    if not selected:
        raise ValueError(f"No {dataset_name}/{split} rows in {source_dataset / 'metadata' / 'images.csv'}")
    image_paths: list[Path] = []
    for row in selected:
        image = source_dataset / str(row.get("exported_image", ""))
        label = source_dataset / str(row.get("exported_label", ""))
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Evaluation pair missing for {dataset_name}/{split}: {image} / {label}")
        image_paths.append(image.resolve())
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{_slug(dataset_name)}_{split}"
    image_list = destination / f"{stem}_images.txt"
    data_yaml = destination / f"{stem}.yaml"
    if image_list.exists() or data_yaml.exists():
        raise FileExistsError(f"Refusing to overwrite release evaluation input: {image_list}")
    image_list.write_text("\n".join(str(path) for path in image_paths) + "\n", encoding="utf-8")
    payload = {
        "path": str(source_dataset.resolve()),
        "train": str(image_list),
        "val": str(image_list),
        "test": str(image_list),
        "names": {0: "vehicle"},
    }
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return image_list, data_yaml, len(image_paths)


def _metrics_payload(metrics: Any) -> dict[str, float]:
    box = metrics.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def evaluate_separate_slices(
    checkpoint: Path,
    source_dataset: Path,
    run_dir: Path,
    *,
    split: str,
    confidences: Iterable[float],
    device: str,
) -> Path:
    """Run the four required AAU/UA-DETRAC confidence/slice evaluations."""
    confidence_values = tuple(float(value) for value in confidences)
    if confidence_values != REQUIRED_CONFIDENCES:
        raise ValueError(f"Release evaluation requires confidence values {REQUIRED_CONFIDENCES}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Release checkpoint not found: {checkpoint}")
    output = run_dir / "separate_evaluation"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite release evaluation output: {output}")
    inputs = output / "inputs"
    output.mkdir(parents=True)
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ultralytics is required for separate release evaluation") from exc
    model = YOLO(str(checkpoint))
    results: list[dict[str, Any]] = []
    for dataset_name in REQUIRED_DATASETS:
        image_list, data_yaml, image_count = build_slice_manifest(source_dataset, inputs, dataset_name, split)
        for confidence in confidence_values:
            name = f"{_slug(dataset_name)}_{split}_conf_{confidence:.2f}"
            metrics = model.val(
                data=str(data_yaml),
                split="val",
                imgsz=320,
                conf=confidence,
                device=device,
                workers=0,
                plots=False,
                save_json=False,
                project=str(output / "ultralytics"),
                name=name,
                exist_ok=False,
                verbose=False,
            )
            results.append(
                {
                    "dataset": dataset_name,
                    "split": split,
                    "confidence": confidence,
                    "image_count": image_count,
                    "input_images": str(image_list),
                    "data_yaml": str(data_yaml),
                    **_metrics_payload(metrics),
                }
            )
    report = {
        "format": "separate-yolo-release-evaluation-v1",
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": _sha256(checkpoint)},
        "source_dataset": str(source_dataset.resolve()),
        "required_datasets": list(REQUIRED_DATASETS),
        "split": split,
        "confidences": list(confidence_values),
        "combined_headline_metric": "PROHIBITED",
        "results": results,
    }
    report_path = output / "evaluation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", choices=VALID_SPLITS, default="cross_test")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = evaluate_separate_slices(
        args.checkpoint.resolve(),
        args.source_dataset.resolve(),
        args.run_dir.resolve(),
        split=args.split,
        confidences=REQUIRED_CONFIDENCES,
        device=args.device,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
