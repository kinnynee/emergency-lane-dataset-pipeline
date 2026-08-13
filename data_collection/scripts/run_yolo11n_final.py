"""Fine-tune the release YOLO11n model on the complete validated export.

This is intentionally distinct from the 500-image smoke runner.  It points
Ultralytics at all train/val/cross_test folders from the validated export,
records the COCO base-weight hash, and produces the manifest required before
ONNX/K230 compilation.  Raw datasets and trained weights remain outside Git.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from run_yolo11n_smoke_v2 import (
    REPO_ROOT,
    _load_ua_ignored_regions,
    _resolve_config_path,
    _write_team_model_manifest,
)
from validate_yolo_dataset import validate_dataset


def _load_final_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {
        "model", "base_weights", "model_variant", "training_stage", "imgsz", "epochs", "patience",
        "batch", "seed", "deterministic", "workers", "optimizer", "pretrained", "plots",
        "prediction_confidences", "source_dataset", "ua_annotation_roots", "model_provenance_policy",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("Final config missing fields: " + ", ".join(missing))
    if config["model_variant"] != "YOLO11n" or not bool(config["pretrained"]):
        raise ValueError("Final training must fine-tune a YOLO11n COCO base checkpoint")
    if config["training_stage"] != "FINAL_FULL_DATASET":
        raise ValueError("Final runner accepts only training_stage=FINAL_FULL_DATASET")
    if config["model_provenance_policy"] != "FINETUNED_TEAM_MODEL_REQUIRED":
        raise ValueError("Missing FINETUNED_TEAM_MODEL_REQUIRED provenance policy")
    if (int(config["imgsz"]), int(config["epochs"]), int(config["patience"]), int(config["seed"])) != (320, 100, 20, 230):
        raise ValueError("Final training requires imgsz=320, epochs=100, patience=20, seed=230")
    if tuple(float(value) for value in config["prediction_confidences"]) != (0.0, 0.50):
        raise ValueError("Final report must evaluate confidence 0.00 and 0.50")
    return config


def _data_yaml(source: Path, run_dir: Path) -> Path:
    """Write a portable Ultralytics data file that references every split."""
    payload = {
        "path": str(source.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/cross_test",
        "names": {0: "vehicle"},
    }
    destination = run_dir / "final_data.yaml"
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "data_collection" / "configs" / "yolo11n_320_final.yaml")
    parser.add_argument("--run-dir", required=True, type=Path, help="New output directory; never an existing run.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-dataset", type=Path, help="Override source_dataset from config.")
    parser.add_argument("--ua-annotation-root", type=Path, action="append", help="Override/append UA XML roots for ignore-region validation.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _load_final_config(config_path)
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing final run: {run_dir}")
    source = args.source_dataset.resolve() if args.source_dataset else _resolve_config_path(config_path, str(config["source_dataset"]))
    validation = validate_dataset(source)
    if validation["status"] != "PASS":
        raise RuntimeError(f"Full export failed QC: {validation['errors']}")
    base_weights = _resolve_config_path(config_path, str(config["base_weights"]))
    if not base_weights.is_file():
        raise FileNotFoundError(f"Fine-tuning base weights not found: {base_weights}")
    annotation_roots = args.ua_annotation_root or [_resolve_config_path(config_path, str(item)) for item in config["ua_annotation_roots"]]
    _load_ua_ignored_regions(path.resolve() for path in annotation_roots)

    run_dir.mkdir(parents=True)
    data_yaml = _data_yaml(source, run_dir)
    # Store resolved paths in the immutable run record.  This lets metric and
    # report commands run from a clean clone or CI working directory.
    run_config = dict(config)
    run_config["source_dataset"] = str(source)
    run_config["base_weights"] = str(base_weights)
    run_config["ua_annotation_roots"] = [str(path.resolve()) for path in annotation_roots]
    (run_dir / "training_config.yaml").write_text(
        yaml.safe_dump(run_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    started = datetime.now(timezone.utc)
    from ultralytics import YOLO

    YOLO(str(base_weights)).train(
        data=str(data_yaml), imgsz=320, epochs=100, patience=20, batch=int(config["batch"]),
        seed=230, deterministic=bool(config["deterministic"]), workers=int(config["workers"]),
        optimizer=config["optimizer"], pretrained=True, plots=False, save=True, project=str(run_dir.parent),
        name=run_dir.name, exist_ok=True, device=args.device,
    )
    checkpoint = run_dir / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected final checkpoint was not written: {checkpoint}")
    _write_team_model_manifest(run_dir, config, checkpoint, base_weights, config_path, source)
    ended = datetime.now(timezone.utc)
    run_parameters: dict[str, Any] = {
        "started_at_utc": started.isoformat(), "finished_at_utc": ended.isoformat(),
        "wall_seconds": (ended - started).total_seconds(), "training_config": config,
        "source_dataset": str(source), "dataset_validation": validation,
        "report_confidences": [0.0, 0.50],
    }
    (run_dir / "run_parameters.json").write_text(json.dumps(run_parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_parameters, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
