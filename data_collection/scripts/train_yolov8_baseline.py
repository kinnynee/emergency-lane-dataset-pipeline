"""Run the fixed YOLO11n 320 baseline after dataset validation passes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from validate_yolo_dataset import validate_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "yolo11n_320_baseline.yaml"
DEFAULT_DATASET = ROOT / "dataset_output" / "dataset-v0.1" / "PILOT_500"


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {"model", "imgsz", "epochs", "batch", "seed", "deterministic", "workers", "project", "name"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Baseline config is missing: {', '.join(missing)}")
    if int(config["imgsz"]) != 320:
        raise ValueError("This baseline is locked to imgsz=320 for the K230 target")
    return config


def train_baseline(dataset: Path, config_path: Path, device: str | None = None, epochs: int | None = None) -> Path:
    report = validate_dataset(dataset)
    if report["status"] != "PASS":
        raise RuntimeError("Dataset QC failed; run validate_yolo_dataset.py and resolve its errors before training")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install data_collection/requirements-training.txt before training") from exc
    config = _load_config(config_path)
    if epochs is not None and epochs <= 0:
        raise ValueError("--epochs must be a positive integer")
    run_arguments = {
        "data": str((dataset / "data.yaml").resolve()),
        "imgsz": int(config["imgsz"]), "epochs": int(epochs or config["epochs"]), "batch": int(config["batch"]),
        "seed": int(config["seed"]), "deterministic": bool(config["deterministic"]), "workers": int(config["workers"]),
        "optimizer": config.get("optimizer", "auto"), "pretrained": bool(config.get("pretrained", True)),
        "patience": int(config.get("patience", 15)), "save": bool(config.get("save", True)),
        "project": str((ROOT / str(config["project"])).resolve()), "name": str(config["name"]), "exist_ok": False,
    }
    if device:
        run_arguments["device"] = device
    model = YOLO(str(config["model"]))
    model.train(**run_arguments)
    save_dir = Path(model.trainer.save_dir)
    run_record = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "baseline_config": config,
        "dataset": str(dataset.resolve()), "dataset_validation": report, "ultralytics_arguments": run_arguments,
    }
    (save_dir / "run_parameters.json").write_text(json.dumps(run_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (save_dir / "data.yaml").write_text((dataset / "data.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (save_dir / "baseline_config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return save_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="Optional Ultralytics device, such as 0 or cpu")
    parser.add_argument("--epochs", type=int, help="Optional short smoke-run override; omit for the locked baseline epoch count")
    args = parser.parse_args()
    try:
        save_dir = train_baseline(args.dataset.resolve(), args.config.resolve(), args.device, args.epochs)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Training complete: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
