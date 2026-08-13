"""Bind a compiled K230 model to one verified fine-tuned team checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_contract(manifest_path: Path, kmodel_path: Path, board_kmodel_path: str) -> dict[str, Any]:
    """Return the immutable Host/K230 parameters for this exact model pair."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != "team-yolo-release-v2":
        raise ValueError("The deployment contract requires a team-yolo-release-v2 manifest")
    if manifest.get("runtime") != {"classes": [0], "confidence": 0.50}:
        raise ValueError("The checkpoint manifest runtime is not the locked one-class 0.50 contract")
    if manifest.get("class_names") != {"0": "vehicle"}:
        raise ValueError("The checkpoint manifest does not describe the one-class vehicle model")
    if not board_kmodel_path.startswith("/sdcard/"):
        raise ValueError("--board-kmodel-path must be an absolute /sdcard/ path")
    return {
        "format": "team-k230-deployment-v1",
        "team_model_manifest_sha256": _sha256(manifest_path),
        "model": {
            "board_path": board_kmodel_path,
            "kmodel_sha256": _sha256(kmodel_path),
            "kmodel_bytes": kmodel_path.stat().st_size,
        },
        "input": {"width": 320, "height": 320, "layout": "NCHW", "type": "uint8", "color": "RGB"},
        "detection": {"class_names": {"0": "vehicle"}, "confidence": 0.50, "nms_iou": 0.50},
        "host_alert": {
            "owner": "host_yolo_loop.py",
            "median_speed_window_sec": 1.5,
            "enter_stop_kmh": 2.0,
            "enter_stop_seconds": 3.0,
            "exit_resume_kmh": 3.0,
            "exit_resume_seconds": 1.0,
        },
        "board_log": {
            "required_markers": ["MODEL_LOAD_OK", "INFERENCE_OK"],
            "manifest_hash_marker": "TEAM_MODEL_MANIFEST_SHA256",
            "kmodel_hash_marker": "KMODEL_SHA256",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-model-manifest", required=True, type=Path)
    parser.add_argument("--kmodel", required=True, type=Path)
    parser.add_argument("--board-kmodel-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest_path, kmodel_path, output = args.team_model_manifest.resolve(), args.kmodel.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite deployment contract: {output}")
    if not manifest_path.is_file() or not kmodel_path.is_file():
        raise FileNotFoundError("Both --team-model-manifest and --kmodel must exist")
    payload = build_contract(manifest_path, kmodel_path, args.board_kmodel_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
