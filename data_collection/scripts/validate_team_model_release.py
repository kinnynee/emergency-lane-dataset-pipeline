"""Validate that a K230 release uses the team model and has board evidence.

This does not emulate a board. It deliberately reports ``NOT_MEASURED`` until
the real K230 has loaded the compiled model and produced inference logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_LOG_MARKERS = ("MODEL_LOAD_OK", "INFERENCE_OK")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_release(manifest_path: Path, kmodel_path: Path, board_log: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "release_status": "BLOCKED_BOARD_RUN_REQUIRED",
        "team_model": "NOT_VERIFIED",
        "kmodel": "NOT_VERIFIED",
        "board_load": "NOT_MEASURED",
        "board_inference": "NOT_MEASURED",
        "reason": [],
    }
    if not manifest_path.is_file():
        result["reason"].append("MISSING_TEAM_MODEL_MANIFEST")
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        result["reason"].append("INVALID_TEAM_MODEL_MANIFEST_JSON")
        return result
    if not isinstance(manifest, dict):
        result["reason"].append("INVALID_TEAM_MODEL_MANIFEST")
        return result
    expected = {
        "format": "team-yolo-release-v1",
        "provenance": "TEAM_TRAINED_ONLY_NO_COCO_FALLBACK",
        "architecture": "yolo11n.yaml",
        "pretrained": False,
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    runtime = manifest.get("runtime")
    class_names = manifest.get("class_names")
    if mismatched or runtime != {"classes": [0], "confidence": 0.50} or class_names != {"0": "vehicle"}:
        result["reason"].append("TEAM_MODEL_POLICY_MISMATCH:" + ",".join(mismatched or ["runtime_or_classes"]))
        return result
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict):
        result["reason"].append("MISSING_CHECKPOINT_PROVENANCE")
        return result
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != checkpoint.get("sha256"):
        result["reason"].append("CHECKPOINT_PROVENANCE_NOT_REPRODUCIBLE")
        return result
    result["team_model"] = "VERIFIED"
    if not kmodel_path.is_file() or kmodel_path.stat().st_size <= 0:
        result["reason"].append("MISSING_OR_EMPTY_KMODEL")
        return result
    result["kmodel"] = "COMPILED_ARTIFACT_PRESENT"
    if board_log is None or not board_log.is_file():
        result["reason"].append("MISSING_K230_BOARD_LOG")
        return result
    log_text = board_log.read_text(encoding="utf-8", errors="replace")
    missing_markers = [marker for marker in REQUIRED_LOG_MARKERS if marker not in log_text]
    if missing_markers:
        result["reason"].append("MISSING_BOARD_LOG_MARKERS:" + ",".join(missing_markers))
        return result
    result["board_load"] = "VERIFIED"
    result["board_inference"] = "VERIFIED"
    result["release_status"] = "READY_FOR_LOCKED_K230_TEST"
    result["reason"].append("K230_METRICS_REMAIN_NOT_MEASURED_UNTIL_LOCKED_TEST_SESSIONS_ARE_EVALUATED")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-model-manifest", required=True, type=Path)
    parser.add_argument("--kmodel", required=True, type=Path)
    parser.add_argument("--board-log", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = validate_release(args.team_model_manifest.resolve(), args.kmodel.resolve(), args.board_log.resolve() if args.board_log else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["release_status"] == "READY_FOR_LOCKED_K230_TEST" else 2


if __name__ == "__main__":
    raise SystemExit(main())
