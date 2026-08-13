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


def _validate_deployment_contract(contract_path: Path | None, manifest_path: Path, kmodel_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if contract_path is None or not contract_path.is_file():
        return None, "MISSING_K230_DEPLOYMENT_CONTRACT"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, "INVALID_K230_DEPLOYMENT_CONTRACT_JSON"
    if not isinstance(contract, dict) or contract.get("format") != "team-k230-deployment-v1":
        return None, "INVALID_K230_DEPLOYMENT_CONTRACT"
    expected_input = {"width": 320, "height": 320, "layout": "NCHW", "type": "uint8", "color": "RGB"}
    expected_detection = {"class_names": {"0": "vehicle"}, "confidence": 0.50, "nms_iou": 0.50}
    model = contract.get("model")
    if (
        contract.get("team_model_manifest_sha256") != _sha256(manifest_path)
        or contract.get("input") != expected_input
        or contract.get("detection") != expected_detection
        or not isinstance(model, dict)
        or model.get("kmodel_sha256") != _sha256(kmodel_path)
    ):
        return None, "K230_DEPLOYMENT_CONTRACT_MISMATCH"
    return contract, None


def validate_release(
    manifest_path: Path,
    kmodel_path: Path,
    board_log: Path | None,
    deployment_contract: Path | None = None,
) -> dict[str, Any]:
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
        "format": "team-yolo-release-v2",
        "provenance": "FINETUNED_TEAM_MODEL_REQUIRED",
        "architecture": "yolo11n",
        "pretrained": True,
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    runtime = manifest.get("runtime")
    class_names = manifest.get("class_names")
    if mismatched or runtime != {"classes": [0], "confidence": 0.50} or class_names != {"0": "vehicle"}:
        result["reason"].append("TEAM_MODEL_POLICY_MISMATCH:" + ",".join(mismatched or ["runtime_or_classes"]))
        return result
    checkpoint = manifest.get("checkpoint")
    base_weights = manifest.get("base_weights")
    finetuned_on = manifest.get("finetuned_on")
    training_run = manifest.get("training_run")
    if not isinstance(checkpoint, dict) or not isinstance(base_weights, dict) or not isinstance(finetuned_on, dict) or not isinstance(training_run, dict):
        result["reason"].append("MISSING_CHECKPOINT_PROVENANCE")
        return result
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    base_weights_path = Path(str(base_weights.get("path", "")))
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != checkpoint.get("sha256"):
        result["reason"].append("CHECKPOINT_PROVENANCE_NOT_REPRODUCIBLE")
        return result
    if not base_weights_path.is_file() or _sha256(base_weights_path) != base_weights.get("sha256"):
        result["reason"].append("BASE_WEIGHTS_PROVENANCE_NOT_REPRODUCIBLE")
        return result
    if checkpoint.get("sha256") == base_weights.get("sha256"):
        result["reason"].append("FINAL_CHECKPOINT_EQUALS_BASE_WEIGHTS")
        return result
    required_finetuning = {"dataset_path", "target_class", "class_mapping", "train_images"}
    required_run = {"run_directory", "config", "stage", "imgsz", "epochs", "seed"}
    if not required_finetuning.issubset(finetuned_on) or not required_run.issubset(training_run):
        result["reason"].append("INCOMPLETE_FINETUNING_EVIDENCE")
        return result
    result["team_model"] = "VERIFIED"
    if not kmodel_path.is_file() or kmodel_path.stat().st_size <= 0:
        result["reason"].append("MISSING_OR_EMPTY_KMODEL")
        return result
    result["kmodel"] = "COMPILED_ARTIFACT_PRESENT"
    contract, contract_error = _validate_deployment_contract(deployment_contract, manifest_path, kmodel_path)
    if contract_error:
        result["reason"].append(contract_error)
        return result
    if board_log is None or not board_log.is_file():
        result["reason"].append("MISSING_K230_BOARD_LOG")
        return result
    log_text = board_log.read_text(encoding="utf-8", errors="replace")
    missing_markers = [marker for marker in REQUIRED_LOG_MARKERS if marker not in log_text]
    assert contract is not None  # validated above; keeps the required hashes explicit.
    expected_hash_markers = [
        f"TEAM_MODEL_MANIFEST_SHA256={contract['team_model_manifest_sha256']}",
        f"KMODEL_SHA256={contract['model']['kmodel_sha256']}",
    ]
    missing_markers.extend(marker for marker in expected_hash_markers if marker not in log_text)
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
    parser.add_argument("--deployment-contract", required=True, type=Path)
    parser.add_argument("--board-log", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = validate_release(
        args.team_model_manifest.resolve(), args.kmodel.resolve(), args.board_log.resolve() if args.board_log else None,
        args.deployment_contract.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["release_status"] == "READY_FOR_LOCKED_K230_TEST" else 2


if __name__ == "__main__":
    raise SystemExit(main())
