"""Fail-closed deployment-contract loader for CanMV/K230.

Copy this file next to the K230 application's ``main.py``.  It intentionally
uses only modules present on normal MicroPython/CanMV builds.  It does not
import Ultralytics, ByteTrack, NumPy or OpenCV.
"""

try:  # CanMV ships ujson; CPython support makes the module testable.
    import ujson as json
except ImportError:  # pragma: no cover - CPython fallback
    import json

try:
    import uhashlib as hashlib
except ImportError:  # pragma: no cover - CPython fallback
    import hashlib

try:
    import uos as os
except ImportError:  # pragma: no cover - CPython fallback
    import os


class DeploymentRejected(Exception):
    """Raised before KPU initialization when the requested model is unsafe."""


def _sha256_file(path):
    try:
        digest = hashlib.sha256()
    except AttributeError:
        raise DeploymentRejected("SHA256_UNAVAILABLE")
    with open(path, "rb") as source:
        while True:
            chunk = source.read(4096)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require(value, description):
    if not value:
        raise DeploymentRejected("INVALID_DEPLOYMENT_CONTRACT:" + description)
    return value


def load_team_deployment(contract_path="/sdcard/model/team_deployment_contract.json"):
    """Validate the exact K230 artifact and return its runtime configuration.

    ``main.py`` must catch :class:`DeploymentRejected`, print
    ``MODEL_LOAD_REJECTED`` and stop.  It must never substitute COCO labels,
    a generic YOLO model or a default ``.kmodel``.
    """
    try:
        with open(contract_path, "r") as source:
            contract = json.load(source)
    except Exception as exc:
        raise DeploymentRejected("CONTRACT_UNREADABLE:" + str(exc))
    _require(isinstance(contract, dict), "root")
    _require(contract.get("format") == "team-k230-deployment-v1", "format")
    model = _require(contract.get("model"), "model")
    input_spec = _require(contract.get("input"), "input")
    detection = _require(contract.get("detection"), "detection")
    _require(input_spec == {"width": 320, "height": 320, "layout": "NCHW", "type": "uint8", "color": "RGB"}, "input")
    _require(detection.get("class_names") == {"0": "vehicle"}, "classes")
    _require(float(detection.get("confidence", -1)) == 0.50, "confidence")
    _require(float(detection.get("nms_iou", -1)) == 0.50, "nms_iou")
    model_path = _require(model.get("board_path"), "model.board_path")
    expected_hash = _require(model.get("kmodel_sha256"), "model.kmodel_sha256")
    try:
        os.stat(model_path)
    except OSError:
        raise DeploymentRejected("KMODEL_MISSING:" + model_path)
    actual_hash = _sha256_file(model_path)
    if actual_hash != expected_hash:
        raise DeploymentRejected("KMODEL_SHA256_MISMATCH")
    _require(contract.get("team_model_manifest_sha256"), "team_model_manifest_sha256")
    return contract
