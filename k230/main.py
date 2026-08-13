"""CanMV K230 team-model bootstrap; fail closed when the release is invalid.

This board entry point deliberately uses only CanMV modules. It never imports
Ultralytics, ByteTrack, OpenCV, or NumPy and never substitutes a COCO model.
The camera/inference loop must only start after ``load_team_model`` succeeds.
"""

import os
import ujson


CONTRACT_PATH = "/sdcard/emergency_lane/runtime_contract.json"


def load_contract(path=CONTRACT_PATH):
    with open(path, "r") as handle:
        contract = ujson.load(handle)
    required = {
        "format": "k230-team-model-runtime-v1",
        "provenance": "TEAM_TRAINED_ONLY_NO_COCO_FALLBACK",
        "input_size": 320,
        "class_ids": [0],
        "confidence": 0.5,
        "output_decoder": "YOLO11_DETECT_ONE_CLASS",
    }
    for key in required:
        if contract.get(key) != required[key]:
            raise RuntimeError("RUNTIME_CONTRACT_MISMATCH:" + key)
    return contract


def load_team_model(contract):
    model_path = contract.get("kmodel_path", "")
    if not model_path or not model_path.endswith(".kmodel") or model_path.find("coco") >= 0:
        raise RuntimeError("INVALID_TEAM_KMODEL_PATH")
    try:
        os.stat(model_path)
    except OSError:
        raise RuntimeError("TEAM_KMODEL_NOT_FOUND:" + model_path)
    import nncase_runtime as nn
    kpu = nn.kpu()
    kpu.load_kmodel(model_path)
    print("MODEL_LOAD_OK", model_path)
    return kpu


def main():
    contract = load_contract()
    load_team_model(contract)
    print("BOARD_RUNTIME_READY")
    print("INFERENCE_NOT_STARTED: integrate camera tensor/preprocess/postprocess before release")


if __name__ == "__main__":
    main()
