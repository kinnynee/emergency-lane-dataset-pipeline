from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from export_ua_detrac_yolo import export_ua_detrac
from create_k230_deployment_contract import build_contract
from validate_k230_evaluation_readiness import validate_sessions
from validate_team_model_release import validate_release


REPO_ROOT = Path(__file__).resolve().parents[2]
MAPPING = REPO_ROOT / "data_collection" / "configs" / "vehicle_class_mapping.yaml"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _make_ua_zip(path: Path) -> None:
    image_path = "DETRAC-Images/DETRAC-Images/MVI_40172/img00001.jpg"
    image_path_2 = "DETRAC-Images/DETRAC-Images/MVI_40172/img00002.jpg"
    temp_image = path.parent / "source.jpg"
    Image.new("RGB", (100, 50), "gray").save(temp_image)
    xml = """<sequence name="MVI_40172">
  <frame num="1"><target_list>
    <target id="1"><box left="90" top="10" width="11" height="20"/><attribute vehicle_type="car"/></target>
    <target id="79"><box left="10" top="5" width="20" height="20"/><attribute vehicle_type="others"/></target>
    <target id="3"><box left="110" top="10" width="5" height="10"/><attribute vehicle_type="car"/></target>
  </target_list></frame>
  <frame num="2"><target_list/></frame>
</sequence>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(temp_image, image_path)
        archive.write(temp_image, image_path_2)
        archive.writestr("DETRAC-Train-Annotations-XML/MVI_40172.xml", xml)


def test_ua_export_clips_keeps_original_class_and_applies_track_exclusion(tmp_path: Path) -> None:
    ua_zip = tmp_path / "ua.zip"
    _make_ua_zip(ua_zip)
    split = tmp_path / "split.csv"
    split.write_text(
        "dataset_name,sequence_id,proposed_split\n"
        "UA-DETRAC Original,MVI_40172,EXTERNAL_TRAIN\n",
        encoding="utf-8",
    )
    output = tmp_path / "yolo"
    summary = export_ua_detrac(ua_zip, output, split, MAPPING)

    label = output / "labels" / "train" / "UA_MVI_40172_00001.txt"
    assert label.read_text(encoding="utf-8") == "0 0.95000000 0.40000000 0.10000000 0.40000000\n"
    assert (output / "labels" / "train" / "UA_MVI_40172_00002.txt").read_text() == ""

    annotations = _rows(output / "metadata" / "annotations.csv")
    assert len(annotations) == 1
    assert annotations[0]["original_class"] == "car"
    assert annotations[0]["mapped_class"] == "vehicle"
    assert annotations[0]["clip_applied"] == "True"
    assert annotations[0]["preserve_original_class"] == "True"

    rejected = _rows(output / "metadata" / "rejected_annotations.csv")
    assert {row["action"] for row in rejected} == {
        "EXCLUDE_NON_VEHICLE_TRACK",
        "EXCLUDE_INVALID_AFTER_CLIP",
    }
    assert summary["counts"]["clipped_boxes"] == 1
    assert summary["counts"]["excluded_track_boxes"] == 1
    assert summary["counts"]["invalid_after_clip"] == 1
    assert json.loads((output / "export_summary.json").read_text())["preserve_original_class"] is True


def test_k230_readiness_counts_only_locked_approved_self_recorded_sessions(tmp_path: Path) -> None:
    images = tmp_path / "day" / "images"
    labels = tmp_path / "day" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (16, 16), "gray").save(images / "frame001.jpg")
    (labels / "frame001.txt").write_text("", encoding="utf-8")
    manifest = tmp_path / "sessions.csv"
    manifest.write_text(
        "session_id,slice,source,split,locked,ground_truth_status,images_dir,labels_dir,predictions_path,notes\n"
        f"DAY_001,DAY,K230_SELF_RECORDED,MAIN_K230_TEST,TRUE,APPROVED,{images},{labels},,ready gt\n"
        f"FAKE_BACKLIT,BACKLIT,OPEN_DATASET,MAIN_K230_TEST,TRUE,APPROVED,{images},{labels},,must not count\n",
        encoding="utf-8",
    )
    rows, diagnostics = validate_sessions(manifest, tmp_path)
    by_slice = {row["slice"]: row for row in rows}
    assert by_slice["DAY"]["status"] == "READY_FOR_INFERENCE"
    assert by_slice["DAY"]["current_evaluable_sequences"] == 1
    assert by_slice["DAY"]["current_map"] == "NOT_AVAILABLE"
    assert by_slice["BACKLIT"]["status"] == "BLOCKED_MISSING_DATA"
    assert by_slice["BACKLIT"]["current_evaluable_sequences"] == 0
    assert any("NOT_K230_SELF_RECORDED" in row["reason"] for row in diagnostics)


def test_release_rejects_a_checkpoint_with_the_same_hash_as_base_weights(tmp_path: Path) -> None:
    base = tmp_path / "yolo11n.pt"
    checkpoint = tmp_path / "last.pt"
    kmodel = tmp_path / "team.kmodel"
    base.write_bytes(b"same weights are not a fine-tuned release")
    checkpoint.write_bytes(base.read_bytes())
    kmodel.write_bytes(b"compiled")
    import hashlib

    digest = hashlib.sha256(base.read_bytes()).hexdigest()
    manifest = tmp_path / "team_model_manifest.json"
    manifest.write_text(json.dumps({
        "format": "team-yolo-release-v2",
        "provenance": "FINETUNED_TEAM_MODEL_REQUIRED",
        "architecture": "yolo11n",
        "pretrained": True,
        "class_names": {"0": "vehicle"},
        "runtime": {"classes": [0], "confidence": 0.50},
        "base_weights": {"path": str(base), "sha256": digest},
        "checkpoint": {"path": str(checkpoint), "sha256": digest},
        "finetuned_on": {
            "dataset_path": "dataset_output/dataset-v1-full",
            "target_class": "vehicle",
            "class_mapping": "CAR_AND_APPROVED_FOUR_WHEEL_ONLY",
            "train_images": 500,
        },
        "training_run": {
            "run_directory": "runs/smoke",
            "config": "yolo11n_320_smoke_v2_500.yaml",
            "stage": "SMOKE_PIPELINE_ONLY",
            "imgsz": 320,
            "epochs": 25,
            "seed": 230,
        },
    }), encoding="utf-8")

    result = validate_release(manifest, kmodel, None)

    assert result["release_status"] == "BLOCKED_BOARD_RUN_REQUIRED"
    assert "FINAL_CHECKPOINT_EQUALS_BASE_WEIGHTS" in result["reason"]


def test_release_requires_a_hash_bound_k230_contract_and_board_log(tmp_path: Path) -> None:
    import hashlib

    base = tmp_path / "yolo11n.pt"
    checkpoint = tmp_path / "best.pt"
    kmodel = tmp_path / "team_yolo11n_320.kmodel"
    base.write_bytes(b"COCO base")
    checkpoint.write_bytes(b"fine tuned team checkpoint")
    kmodel.write_bytes(b"compiled team kmodel")
    manifest = tmp_path / "team_model_manifest.json"
    manifest.write_text(json.dumps({
        "format": "team-yolo-release-v2",
        "provenance": "FINETUNED_TEAM_MODEL_REQUIRED",
        "architecture": "yolo11n",
        "pretrained": True,
        "class_names": {"0": "vehicle"},
        "runtime": {"classes": [0], "confidence": 0.50},
        "base_weights": {"path": str(base), "sha256": hashlib.sha256(base.read_bytes()).hexdigest()},
        "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        "finetuned_on": {"dataset_path": "dataset_output/dataset-v1-full", "target_class": "vehicle", "class_mapping": "CAR_AND_APPROVED_FOUR_WHEEL_ONLY", "train_images": 1200},
        "training_run": {"run_directory": "runs/final", "config": "yolo11n_320_final.yaml", "stage": "FINAL_FULL_DATASET", "imgsz": 320, "epochs": 100, "seed": 230},
    }), encoding="utf-8")
    contract = tmp_path / "team_deployment_contract.json"
    payload = build_contract(manifest, kmodel, "/sdcard/model/team_yolo11n_320.kmodel")
    contract.write_text(json.dumps(payload), encoding="utf-8")
    log = tmp_path / "board.log"
    log.write_text(
        "MODEL_LOAD_OK\n"
        f"TEAM_MODEL_MANIFEST_SHA256={payload['team_model_manifest_sha256']}\n"
        f"KMODEL_SHA256={payload['model']['kmodel_sha256']}\n"
        "INFERENCE_OK\n",
        encoding="utf-8",
    )

    result = validate_release(manifest, kmodel, log, contract)

    assert result["release_status"] == "READY_FOR_LOCKED_K230_TEST"
    payload["detection"]["confidence"] = 0.25
    contract.write_text(json.dumps(payload), encoding="utf-8")
    assert "K230_DEPLOYMENT_CONTRACT_MISMATCH" in validate_release(manifest, kmodel, log, contract)["reason"]
