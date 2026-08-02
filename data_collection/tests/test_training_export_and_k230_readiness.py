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
from validate_k230_evaluation_readiness import validate_sessions


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
