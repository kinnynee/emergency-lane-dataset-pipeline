from __future__ import annotations

import csv
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from export_unified_yolo import export_unified_yolo
from ignored_region import filter_boxes_by_ignored_region
from analyze_yolo_bbox_sizes import analyze_bbox_sizes
from extract_condition_frames import _conditions
from validate_pilot_500 import validate_pilot_500
from validate_yolo_dataset import validate_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
MAPPING = REPO_ROOT / "data_collection" / "configs" / "vehicle_class_mapping.yaml"


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 50), "gray").save(buffer, format="JPEG")
    return buffer.getvalue()


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 40), "gray").save(buffer, format="PNG")
    return buffer.getvalue()


def _make_mio(path: Path) -> None:
    image = _jpeg_bytes()
    with tarfile.open(path, "w") as archive:
        for name in ("MIO/train/mio_1.jpg", "MIO/train/mio_negative.jpg"):
            info = tarfile.TarInfo(name)
            info.size = len(image)
            archive.addfile(info, io.BytesIO(image))
        labels = b"mio_1,car,-5,5,50,30\nmio_1,bicycle,1,1,4,4\n"
        info = tarfile.TarInfo("MIO/gt_train.csv")
        info.size = len(labels)
        archive.addfile(info, io.BytesIO(labels))


def _make_aau(path: Path) -> None:
    (path / "seq_a").mkdir(parents=True)
    (path / "seq_a" / "frame.png").write_bytes(_png_bytes())
    (path / "aauRainSnow-rgb.json").write_text(
        """{
  "images": [{"id": 1, "file_name": "seq_a/frame.png", "width": 80, "height": 40}],
  "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "person"}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [70, 10, 20, 15]},
    {"id": 2, "image_id": 1, "category_id": 2, "bbox": [1, 1, 3, 4]}
  ]
}""",
        encoding="utf-8",
    )


def _make_ua(path: Path, image_path: Path, include_ignored_region: bool = False) -> None:
    image_path.write_bytes(_jpeg_bytes())
    ignored = '<ignored_region><box left="0" top="0" width="20" height="20"/></ignored_region>' if include_ignored_region else ""
    xml = f"""<sequence name="MVI_40172">
  {ignored}
  <frame num="1"><target_list>
    <target id="1"><box left="90" top="10" width="20" height="20"/><attribute vehicle_type="car"/></target>
    <target id="79"><box left="10" top="5" width="20" height="20"/><attribute vehicle_type="others"/></target>
  </target_list></frame>
</sequence>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(image_path, "DETRAC-Images/DETRAC-Images/MVI_40172/img00001.jpg")
        archive.writestr("DETRAC-Train-Annotations-XML/MVI_40172.xml", xml)


def test_unified_export_is_one_class_and_reconciles_counts(tmp_path: Path) -> None:
    mio = tmp_path / "mio.tar"
    aau = tmp_path / "aau"
    ua = tmp_path / "ua.zip"
    image = tmp_path / "ua.jpg"
    _make_mio(mio)
    _make_aau(aau)
    _make_ua(ua, image)
    split = tmp_path / "split.csv"
    split.write_text(
        "dataset_name,sequence_id,proposed_split\n"
        "AAU RainSnow,seq_a,EXTERNAL_VALIDATION\n"
        "UA-DETRAC Original,MVI_40172,CROSS_DATASET_TEST\n",
        encoding="utf-8",
    )
    output = tmp_path / "dataset_v1"
    summary = export_unified_yolo(mio, aau, ua, output, split, MAPPING)
    report = validate_dataset(output)

    assert report["status"] == "PASS"
    assert summary["images_by_split"] == {"cross_test": 1, "train": 2, "val": 1}
    assert summary["counts"]["exported_boxes"] == 3
    assert summary["counts"]["rejected_annotations"] == 2
    assert summary["counts"]["ignored_annotations"] == 1
    assert (output / "data.yaml").read_text(encoding="utf-8").find("vehicle") >= 0
    assert (output / "labels" / "train" / "MIO_mio_negative.txt").read_text(encoding="utf-8") == ""
    with (output / "metadata" / "negative_samples.csv").open(encoding="utf-8", newline="") as handle:
        negatives = list(csv.DictReader(handle))
    assert any(row["image_id"] == "MIO_mio_negative" for row in negatives)
    with (output / "metadata" / "annotations.csv").open(encoding="utf-8", newline="") as handle:
        annotations = list(csv.DictReader(handle))
    assert {row["class_id"] for row in annotations} == {"0"}
    assert {row["original_class"] for row in annotations} == {"car"}
    (output / "metadata" / "negative_samples.csv").write_text(
        "image_id,dataset,split,sequence_id,frame_id,exported_image,exported_label,negative_reason\n",
        encoding="utf-8",
    )
    assert "UNREGISTERED_EMPTY_LABEL:train/MIO_mio_negative" in validate_dataset(output)["errors"]


def test_bbox_size_eda_uses_letterbox_scale_and_25px_rule(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    metadata = dataset / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "images.csv").write_text(
        "image_id,split,width,height\nimage_a,train,640,320\nimage_b,val,320,320\n",
        encoding="utf-8",
    )
    (metadata / "annotations.csv").write_text(
        "image_id,clipped_xmin,clipped_ymin,clipped_xmax,clipped_ymax\n"
        "image_a,0,0,40,20\n"  # 20x10 after letterbox: below 25 by longest side
        "image_b,0,0,30,10\n",  # 30x10 after letterbox: not below 25
        encoding="utf-8",
    )
    report = analyze_bbox_sizes(dataset)
    assert report["box_count"] == 2
    assert report["under_threshold_count"] == 1
    assert report["under_threshold_ratio"] == 0.5
    assert (metadata / "eda" / "bbox_size_320_histogram.png").is_file()


def test_unified_export_accepts_extracted_ua_directory(tmp_path: Path) -> None:
    ua_zip = tmp_path / "ua.zip"
    source_image = tmp_path / "ua.jpg"
    _make_ua(ua_zip, source_image)
    ua_directory = tmp_path / "ua_extracted"
    with zipfile.ZipFile(ua_zip) as archive:
        archive.extractall(ua_directory)
    split = tmp_path / "split.csv"
    split.write_text(
        "dataset_name,sequence_id,proposed_split\n"
        "UA-DETRAC Original,MVI_40172,CROSS_DATASET_TEST\n",
        encoding="utf-8",
    )

    output = tmp_path / "dataset_from_directory"
    summary = export_unified_yolo(None, None, ua_directory, output, split, MAPPING)

    assert summary["images_by_split"] == {"cross_test": 1}
    assert (output / "images" / "cross_test" / "UA_MVI_40172_00001.jpg").is_file()
    assert validate_dataset(output)["status"] == "PASS"


def test_ua_ignored_region_is_masked_only_for_train_and_prediction_centres_are_filtered(tmp_path: Path) -> None:
    ua = tmp_path / "ua.zip"
    image = tmp_path / "ua.jpg"
    _make_ua(ua, image, include_ignored_region=True)
    split = tmp_path / "split.csv"
    split.write_text(
        "dataset_name,sequence_id,proposed_split\n"
        "UA-DETRAC Original,MVI_40172,EXTERNAL_TRAIN\n",
        encoding="utf-8",
    )
    output = tmp_path / "dataset"
    summary = export_unified_yolo(None, None, ua, output, split, MAPPING)
    with Image.open(output / "images" / "train" / "UA_MVI_40172_00001.jpg") as masked:
        assert max(masked.convert("RGB").getpixel((10, 10))) <= 5
        assert min(masked.convert("RGB").getpixel((50, 25))) >= 100
    with (output / "metadata" / "images.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["ignored_region_mask_applied"] == "True"
    assert row["ignored_region_count"] == "1"
    assert summary["counts"]["ignored_region_masked_images"] == 1
    kept, ignored = filter_boxes_by_ignored_region([(1, 1, 5, 5), (40, 10, 60, 30)], [(0, 0, 20, 20)])
    assert kept == [(40, 10, 60, 30)]
    assert ignored == [(1, 1, 5, 5)]


def test_backlit_extraction_requires_manual_review() -> None:
    with pytest.raises(ValueError, match="BACKLIT requires"):
        _conditions({"lighting_condition": "BACKLIT"})
    conditions, basis = _conditions({"lighting_condition": "BACKLIT", "backlit_review_status": "APPROVED"})
    assert conditions == ["BACKLIT"]
    assert "APPROVED" in basis


def test_subset_export_requires_explicit_proposal_override_and_exports_only_manifest_rows(tmp_path: Path) -> None:
    mio = tmp_path / "mio.tar"
    aau = tmp_path / "aau"
    ua = tmp_path / "ua.zip"
    image = tmp_path / "ua.jpg"
    _make_mio(mio)
    _make_aau(aau)
    _make_ua(ua, image)
    split = tmp_path / "split.csv"
    split.write_text(
        "dataset_name,sequence_id,proposed_split\n"
        "AAU RainSnow,seq_a,EXTERNAL_VALIDATION\n"
        "UA-DETRAC Original,MVI_40172,CROSS_DATASET_TEST\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "selection.csv"
    manifest.write_text(
        "dataset_name,source_file,target_subset,selected\n"
        "MIO-TCD Localization,MIO/train/mio_1.jpg,SMOKE,FALSE\n"
        "MIO-TCD Localization,MIO/train/mio_negative.jpg,SMOKE,FALSE\n"
        "AAU RainSnow,seq_a/frame.png,SMOKE,FALSE\n"
        "UA-DETRAC Original,DETRAC-Images/DETRAC-Images/MVI_40172/img00001.jpg,SMOKE,FALSE\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PROPOSAL_ONLY"):
        export_unified_yolo(mio, aau, ua, tmp_path / "blocked", split, MAPPING, manifest, "SMOKE")

    output = tmp_path / "subset"
    summary = export_unified_yolo(mio, aau, ua, output, split, MAPPING, manifest, "SMOKE", True)
    assert summary["counts"]["exported_images"] == 4
    assert summary["selection"]["selection_subset"] == "SMOKE"
    assert summary["selection"]["selection_proposal_override_used"] is True
    assert (output / "metadata" / "selection_manifest.csv").is_file()
    assert validate_dataset(output)["status"] == "PASS"


def test_pilot_validator_enforces_the_complete_release_contract(tmp_path: Path) -> None:
    output = tmp_path / "PILOT_500"
    source_counts = {
        "MIO-TCD Localization": 170,
        "AAU RainSnow": 165,
        "UA-DETRAC Original": 165,
    }
    split_counts = {"train": 421, "val": 51, "cross_test": 28}
    source_split_counts = [
        ("MIO-TCD Localization", "train", 170),
        ("AAU RainSnow", "train", 136),
        ("AAU RainSnow", "val", 15),
        ("AAU RainSnow", "cross_test", 14),
        ("UA-DETRAC Original", "train", 115),
        ("UA-DETRAC Original", "val", 36),
        ("UA-DETRAC Original", "cross_test", 14),
    ]
    rows: list[dict[str, str]] = []
    index = 0
    for dataset, split_name, count in source_split_counts:
        images = output / "images" / split_name
        labels = output / "labels" / split_name
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for _ in range(count):
            index += 1
            stem = f"frame_{index:04d}"
            (images / f"{stem}.jpg").write_bytes(b"placeholder")
            (labels / f"{stem}.txt").write_text("", encoding="utf-8")
            rows.append({"image_id": stem, "dataset": dataset, "split": split_name})
    metadata = output / "metadata"
    metadata.mkdir()
    with (metadata / "images.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "dataset", "split"])
        writer.writeheader()
        writer.writerows(rows)
    (metadata / "annotations.csv").write_text(
        "image_id,dataset,split,sequence_id,frame_id,annotation_id,track_id,original_class,mapped_class,class_id\n",
        encoding="utf-8",
    )
    (metadata / "sequence_splits.csv").write_text(
        "dataset,sequence_id,split,split_source\n"
        "MIO-TCD Localization,mio,train,manifest\n"
        "AAU RainSnow,aau-train,train,manifest\n"
        "AAU RainSnow,aau-val,val,manifest\n"
        "AAU RainSnow,aau-test,cross_test,manifest\n"
        "UA-DETRAC Original,ua-train,train,manifest\n"
        "UA-DETRAC Original,ua-val,val,manifest\n"
        "UA-DETRAC Original,ua-test,cross_test,manifest\n",
        encoding="utf-8",
    )
    (metadata / "selection_manifest.csv").write_text("selection_id\n", encoding="utf-8")
    summary = {
        "counts": {"input_annotations": 0, "exported_boxes": 0, "rejected_annotations": 0, "input_images": 500, "exported_images": 500},
        "images_by_split": split_counts,
        "boxes_by_split": {"train": 0, "val": 0, "cross_test": 0},
        "selection": {
            "selection_subset": "PILOT_500",
            "selection_image_count": 500,
            "selection_images_by_dataset": source_counts,
        },
    }
    (metadata / "export_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (output / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/cross_test\nnames: {0: vehicle}\n", encoding="utf-8")

    assert validate_pilot_500(output, REPO_ROOT / "data_collection" / "configs" / "dataset_config.yaml")["status"] == "PASS"
