from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from remediate_t2_dataset import apply_remediation, audit_dataset  # noqa: E402


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 16), color=(10, 20, 30)).save(path)


def _label(path: Path, boxes: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("0 0.5 0.5 0.2 0.2\n" for _ in range(boxes)), encoding="utf-8")


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fixture_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset-v1-full"
    fields = [
        "image_id", "dataset", "split", "sequence_id", "frame_id", "source_image",
        "exported_image", "exported_label", "width", "height", "vehicle_box_count",
        "boundary_clipped_box_count",
    ]
    rows: list[dict[str, str]] = []
    assets = [
        ("val", "UPX_1_frame_000001.jpg", "UPX_1_000001", "UPX_1"),
        ("cross_test", "UPX_2_frame_000001.jpg", "UPX_2_000001", "UPX_2"),
        ("train", "MIO_001.jpg", "MIO_001", "MIO_NO_SEQUENCE_TRAIN_ONLY"),
    ]
    for split, name, image_id, sequence in assets:
        image = dataset / "images" / split / name
        label = dataset / "labels" / split / f"{Path(name).stem}.txt"
        _image(image)
        _label(label)
        rows.append({
            "image_id": image_id,
            "dataset": "ONLINE_DATA User Provided Video" if name.startswith("UPX_") else "MIO-TCD Localization",
            "split": split,
            "sequence_id": sequence,
            "frame_id": "000001",
            "source_image": name,
            "exported_image": image.relative_to(dataset).as_posix(),
            "exported_label": label.relative_to(dataset).as_posix(),
            "width": "32",
            "height": "16",
            "vehicle_box_count": "1",
            "boundary_clipped_box_count": "0",
        })
    for name, sequence in (("ONLINE_HW_DAY_01_f0001.jpg", "ONLINE_HW_DAY_01"), ("ONLINE_HW_DAY_02_f0001.jpg", "ONLINE_HW_DAY_02")):
        _image(dataset / "images" / "train" / name)
        _label(dataset / "labels" / "train" / f"{Path(name).stem}.txt", boxes=2)
    _write_csv(dataset / "metadata" / "images.csv", fields, rows)
    _write_csv(
        dataset / "metadata" / "sequence_scene_metadata.csv",
        ["dataset_name", "sequence_id", "road_type", "weather", "lighting"],
        [
            {"dataset_name": "ONLINE_DATA", "sequence_id": "UPX_1", "road_type": "HIGHWAY", "weather": "CLEAR", "lighting": "DAY"},
            {"dataset_name": "ONLINE_DATA", "sequence_id": "UPX_2", "road_type": "HIGHWAY", "weather": "CLEAR", "lighting": "NIGHT"},
            {"dataset_name": "ONLINE_DATA", "sequence_id": "ONLINE_HW_DAY_01", "road_type": "HIGHWAY", "weather": "CLEAR", "lighting": "DAY"},
            {"dataset_name": "ONLINE_DATA", "sequence_id": "ONLINE_HW_DAY_02", "road_type": "HIGHWAY", "weather": "CLEAR", "lighting": "DAY"},
        ],
    )
    (dataset / "metadata" / "export_summary.json").write_text(json.dumps({
        "counts": {"input_images": 5, "exported_images": 5, "exported_boxes": 7},
        "images_by_split": {"train": 3, "val": 1, "cross_test": 1},
        "boxes_by_split": {"train": 5, "val": 1, "cross_test": 1},
    }), encoding="utf-8")
    return dataset


def test_audit_reports_upx_and_orphan_online_hw(tmp_path: Path) -> None:
    dataset = _fixture_dataset(tmp_path)
    plan = audit_dataset(dataset)

    assert plan["status"] == "REVIEW_COUNTS_BEFORE_APPLY"
    assert plan["upx"]["evaluation_images_by_split"] == {"val": 1, "cross_test": 1}
    assert plan["online_hw"]["orphan_image_count"] == 2
    assert plan["upx"]["distribution_by_road_type_and_lighting"] == {
        "cross_test / HIGHWAY / NIGHT": 1,
        "val / HIGHWAY / DAY": 1,
    }


def test_apply_requires_ready_plan_and_quarantines_evaluation_assets(tmp_path: Path) -> None:
    dataset = _fixture_dataset(tmp_path)
    plan = audit_dataset(dataset)
    plan["status"] = "READY_TO_APPLY"  # The small fixture intentionally does not use production counts.
    result = apply_remediation(dataset, plan)

    assert result["status"] == "PASS"
    assert not list((dataset / "images" / "val").glob("UPX_*"))
    assert not list((dataset / "images" / "cross_test").glob("UPX_*"))
    assert (dataset / "review_pending" / "t2_upx_auto_label_evaluation_20260816" / "images" / "val" / "UPX_1_frame_000001.jpg").is_file()

    with (dataset / "metadata" / "images.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "label_source" in rows[0]
    assert sum(row["split"] == "pending_review" for row in rows) == 2
    assert sum(row["remediation_note"].startswith("Backfilled") for row in rows) == 2

    summary = json.loads((dataset / "metadata" / "export_summary.json").read_text(encoding="utf-8"))
    assert summary["images_by_split"] == {"train": 3, "val": 0, "cross_test": 0}
    assert summary["t2_remediation"]["upx_quarantined_boxes"] == 2
