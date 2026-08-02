"""Export UA-DETRAC XML annotations to a leakage-safe one-class YOLO dataset.

The exporter clips boundary-crossing boxes, applies the reviewed class mapping
and track exclusions, and writes a companion per-annotation manifest retaining
the original UA class and track ID.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from external_eda_common import (
    ROOT,
    clip_bbox_to_image,
    image_from_bytes,
    load_yaml,
    read_csv,
    safe_sequence,
    validate_bbox,
)


DEFAULT_UA = (
    ROOT
    / "storage_placeholders"
    / "online_data"
    / "raw"
    / "ua_detrac_orig"
    / "ua-detrac-orig.zip"
)
DEFAULT_OUTPUT = ROOT / "dataset_output" / "ua_detrac_yolo"
DEFAULT_SPLIT = ROOT / "reports" / "external_eda" / "split_proposal.csv"
DEFAULT_MAPPING = ROOT / "configs" / "vehicle_class_mapping.yaml"

SPLIT_NAMES = {
    "EXTERNAL_TRAIN": "train",
    "EXTERNAL_VALIDATION": "val",
    "CROSS_DATASET_TEST": "cross_test",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

ANNOTATION_FIELDS = [
    "image_id",
    "split",
    "sequence_id",
    "frame_id",
    "track_id",
    "original_class",
    "mapped_class",
    "class_id",
    "source_image",
    "source_annotation",
    "raw_xmin",
    "raw_ymin",
    "raw_xmax",
    "raw_ymax",
    "clipped_xmin",
    "clipped_ymin",
    "clipped_xmax",
    "clipped_ymax",
    "clip_applied",
    "clip_adjustments",
    "preserve_original_class",
]
REJECTED_FIELDS = [
    "sequence_id",
    "frame_id",
    "track_id",
    "original_class",
    "source_annotation",
    "action",
    "reason",
]
IMAGE_FIELDS = [
    "image_id",
    "split",
    "sequence_id",
    "frame_id",
    "source_image",
    "exported_image",
    "exported_label",
    "width",
    "height",
    "vehicle_box_count",
    "boundary_clipped_box_count",
]


def _csv_writer(path: Path, fields: list[str]) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    return handle, writer


def _split_by_sequence(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in read_csv(path):
        if row.get("dataset_name") != "UA-DETRAC Original":
            continue
        sequence = str(row.get("sequence_id", "")).strip()
        proposed = str(row.get("proposed_split", "")).strip()
        if sequence and proposed in SPLIT_NAMES:
            result[sequence] = SPLIT_NAMES[proposed]
    return result


def _track_exclusions(rule: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(row.get("sequence_id", "")), str(row.get("track_id", "")))
        for row in rule.get("track_exclusions", [])
        if row.get("action") == "EXCLUDE_NON_VEHICLE_TRACK"
    }


def _image_dimensions(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[int, int]:
    image = image_from_bytes(archive.read(info))
    if image is None:
        raise ValueError(f"Unreadable UA image: {info.filename}")
    return int(image.shape[1]), int(image.shape[0])


def _yolo_line(box: tuple[float, float, float, float], width: int, height: int) -> str:
    xmin, ymin, xmax, ymax = box
    center_x = ((xmin + xmax) / 2.0) / width
    center_y = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    values = (center_x, center_y, box_width, box_height)
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Normalized YOLO box outside [0,1]: {values}")
    return "0 " + " ".join(f"{value:.8f}" for value in values)


def _prepare_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output is not empty: {output}. Choose a new directory; the exporter never deletes existing data."
        )
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES.values():
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "metadata").mkdir(parents=True, exist_ok=True)


def export_ua_detrac(
    ua_path: Path,
    output: Path,
    split_path: Path,
    mapping_path: Path,
    limit_images: int | None = None,
) -> dict[str, Any]:
    if not ua_path.is_file():
        raise FileNotFoundError(f"UA-DETRAC ZIP not found: {ua_path}")
    split_map = _split_by_sequence(split_path)
    if not split_map:
        raise ValueError(f"No UA sequence split assignments found in {split_path}")
    mapping = load_yaml(mapping_path)
    ua_rules = mapping.get("ua_detrac", {})
    others_rule = ua_rules.get("others", {})
    exclusions = _track_exclusions(others_rule)
    preserve_original = bool(mapping.get("preserve_original_class"))
    if not preserve_original:
        raise ValueError("vehicle_class_mapping.yaml must keep preserve_original_class=true")

    _prepare_output(output)
    annotation_handle, annotation_writer = _csv_writer(
        output / "metadata" / "annotations.csv", ANNOTATION_FIELDS
    )
    rejected_handle, rejected_writer = _csv_writer(
        output / "metadata" / "rejected_annotations.csv", REJECTED_FIELDS
    )
    image_handle, image_writer = _csv_writer(
        output / "metadata" / "images.csv", IMAGE_FIELDS
    )

    counts: Counter[str] = Counter()
    split_image_counts: Counter[str] = Counter()
    split_box_counts: Counter[str] = Counter()
    exported_images = 0
    try:
        with zipfile.ZipFile(ua_path) as archive:
            infos = archive.infolist()
            image_by_frame: dict[tuple[str, int], zipfile.ZipInfo] = {}
            first_by_sequence: dict[str, zipfile.ZipInfo] = {}
            for info in infos:
                suffix = Path(info.filename).suffix.lower()
                if suffix not in IMAGE_SUFFIXES or "/detrac-images/" not in info.filename.lower():
                    continue
                sequence = safe_sequence(info.filename)
                try:
                    frame_id = int(Path(info.filename).stem.lower().replace("img", ""))
                except ValueError:
                    continue
                image_by_frame[(sequence, frame_id)] = info
                first_by_sequence.setdefault(sequence, info)
            dimensions = {
                sequence: _image_dimensions(archive, info)
                for sequence, info in first_by_sequence.items()
            }

            xml_infos = sorted(
                (
                    info
                    for info in infos
                    if info.filename.lower().endswith(".xml")
                    and "annotations-xml" in info.filename.lower()
                ),
                key=lambda item: item.filename,
            )
            stop = False
            for xml_info in xml_infos:
                root = ET.fromstring(archive.read(xml_info))
                sequence = root.attrib.get("name") or Path(xml_info.filename).stem
                split = split_map.get(sequence)
                if split is None:
                    raise ValueError(f"Missing split assignment for UA sequence {sequence}")
                width, height = dimensions.get(sequence, (0, 0))
                if width <= 0 or height <= 0:
                    raise ValueError(f"Missing image dimensions for UA sequence {sequence}")

                for frame_node in root.findall("frame"):
                    if limit_images is not None and exported_images >= limit_images:
                        stop = True
                        break
                    frame_id = int(frame_node.attrib.get("num", "0"))
                    source_info = image_by_frame.get((sequence, frame_id))
                    if source_info is None:
                        counts["missing_source_image"] += 1
                        continue
                    image_id = f"UA_{sequence}_{frame_id:05d}"
                    suffix = Path(source_info.filename).suffix.lower()
                    image_target = output / "images" / split / f"{image_id}{suffix}"
                    label_target = output / "labels" / split / f"{image_id}.txt"
                    lines: list[str] = []
                    clipped_in_image = 0

                    for target in frame_node.findall("./target_list/target"):
                        track_id = str(target.attrib.get("id", ""))
                        attribute = target.find("attribute")
                        box = target.find("box")
                        original_class = (
                            str(attribute.attrib.get("vehicle_type", "MISSING"))
                            if attribute is not None
                            else "MISSING"
                        )
                        base_reject = {
                            "sequence_id": sequence,
                            "frame_id": frame_id,
                            "track_id": track_id,
                            "original_class": original_class,
                            "source_annotation": xml_info.filename,
                        }
                        rule = ua_rules.get(original_class)
                        if rule is None or not rule.get("include", False):
                            rejected_writer.writerow(
                                {**base_reject, "action": "EXCLUDE_CLASS", "reason": "CLASS_NOT_INCLUDED"}
                            )
                            counts["excluded_class"] += 1
                            continue
                        if (sequence, track_id) in exclusions:
                            rejected_writer.writerow(
                                {
                                    **base_reject,
                                    "action": "EXCLUDE_NON_VEHICLE_TRACK",
                                    "reason": "DATA_LEAD_TRACK_EXCLUSION",
                                }
                            )
                            counts["excluded_track_boxes"] += 1
                            continue
                        if box is None:
                            rejected_writer.writerow(
                                {**base_reject, "action": "EXCLUDE_INVALID", "reason": "MISSING_BOX"}
                            )
                            counts["invalid_before_clip"] += 1
                            continue

                        try:
                            xmin = float(box.attrib["left"])
                            ymin = float(box.attrib["top"])
                            box_width = float(box.attrib["width"])
                            box_height = float(box.attrib["height"])
                            xmax, ymax = xmin + box_width, ymin + box_height
                        except (KeyError, TypeError, ValueError):
                            rejected_writer.writerow(
                                {**base_reject, "action": "EXCLUDE_INVALID", "reason": "NON_NUMERIC_BOX"}
                            )
                            counts["invalid_before_clip"] += 1
                            continue
                        if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
                            before_issues = ["NAN_OR_INFINITY"]
                        else:
                            before_issues = validate_bbox(xmin, ymin, xmax, ymax, width, height)
                        if "NAN_OR_INFINITY" in before_issues or "NON_POSITIVE_SIZE" in before_issues:
                            rejected_writer.writerow(
                                {
                                    **base_reject,
                                    "action": "EXCLUDE_INVALID",
                                    "reason": "|".join(before_issues),
                                }
                            )
                            counts["invalid_before_clip"] += 1
                            continue

                        clipped, adjustments = clip_bbox_to_image(
                            xmin, ymin, xmax, ymax, width, height
                        )
                        after_issues = validate_bbox(*clipped, width, height)
                        if after_issues:
                            rejected_writer.writerow(
                                {
                                    **base_reject,
                                    "action": "EXCLUDE_INVALID_AFTER_CLIP",
                                    "reason": "|".join(after_issues),
                                }
                            )
                            counts["invalid_after_clip"] += 1
                            continue

                        lines.append(_yolo_line(clipped, width, height))
                        if adjustments:
                            clipped_in_image += 1
                            counts["clipped_boxes"] += 1
                        counts["exported_boxes"] += 1
                        split_box_counts[split] += 1
                        annotation_writer.writerow(
                            {
                                "image_id": image_id,
                                "split": split,
                                "sequence_id": sequence,
                                "frame_id": frame_id,
                                "track_id": track_id,
                                "original_class": original_class,
                                "mapped_class": "vehicle",
                                "class_id": 0,
                                "source_image": source_info.filename,
                                "source_annotation": xml_info.filename,
                                "raw_xmin": xmin,
                                "raw_ymin": ymin,
                                "raw_xmax": xmax,
                                "raw_ymax": ymax,
                                "clipped_xmin": clipped[0],
                                "clipped_ymin": clipped[1],
                                "clipped_xmax": clipped[2],
                                "clipped_ymax": clipped[3],
                                "clip_applied": bool(adjustments),
                                "clip_adjustments": "|".join(adjustments),
                                "preserve_original_class": True,
                            }
                        )

                    image_target.write_bytes(archive.read(source_info))
                    label_target.write_text(
                        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                    )
                    image_writer.writerow(
                        {
                            "image_id": image_id,
                            "split": split,
                            "sequence_id": sequence,
                            "frame_id": frame_id,
                            "source_image": source_info.filename,
                            "exported_image": image_target.relative_to(output).as_posix(),
                            "exported_label": label_target.relative_to(output).as_posix(),
                            "width": width,
                            "height": height,
                            "vehicle_box_count": len(lines),
                            "boundary_clipped_box_count": clipped_in_image,
                        }
                    )
                    exported_images += 1
                    split_image_counts[split] += 1
                    counts["exported_images"] += 1
                if stop:
                    break
    finally:
        annotation_handle.close()
        rejected_handle.close()
        image_handle.close()

    dataset_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/cross_test",
        "names": {0: "vehicle"},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    summary = {
        "dataset": "UA-DETRAC Original",
        "target_class": "vehicle",
        "preserve_original_class": True,
        "source_zip": str(ua_path.resolve()),
        "split_source": str(split_path.resolve()),
        "mapping_source": str(mapping_path.resolve()),
        "limit_images": limit_images,
        "counts": dict(sorted(counts.items())),
        "images_by_split": dict(sorted(split_image_counts.items())),
        "boxes_by_split": dict(sorted(split_box_counts.items())),
        "track_exclusions": sorted([{"sequence_id": a, "track_id": b} for a, b in exclusions], key=str),
    }
    (output / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ua-path", type=Path, default=DEFAULT_UA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--mapping-path", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--limit-images", type=int)
    args = parser.parse_args()
    if args.limit_images is not None and args.limit_images <= 0:
        parser.error("--limit-images must be positive")
    summary = export_ua_detrac(
        args.ua_path.resolve(),
        args.output.resolve(),
        args.split_path.resolve(),
        args.mapping_path.resolve(),
        args.limit_images,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
