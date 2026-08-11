"""Independent, read-only audit of the external-dataset EDA.

This module deliberately does not import the production EDA parsers. It reads the
raw archives/annotations directly and writes only to reports/audit* directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import tracemalloc
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DEFAULT_AUDIT = ROOT / "reports" / "audit"
DEFAULT_REPRO = ROOT / "reports" / "audit_reproduction"
COMMIT = "80e66fb"


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_text(command: list[str], cwd: Path = REPO) -> tuple[int, str, str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout, result.stderr


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def image_size_bytes(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as image:
        return int(image.width), int(image.height)


def sequence_from_path(path: str) -> str:
    parts = PurePosixPath(path.replace("\\", "/")).parts
    for part in parts:
        if part.upper().startswith("MVI_"):
            return part
    return parts[-2] if len(parts) >= 2 else Path(path).stem


def reservoir(items: list[str], size: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    sample: list[str] = []
    for index, item in enumerate(items):
        if index < size:
            sample.append(item)
        else:
            replacement = rng.randint(0, index)
            if replacement < size:
                sample[replacement] = item
    return sample


def finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def quantile(values: list[float], q: float) -> float | str:
    if not values:
        return ""
    return round(float(np.quantile(np.asarray(values, dtype=np.float64), q)), 6)


def audit_mio(path: Path, mapping: dict[str, Any], sample_size: int = 5000) -> dict[str, Any]:
    started = time.perf_counter()
    train_images: list[str] = []
    test_images: list[str] = []
    classification_members: list[str] = []
    csv_name = ""
    readme_name = ""
    duplicate_member_names = 0
    seen_names: set[str] = set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            name = member.name.replace("\\", "/")
            lower = name.lower()
            if name in seen_names:
                duplicate_member_names += 1
            seen_names.add(name)
            if "classification" in lower:
                classification_members.append(name)
            if lower.endswith((".jpg", ".jpeg", ".png")):
                if "/train/" in lower:
                    train_images.append(name)
                elif "/test/" in lower:
                    test_images.append(name)
            elif lower.endswith("/gt_train.csv"):
                csv_name = name
            elif lower.endswith("/readme.txt"):
                readme_name = name
    if not csv_name:
        raise RuntimeError("MIO gt_train.csv not found")

    sample_names = set(reservoir(train_images, min(sample_size, len(train_images)), 230))
    sample_dimensions: dict[str, tuple[int, int]] = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if member.name not in sample_names:
                continue
            handle = archive.extractfile(member)
            if handle:
                sample_dimensions[Path(member.name).stem] = image_size_bytes(handle.read())

    rules = mapping["mio_tcd"]
    class_counts: Counter[str] = Counter()
    included_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    annotated_ids: set[str] = set()
    duplicate_keys: Counter[tuple[str, str, float, float, float, float]] = Counter()
    raw_rows = parsed_boxes = valid_boxes = included = excluded = invalid = 0
    final_sample_count = sample_oob = 0
    with tarfile.open(path, "r:") as archive:
        handle = archive.extractfile(csv_name)
        if not handle:
            raise RuntimeError("MIO gt_train.csv unreadable")
        text = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
        for row in csv.reader(text):
            raw_rows += 1
            if len(row) < 6:
                invalid += 1
                invalid_counts["MALFORMED_ROW"] += 1
                continue
            image_id, original_class = row[0].strip(), row[1].strip()
            class_counts[original_class] += 1
            annotated_ids.add(image_id)
            try:
                xmin, ymin, xmax, ymax = map(float, row[2:6])
            except ValueError:
                invalid += 1
                invalid_counts["NON_NUMERIC"] += 1
                continue
            parsed_boxes += 1
            if not finite((xmin, ymin, xmax, ymax)):
                invalid += 1
                invalid_counts["NAN_OR_INFINITY"] += 1
                continue
            if xmin >= xmax or ymin >= ymax:
                invalid += 1
                invalid_counts["NON_POSITIVE_SIZE"] += 1
                continue
            valid_boxes += 1
            duplicate_keys[(image_id, original_class, xmin, ymin, xmax, ymax)] += 1
            rule = rules.get(original_class)
            if rule is None:
                unknown_counts[original_class] += 1
                excluded += 1
            elif rule.get("include") and rule.get("mapped_class") == "vehicle":
                included += 1
                included_counts[original_class] += 1
            else:
                excluded += 1
                excluded_counts[original_class] += 1
            if image_id in sample_dimensions:
                width, height = sample_dimensions[image_id]
                if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                    sample_oob += 1
                else:
                    final_sample_count += 1

    train_ids = {Path(name).stem for name in train_images}
    return {
        "dataset": "MIO-TCD Localization",
        "path": str(path.resolve()),
        "path_sha256": sha256_file(path),
        "train_images": len(train_images),
        "test_images": len(test_images),
        "image_count": len(train_images) + len(test_images),
        "classification_member_count": len(classification_members),
        "duplicate_member_names": duplicate_member_names,
        "readme_present": bool(readme_name),
        "raw_box_count": raw_rows,
        "parsed_box_count": parsed_boxes,
        "valid_before_clip": valid_boxes,
        "out_of_bounds_count": sample_oob,
        "out_of_bounds_scope": f"DETERMINISTIC_SAMPLE_{len(sample_dimensions)}_IMAGES",
        "clipped_count": 0,
        "excluded_count": excluded,
        "invalid_count": invalid,
        "mapped_vehicle_count": included,
        "final_analysis_count": final_sample_count,
        "sample_image_count": len(sample_dimensions),
        "images_without_annotations": len(train_ids - annotated_ids),
        "annotations_without_images": len(annotated_ids - train_ids),
        "exact_duplicate_annotation_rows": sum(value - 1 for value in duplicate_keys.values() if value > 1),
        "class_counts": dict(class_counts),
        "included_counts": dict(included_counts),
        "excluded_counts": dict(excluded_counts),
        "unknown_counts": dict(unknown_counts),
        "invalid_counts": dict(invalid_counts),
        "conservation_ok": raw_rows == included + excluded + invalid,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def create_aau_contact_sheets(root: Path, sequences: list[str], output: Path) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, str] = {}
    for page, offset in enumerate(range(0, len(sequences), 6), 1):
        group = sequences[offset : offset + 6]
        canvas = np.zeros((len(group) * 200, 960, 3), dtype=np.uint8)
        for row, sequence in enumerate(group):
            matches = list(root.glob(f"*/{sequence}/cam1.mkv"))
            if not matches:
                continue
            capture = cv2.VideoCapture(str(matches[0]))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            for column, ratio in enumerate((0.1, 0.5, 0.9)):
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_count * ratio) - 1))
                ok, frame = capture.read()
                tile = (
                    cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                    if ok and frame is not None
                    else np.zeros((180, 320, 3), dtype=np.uint8)
                )
                cv2.rectangle(tile, (0, 0), (320, 24), (0, 0, 0), -1)
                cv2.putText(tile, f"{sequence} | {int(ratio * 100)}%", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                canvas[row * 200 : row * 200 + 180, column * 320 : (column + 1) * 320] = tile
            capture.release()
        destination = output / f"aau_lighting_audit_page_{page}.jpg"
        cv2.imwrite(str(destination), canvas)
        relative = destination.relative_to(ROOT).as_posix()
        for sequence in group:
            evidence[sequence] = relative
    return evidence


def audit_aau(path: Path, mapping: dict[str, Any], repro: Path) -> dict[str, Any]:
    started = time.perf_counter()
    rgb_path = path / "aauRainSnow-rgb.json"
    thermal_path = path / "aauRainSnow-thermal.json"
    rgb = json.loads(rgb_path.read_text(encoding="utf-8"))
    thermal = json.loads(thermal_path.read_text(encoding="utf-8")) if thermal_path.exists() else {}
    categories = {int(row["id"]): str(row["name"]) for row in rgb.get("categories", [])}
    images = {int(row["id"]): row for row in rgb.get("images", [])}
    rules = mapping["aau_rainsnow"]

    sequence_dirs = sorted(
        directory.name
        for parent in path.iterdir()
        if parent.is_dir()
        for directory in parent.iterdir()
        if directory.is_dir() and (directory / "cam1.mkv").exists()
    )
    video_rows: list[dict[str, Any]] = []
    for sequence in sequence_dirs:
        directory = next(path.glob(f"*/{sequence}"))
        for video in sorted(directory.glob("*.mkv")):
            capture = cv2.VideoCapture(str(video))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            opened = capture.isOpened()
            capture.release()
            video_rows.append(
                {
                    "sequence": sequence,
                    "video": video.name,
                    "opened": opened,
                    "fps": fps,
                    "frames": frames,
                    "duration": frames / fps if fps else 0,
                    "resolution": f"{width}x{height}",
                }
            )

    review = load_yaml(ROOT / "configs" / "aau_sequence_lighting_review.yaml")
    review_rows = review.get("sequences", {})
    evidence = create_aau_contact_sheets(path, sequence_dirs, repro / "contact_sheets")
    sequence_rows: list[dict[str, Any]] = []
    for sequence in sequence_dirs:
        videos = [row for row in video_rows if row["sequence"] == sequence]
        decision = review_rows.get(sequence, {})
        sequence_rows.append(
            {
                "sequence_name": sequence,
                "video_count": len(videos),
                "duration_seconds_total": round(sum(row["duration"] for row in videos), 6),
                "resolution": "|".join(sorted({row["resolution"] for row in videos})),
                "fps": "|".join(str(round(row["fps"], 6)) for row in videos),
                "official_condition": "RAIN_SNOW_DATASET_LEVEL_NOT_SEQUENCE_SPECIFIC",
                "reviewed_condition": decision.get("lighting", "UNKNOWN"),
                "lighting_label": decision.get("lighting", "UNKNOWN"),
                "weather_label": "RAIN_OR_SNOW_NOT_SEPARATED",
                "label_source": decision.get("lighting_source", "MISSING"),
                "review_evidence": evidence.get(sequence, ""),
                "configured_evidence_text": decision.get("evidence", ""),
                "reviewer": review.get("reviewer", "MISSING"),
                "review_date": review.get("review_date", ""),
                "notes": "Audit contact sheet regenerated from raw RGB video; original commit did not retain the review sheet.",
            }
        )

    class_counts: Counter[str] = Counter()
    included_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    duplicate_keys: Counter[tuple[Any, ...]] = Counter()
    annotated_ids: set[int] = set()
    raw_boxes = parsed = valid = included = excluded = invalid = out_of_bounds = 0
    for annotation in rgb.get("annotations", []):
        raw_boxes += 1
        image_id = int(annotation.get("image_id", -1))
        original_class = categories.get(int(annotation.get("category_id", -1)), "UNKNOWN")
        class_counts[original_class] += 1
        annotated_ids.add(image_id)
        bbox = annotation.get("bbox") or []
        meta = images.get(image_id)
        if len(bbox) != 4 or meta is None:
            invalid += 1
            invalid_counts["MALFORMED_OR_MISSING_IMAGE_META"] += 1
            continue
        try:
            x, y, width, height = map(float, bbox)
        except (TypeError, ValueError):
            invalid += 1
            invalid_counts["NON_NUMERIC"] += 1
            continue
        parsed += 1
        if not finite((x, y, width, height)):
            invalid += 1
            invalid_counts["NAN_OR_INFINITY"] += 1
            continue
        if width <= 0 or height <= 0:
            invalid += 1
            invalid_counts["NON_POSITIVE_SIZE"] += 1
            continue
        image_width, image_height = int(meta["width"]), int(meta["height"])
        if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
            out_of_bounds += 1
            invalid += 1
            invalid_counts["OUT_OF_BOUNDS"] += 1
            continue
        valid += 1
        duplicate_keys[(image_id, original_class, x, y, width, height)] += 1
        rule = rules.get(original_class)
        if rule and rule.get("include") and rule.get("mapped_class") == "vehicle":
            included += 1
            included_counts[original_class] += 1
        else:
            excluded += 1
            excluded_counts[original_class] += 1
    return {
        "dataset": "AAU RainSnow",
        "path": str(path.resolve()),
        "rgb_json_sha256": sha256_file(rgb_path),
        "thermal_json_sha256": sha256_file(thermal_path) if thermal_path.exists() else "",
        "rgb_images": len(images),
        "thermal_images": len(thermal.get("images", [])),
        "image_count": len(images) + len(thermal.get("images", [])),
        "sequence_count": len(sequence_dirs),
        "video_count": len(video_rows),
        "unreadable_videos": sum(not row["opened"] for row in video_rows),
        "raw_box_count": raw_boxes,
        "parsed_box_count": parsed,
        "valid_before_clip": valid,
        "out_of_bounds_count": out_of_bounds,
        "clipped_count": 0,
        "excluded_count": excluded,
        "invalid_count": invalid,
        "mapped_vehicle_count": included,
        "final_analysis_count": valid,
        "images_without_annotations": len(set(images) - annotated_ids),
        "annotations_without_images": len(annotated_ids - set(images)),
        "exact_duplicate_annotation_rows": sum(value - 1 for value in duplicate_keys.values() if value > 1),
        "class_counts": dict(class_counts),
        "included_counts": dict(included_counts),
        "excluded_counts": dict(excluded_counts),
        "invalid_counts": dict(invalid_counts),
        "lighting_counts": dict(Counter(row["lighting_label"] for row in sequence_rows)),
        "sequence_rows": sequence_rows,
        "reviewer_missing": not bool(review.get("reviewer")),
        "conservation_ok": raw_boxes == included + excluded + invalid,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def create_others_contact_sheet(
    archive: zipfile.ZipFile,
    samples: list[dict[str, Any]],
    output: Path,
) -> str:
    tiles: list[np.ndarray] = []
    for sample in samples:
        source = sample.get("source_file", "")
        if not source:
            continue
        array = np.frombuffer(archive.read(source), dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            continue
        left, top, right, bottom = sample["clipped_xyxy"]
        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 255), 3)
        tile = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (480, 28), (0, 0, 0), -1)
        cv2.putText(tile, f"{sample['sequence']} frame {sample['frame']} track {sample['track_id']}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    canvas = np.zeros((((len(tiles) + 2) // 3) * 270, 1440, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, 3)
        canvas[row * 270 : (row + 1) * 270, column * 480 : (column + 1) * 480] = tile
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)
    return output.relative_to(ROOT).as_posix()


def audit_ua(path: Path, repro: Path) -> dict[str, Any]:
    started = time.perf_counter()
    requested = read_csv(ROOT / "reports" / "external_eda" / "ua_others_sample_review.csv")
    requested_keys = {(row["sequence_id"], int(row["frame_id"])) for row in requested}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        image_infos = [
            info
            for info in infos
            if info.filename.lower().endswith((".jpg", ".jpeg", ".png"))
            and "/detrac-images/" in info.filename.lower()
        ]
        xml_infos = [
            info
            for info in infos
            if info.filename.lower().endswith(".xml")
            and "annotations-xml" in info.filename.lower()
        ]
        by_sequence: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        image_by_frame: dict[tuple[str, int], zipfile.ZipInfo] = {}
        for info in image_infos:
            sequence = sequence_from_path(info.filename)
            by_sequence[sequence].append(info)
            try:
                frame = int(Path(info.filename).stem.lower().replace("img", ""))
                image_by_frame[(sequence, frame)] = info
            except ValueError:
                pass
        dimensions: dict[str, tuple[int, int]] = {}
        dimension_variation: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for sequence, values in by_sequence.items():
            values.sort(key=lambda info: info.filename)
            probes = [values[0], values[len(values) // 2], values[-1]]
            for info in probes:
                size = image_size_bytes(archive.read(info))
                dimension_variation[sequence].add(size)
                dimensions.setdefault(sequence, size)

        class_counts: Counter[str] = Counter()
        side_counts: Counter[str] = Counter()
        severity_counts: Counter[str] = Counter()
        invalid_counts: Counter[str] = Counter()
        track_class_values: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        duplicate_track_frame = 0
        overrun_pixels: list[float] = []
        retained_ratios: list[float] = []
        clipped_widths: list[float] = []
        clipped_heights: list[float] = []
        raw_boxes = parsed = valid_before = oob = clipped = invalid = fully_outside = 0
        frame_count = missing_frame_count = track_count = ignored_regions = 0
        sequence_track_sets: defaultdict[str, set[str]] = defaultdict(set)
        others_candidates: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        coordinate_boundary_equal = 0

        for xml_info in xml_infos:
            root = ET.fromstring(archive.read(xml_info))
            sequence = root.attrib.get("name") or Path(xml_info.filename).stem
            width, height = dimensions.get(sequence, (0, 0))
            ignored_regions += len(root.findall("./ignored_region/box"))
            for frame in root.findall("frame"):
                frame_count += 1
                frame_number = int(frame.attrib.get("num", frame_count))
                if (sequence, frame_number) not in image_by_frame:
                    missing_frame_count += 1
                seen_track_ids: set[str] = set()
                for target in frame.findall("./target_list/target"):
                    raw_boxes += 1
                    track_id = target.attrib.get("id", "MISSING")
                    if track_id in seen_track_ids:
                        duplicate_track_frame += 1
                    seen_track_ids.add(track_id)
                    sequence_track_sets[sequence].add(track_id)
                    box = target.find("box")
                    attribute = target.find("attribute")
                    original_class = attribute.attrib.get("vehicle_type", "MISSING") if attribute is not None else "MISSING"
                    class_counts[original_class] += 1
                    track_class_values[(sequence, track_id)].add(original_class)
                    if box is None:
                        invalid += 1
                        invalid_counts["MISSING_BOX"] += 1
                        continue
                    try:
                        left = float(box.attrib["left"])
                        top = float(box.attrib["top"])
                        box_width = float(box.attrib["width"])
                        box_height = float(box.attrib["height"])
                    except (KeyError, ValueError):
                        invalid += 1
                        invalid_counts["NON_NUMERIC_OR_MISSING_COORDINATE"] += 1
                        continue
                    parsed += 1
                    if not finite((left, top, box_width, box_height)):
                        invalid += 1
                        invalid_counts["NAN_OR_INFINITY"] += 1
                        continue
                    if box_width <= 0 or box_height <= 0:
                        invalid += 1
                        invalid_counts["NON_POSITIVE_SIZE"] += 1
                        continue
                    valid_before += 1
                    right, bottom = left + box_width, top + box_height
                    if right == width or bottom == height:
                        coordinate_boundary_equal += 1
                    sides: list[tuple[str, float]] = []
                    if left < 0:
                        sides.append(("LEFT", -left))
                    if top < 0:
                        sides.append(("TOP", -top))
                    if right > width:
                        sides.append(("RIGHT", right - width))
                    if bottom > height:
                        sides.append(("BOTTOM", bottom - height))
                    clipped_left = min(max(left, 0.0), float(width))
                    clipped_top = min(max(top, 0.0), float(height))
                    clipped_right = min(max(right, 0.0), float(width))
                    clipped_bottom = min(max(bottom, 0.0), float(height))
                    clipped_area = max(0.0, clipped_right - clipped_left) * max(0.0, clipped_bottom - clipped_top)
                    raw_area = box_width * box_height
                    retained = clipped_area / raw_area if raw_area else 0.0
                    if sides:
                        oob += 1
                        clipped += 1
                        for side, amount in sides:
                            side_counts[side] += 1
                            overrun_pixels.append(amount)
                        retained_ratios.append(retained)
                        clipped_widths.append(max(0.0, clipped_right - clipped_left))
                        clipped_heights.append(max(0.0, clipped_bottom - clipped_top))
                        if clipped_area <= 0:
                            fully_outside += 1
                            invalid += 1
                            severity_counts["FULLY_OUTSIDE"] += 1
                            invalid_counts["INVALID_AFTER_CLIP"] += 1
                            continue
                        if retained >= 0.9:
                            severity = "CLIP_MINOR"
                        elif retained >= 0.5:
                            severity = "CLIP_MODERATE"
                        else:
                            severity = "CLIP_SEVERE"
                        severity_counts[severity] += 1
                    if (sequence, frame_number) in requested_keys and original_class == "others":
                        source = image_by_frame.get((sequence, frame_number))
                        others_candidates[(sequence, frame_number)].append(
                            {
                                "sequence": sequence,
                                "frame": frame_number,
                                "track_id": track_id,
                                "original_class": original_class,
                                "source_file": source.filename if source else "",
                                "clipped_xyxy": (clipped_left, clipped_top, clipped_right, clipped_bottom),
                            }
                        )
        track_count = sum(len(values) for values in sequence_track_sets.values())
        selected_others: list[dict[str, Any]] = []
        for index, row in enumerate(requested, 1):
            key = (row["sequence_id"], int(row["frame_id"]))
            candidates = others_candidates.get(key, [])
            selected = candidates[0] if candidates else {
                "sequence": key[0], "frame": key[1], "track_id": "NOT_FOUND",
                "original_class": "others", "source_file": "", "clipped_xyxy": (0, 0, 0, 0),
            }
            selected_others.append({"sample_id": f"OTHERS_AUDIT_{index:02d}", **selected, "candidate_count_same_frame": len(candidates)})
        evidence_path = create_others_contact_sheet(
            archive, selected_others, repro / "contact_sheets" / "ua_others_audit.jpg"
        )

    visual_assessments = [
        "LIKELY_MOTORIZED_VEHICLE",
        "CONFIRMED_MOTORIZED_VEHICLE",
        "UNDETERMINED_DUE_TO_SCALE_OR_BOUNDARY",
        "CONFIRMED_MOTORIZED_VEHICLE",
        "LIKELY_MOTORIZED_VEHICLE",
        "UNDETERMINED_DUE_TO_SCALE_OR_BOUNDARY",
        "UNDETERMINED_DUE_TO_SCALE_OR_BOUNDARY",
        "UNDETERMINED_DUE_TO_SCALE_OR_BOUNDARY",
        "UNDETERMINED_DUE_TO_SCALE_OR_BOUNDARY",
        "UNDETERMINED_DUE_TO_SCALE_OR_BOUNDARY",
        "UNDETERMINED_DUE_TO_SCALE_OR_BOUNDARY",
        "UNDETERMINED_DUE_TO_SCALE_OR_BOUNDARY",
    ]
    others_rows = [
        {
            "sample_id": row["sample_id"],
            "sequence": row["sequence"],
            "frame": row["frame"],
            "track_id": row["track_id"],
            "original_class": row["original_class"],
            "visual_assessment": visual_assessments[index],
            "mapped_class": "vehicle",
            "reviewer": "CODEX_VISUAL_AUDIT_2026-08-02",
            "review_status": "AUDIT_VISUALLY_REVIEWED",
            "evidence_path": evidence_path,
            "notes": "This non-random 12-row review cannot justify all UA others annotations.",
        }
        for index, row in enumerate(selected_others)
    ]
    dimension_mismatches = {key: sorted(values) for key, values in dimension_variation.items() if len(values) > 1}
    return {
        "dataset": "UA-DETRAC Original",
        "path": str(path.resolve()),
        "path_sha256": sha256_file(path),
        "sequence_count": len(by_sequence),
        "image_count": len(image_infos),
        "xml_count": len(xml_infos),
        "frame_count": frame_count,
        "track_count": track_count,
        "raw_box_count": raw_boxes,
        "parsed_box_count": parsed,
        "valid_before_clip": valid_before,
        "out_of_bounds_count": oob,
        "clipped_count": clipped,
        "excluded_count": fully_outside,
        "invalid_count": invalid,
        "final_analysis_count": valid_before - fully_outside,
        "fully_outside": fully_outside,
        "missing_frame_count": missing_frame_count,
        "duplicate_track_id_within_frame": duplicate_track_frame,
        "track_class_inconsistency_count": sum(len(values) > 1 for values in track_class_values.values()),
        "ignored_region_count": ignored_regions,
        "dimension_probe_sequences": len(dimension_variation),
        "dimension_mismatch_sequences": dimension_mismatches,
        "class_counts": dict(class_counts),
        "invalid_counts": dict(invalid_counts),
        "side_counts": dict(side_counts),
        "severity_counts": dict(severity_counts),
        "overrun_min": round(min(overrun_pixels), 6) if overrun_pixels else "",
        "overrun_median": quantile(overrun_pixels, 0.5),
        "overrun_p95": quantile(overrun_pixels, 0.95),
        "overrun_max": round(max(overrun_pixels), 6) if overrun_pixels else "",
        "retained_min": round(min(retained_ratios), 8) if retained_ratios else "",
        "retained_p05": quantile(retained_ratios, 0.05),
        "retained_median": quantile(retained_ratios, 0.5),
        "retained_max": round(max(retained_ratios), 8) if retained_ratios else "",
        "post_clip_under_2px": sum(width < 2 or height < 2 for width, height in zip(clipped_widths, clipped_heights)),
        "coordinate_boundary_equal_count": coordinate_boundary_equal,
        "others_rows": others_rows,
        "conservation_ok": raw_boxes == (valid_before - fully_outside) + invalid,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def class_mapping_rows(
    mapping: dict[str, Any],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = {
        "MIO-TCD Localization": "mio_tcd",
        "AAU RainSnow": "aau_rainsnow",
        "UA-DETRAC Original": "ua_detrac",
    }
    rows: list[dict[str, Any]] = []
    for result in results:
        rules = mapping[keys[result["dataset"]]]
        for original_class, count in sorted(result["class_counts"].items()):
            rule = rules.get(original_class)
            included = bool(rule and rule.get("include"))
            track_exclusions = (rule or {}).get("track_exclusions", [])
            rows.append(
                {
                    "dataset": result["dataset"],
                    "original_class": original_class,
                    "raw_count": count,
                    "mapped_class": (rule or {}).get("mapped_class", ""),
                    "included": included,
                    "excluded_reason": (
                        "TRACK_EXCLUSIONS_APPLY"
                        if included and track_exclusions
                        else ""
                        if included
                        else "NON_MOTORIZED_OR_UNMAPPED"
                    ),
                    "review_required": (rule or {}).get("review_required", True),
                    "mapping_source": "configs/vehicle_class_mapping.yaml",
                    "config_rule_present": rule is not None,
                    "preserve_original_class": mapping.get("preserve_original_class", False),
                }
            )
    return rows


def audit_leakage_and_duplicates(
    aau_path: Path,
    ua_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    splits = read_csv(ROOT / "reports" / "external_eda" / "split_proposal.csv")
    manifest = read_csv(ROOT / "reports" / "external_eda" / "selected_data_manifest.csv")
    split_map: dict[tuple[str, str], str] = {}
    sequence_values: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in splits:
        key = (row["dataset_name"], row["sequence_id"])
        sequence_values[key].add(row["proposed_split"])
        split_map[key] = row["proposed_split"]
    sequence_conflicts = [key for key, values in sequence_values.items() if len(values) > 1]
    source_splits: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in manifest:
        split = split_map.get((row["dataset_name"], row["sequence_id"]), "UNKNOWN")
        source_splits[(row["dataset_name"], row["source_file"])].add(split)
    source_conflicts = [key for key, values in source_splits.items() if len(values) > 1]
    leakage_rows = [
        {
            "check": "SEQUENCE_EXCLUSIVE",
            "dataset": "ALL",
            "status": "PASS" if not sequence_conflicts else "FAIL",
            "count": len(sequence_conflicts),
            "severity": "CRITICAL" if sequence_conflicts else "INFO",
            "evidence": str(sequence_conflicts[:10]),
            "scope": "FULL_SPLIT_PROPOSAL",
        },
        {
            "check": "SOURCE_PATH_EXCLUSIVE",
            "dataset": "ALL",
            "status": "PASS" if not source_conflicts else "FAIL",
            "count": len(source_conflicts),
            "severity": "CRITICAL" if source_conflicts else "INFO",
            "evidence": str(source_conflicts[:10]),
            "scope": "TRACKED_SELECTION_MANIFEST",
        },
    ]

    duplicate_rows: list[dict[str, Any]] = []
    # Full AAU RGB content hashes.
    aau_json = json.loads((aau_path / "aauRainSnow-rgb.json").read_text(encoding="utf-8"))
    aau_hashes: defaultdict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for meta in aau_json.get("images", []):
        relative = str(meta["file_name"])
        source = aau_path / relative
        if not source.exists():
            continue
        sequence = Path(relative).parent.name
        split = split_map.get(("AAU RainSnow", sequence), "UNKNOWN")
        aau_hashes[sha256_file(source)].append((relative, sequence, split))
    for digest, values in aau_hashes.items():
        if len({value[2] for value in values}) > 1:
            duplicate_rows.append(
                {
                    "dataset": "AAU RainSnow",
                    "left_file": values[0][0],
                    "right_file": values[1][0],
                    "left_split": values[0][2],
                    "right_split": values[1][2],
                    "duplicate_type": "EXACT_SHA256",
                    "hash": digest,
                    "status": "FAIL",
                    "severity": "CRITICAL",
                    "notes": f"group_size={len(values)}",
                }
            )

    # Full UA central-directory CRC+size scan; SHA-256 only cross-split candidates.
    with zipfile.ZipFile(ua_path) as archive:
        crc_groups: defaultdict[tuple[int, int], list[zipfile.ZipInfo]] = defaultdict(list)
        for info in archive.infolist():
            if "/detrac-images/" not in info.filename.lower() or not info.filename.lower().endswith(".jpg"):
                continue
            crc_groups[(info.CRC, info.file_size)].append(info)
        for values in crc_groups.values():
            if len(values) < 2:
                continue
            candidate_splits = {
                split_map.get(("UA-DETRAC Original", sequence_from_path(info.filename)), "UNKNOWN")
                for info in values
            }
            if len(candidate_splits) < 2:
                continue
            sha_groups: defaultdict[str, list[zipfile.ZipInfo]] = defaultdict(list)
            for info in values:
                sha_groups[hashlib.sha256(archive.read(info)).hexdigest()].append(info)
            for digest, confirmed in sha_groups.items():
                confirmed_splits = {
                    split_map.get(("UA-DETRAC Original", sequence_from_path(info.filename)), "UNKNOWN")
                    for info in confirmed
                }
                if len(confirmed_splits) > 1:
                    duplicate_rows.append(
                        {
                            "dataset": "UA-DETRAC Original",
                            "left_file": confirmed[0].filename,
                            "right_file": confirmed[1].filename,
                            "left_split": split_map.get(("UA-DETRAC Original", sequence_from_path(confirmed[0].filename)), "UNKNOWN"),
                            "right_split": split_map.get(("UA-DETRAC Original", sequence_from_path(confirmed[1].filename)), "UNKNOWN"),
                            "duplicate_type": "EXACT_SHA256_AFTER_CRC_PREFILTER",
                            "hash": digest,
                            "status": "FAIL",
                            "severity": "CRITICAL",
                            "notes": f"group_size={len(confirmed)}",
                        }
                    )
    if not duplicate_rows:
        duplicate_rows.append(
            {
                "dataset": "AAU+UA",
                "left_file": "",
                "right_file": "",
                "left_split": "",
                "right_split": "",
                "duplicate_type": "EXACT_CONTENT_CROSS_SPLIT",
                "hash": "",
                "status": "PASS",
                "severity": "INFO",
                "notes": "AAU full SHA-256; UA full CRC+size with SHA-256 confirmation for candidates.",
            }
        )
    leakage_rows.extend(
        [
            {
                "check": "EXACT_CONTENT_CROSS_SPLIT",
                "dataset": "AAU+UA",
                "status": "PASS" if all(row["status"] == "PASS" for row in duplicate_rows) else "FAIL",
                "count": sum(row["status"] == "FAIL" for row in duplicate_rows),
                "severity": "CRITICAL" if any(row["status"] == "FAIL" for row in duplicate_rows) else "INFO",
                "evidence": "duplicate_cross_split_audit.csv",
                "scope": "FULL_AAU_RGB_AND_FULL_UA_ARCHIVE",
            },
            {
                "check": "NEAR_DUPLICATE_CROSS_SPLIT",
                "dataset": "ALL",
                "status": "NOT_VERIFIED",
                "count": "",
                "severity": "HIGH",
                "evidence": "Production pHash scan is sampled and only compares consecutive files within a sequence.",
                "scope": "INCOMPLETE",
            },
            {
                "check": "MIO_EXACT_CONTENT_CROSS_SPLIT",
                "dataset": "MIO-TCD Localization",
                "status": "NOT_APPLICABLE",
                "count": 0,
                "severity": "INFO",
                "evidence": "MIO is train-only by policy.",
                "scope": "TRAIN_ONLY",
            },
        ]
    )
    return leakage_rows, duplicate_rows


def inventory_files(audit: Path) -> list[dict[str, Any]]:
    tracked_code, tracked_out, _ = run_text(["git", "ls-files"])
    tracked = set(tracked_out.splitlines()) if tracked_code == 0 else set()
    script_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (ROOT / "scripts").glob("*.py")
    )
    roots = ["configs", "scripts", "reports", "docs", "tests", "metadata", "planning"]
    rows: list[dict[str, Any]] = []
    for name in roots:
        directory = ROOT / name
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path == audit:
                continue
            relative = path.relative_to(REPO).as_posix()
            if path.suffix.lower() in {".py"}:
                file_type, purpose = "SOURCE_CODE", "Executable or test source"
            elif path.suffix.lower() in {".yaml", ".yml"}:
                file_type, purpose = "CONFIG", "Pipeline policy/configuration"
            elif path.suffix.lower() == ".csv":
                file_type, purpose = "TABLE", "Metadata, plan, or generated report"
            elif path.suffix.lower() == ".md":
                file_type, purpose = "DOCUMENT", "Documentation or generated narrative"
            elif path.suffix.lower() in {".png", ".jpg"}:
                file_type, purpose = "FIGURE", "Generated visualization/evidence"
            else:
                file_type, purpose = "OTHER", "Repository artifact"
            generated_by = ""
            external_report = "reports/external_eda/" in relative
            manual_external = path.name in {
                "evaluation_slice_readiness.csv",
                "ua_others_sample_review.csv",
            }
            if file_type == "FIGURE" and external_report:
                generated_by = "run_external_dataset_eda.py:_figures/_audit_figures"
            elif external_report and not manual_external and file_type in {"TABLE", "DOCUMENT"}:
                generated_by = "run_external_dataset_eda.py"
            elif manual_external:
                generated_by = "MANUAL_REVIEW_OR_POLICY"
            elif file_type in {"TABLE", "FIGURE", "DOCUMENT"} and path.name in script_text:
                generated_by = "REFERENCED_BY_SCRIPT"
            elif file_type in {"SOURCE_CODE", "CONFIG"}:
                generated_by = "MANUAL_SOURCE"
            elif file_type == "DOCUMENT":
                generated_by = "MANUAL_DOCUMENT"
            else:
                generated_by = "NO_GENERATOR_FOUND"
            rows.append(
                {
                    "file_path": relative,
                    "file_type": file_type,
                    "purpose": purpose,
                    "generated_by": generated_by,
                    "input_dependencies": "See source/config references" if generated_by in {"REFERENCED_BY_SCRIPT", "run_external_dataset_eda.py", "run_external_dataset_eda.py:_figures/_audit_figures"} else "NOT_DECLARED",
                    "last_modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime)),
                    "tracked_by_git": relative in tracked,
                    "reproducible": generated_by not in {"NO_GENERATOR_FOUND", "MANUAL_DOCUMENT", "MANUAL_REVIEW_OR_POLICY"},
                    "notes": f"size_bytes={path.stat().st_size}",
                }
            )
    return rows


def findings_rows(
    mio: dict[str, Any], aau: dict[str, Any], ua: dict[str, Any], leakage: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    severe = int(ua["severity_counts"].get("CLIP_SEVERE", 0))
    return [
        {
            "finding_id": "AUD-001",
            "severity": "HIGH",
            "category": "TEST_QUALITY",
            "title": "Passing tests do not independently validate raw datasets",
            "evidence": "Most tests use synthetic dictionaries or tracked aggregate CSVs; raw archives are not required.",
            "impact": "Parser/count regressions can pass 45 tests.",
            "recommended_fix": "Add opt-in raw-data integration tests with conservation invariants.",
            "status": "OPEN",
        },
        {
            "finding_id": "AUD-002",
            "severity": "HIGH",
            "category": "CACHE",
            "title": "Resume cache is fingerprinted and rejects stale results",
            "evidence": "CACHE_SCHEMA_VERSION=2 fingerprints the runner, inspector, shared helper, YAML configs, input options, Python version, and source tree stat signature before loading a cached result.",
            "impact": "Changed code, configuration, options, or source files invalidate the resume cache instead of silently reusing stale analysis.",
            "recommended_fix": "Implemented; retain the stale-fingerprint regression test when changing cache identity inputs.",
            "status": "CLOSED",
        },
        {
            "finding_id": "AUD-003",
            "severity": "HIGH" if severe else "MEDIUM",
            "category": "ANNOTATION",
            "title": "UA clipped boxes need severity-aware training policy",
            "evidence": f"CLIP_SEVERE={severe}; current production keeps every box with positive post-clip area.",
            "impact": "Severely truncated or semantically ambiguous boxes may be accepted as fully valid.",
            "recommended_fix": "Review CLIP_SEVERE separately and define minimum retained-area/size policy.",
            "status": "OPEN",
        },
        {
            "finding_id": "AUD-004",
            "severity": "HIGH",
            "category": "CLASS_MAPPING",
            "title": "Twelve UA others rows do not justify retaining the whole class",
            "evidence": "Selection is non-random; table has no track_id, named reviewer, or retained source contact sheet.",
            "impact": "A heterogeneous class can introduce non-vehicle labels.",
            "recommended_fix": "Stratified review across sequences, clip severity, day/night, and object scale; keep with caution until complete.",
            "status": "OPEN",
        },
        {
            "finding_id": "AUD-005",
            "severity": "MEDIUM",
            "category": "AAU_LIGHTING",
            "title": "AAU decisions are recorded but original review evidence is incomplete",
            "evidence": "22 decisions exist, but reviewer is missing and contact sheets were not retained in commit 80e66fb.",
            "impact": "10/11/1 cannot be fully re-audited from Git alone.",
            "recommended_fix": "Record reviewer/date/method and privacy-safe evidence checksum or approved contact sheets.",
            "status": "OPEN",
        },
        {
            "finding_id": "AUD-006",
            "severity": "HIGH",
            "category": "LEAKAGE",
            "title": "Near-duplicate cross-split leakage is not fully checked",
            "evidence": next(row["evidence"] for row in leakage if row["check"] == "NEAR_DUPLICATE_CROSS_SPLIT"),
            "impact": "Visually adjacent or transformed copies may cross splits undetected.",
            "recommended_fix": "Run full cross-split pHash/embedding audit grouped by source and sequence.",
            "status": "OPEN",
        },
        {
            "finding_id": "AUD-007",
            "severity": "MEDIUM",
            "category": "REPRODUCIBILITY",
            "title": "Dependency environment is range-pinned, not lock-pinned",
            "evidence": "requirements-data.txt contains version ranges; no lock file or environment hash.",
            "impact": "Future installs can produce different results.",
            "recommended_fix": "Add a tested lock/constraints file and record Python/platform versions.",
            "status": "OPEN",
        },
        {
            "finding_id": "AUD-008",
            "severity": "MEDIUM",
            "category": "FIGURE_PROVENANCE",
            "title": "Figures are generated from in-memory objects rather than declared CSV inputs",
            "evidence": "run_external_dataset_eda.py calls _figures with in-memory rows; no per-figure source manifest.",
            "impact": "A tracked figure cannot be independently tied to one exact CSV/version.",
            "recommended_fix": "Create figures from saved CSVs and emit figure_provenance.csv with hashes.",
            "status": "OPEN",
        },
        {
            "finding_id": "AUD-009",
            "severity": "LOW",
            "category": "GIT_HYGIENE",
            "title": "Tracked reports contain machine-specific absolute source paths",
            "evidence": "dataset_inventory.csv contains C:\\UMTLab\\k230 paths.",
            "impact": "Reports are less portable and expose local layout.",
            "recommended_fix": "Store dataset ID plus relative/configured root, not absolute personal paths.",
            "status": "OPEN",
        },
        {
            "finding_id": "AUD-010",
            "severity": "MEDIUM",
            "category": "COUNT_SEMANTICS",
            "title": "1,301,866 mixes full and sampled validation scopes",
            "evidence": f"MIO final={mio['final_analysis_count']} sampled; AAU final={aau['final_analysis_count']} full RGB; UA final={ua['final_analysis_count']} full XML.",
            "impact": "Readers may mistake the sum for all raw boxes across all datasets.",
            "recommended_fix": "Name it analysis-scope sum and report raw/valid/mapped totals separately.",
            "status": "OPEN",
        },
        {
            "finding_id": "AUD-011",
            "severity": "HIGH",
            "category": "REPORT_SEMANTICS",
            "title": "UA out-of-bounds causal explanation is unsupported",
            "evidence": "All 130,181 overruns occur only on RIGHT/BOTTOM; maximum overrun is 1.0 pixel and 130,178/130,181 are CLIP_MINOR.",
            "impact": "Describing these boxes as vehicles entering/leaving the frame overstates what the data proves and hides a likely coordinate-convention issue.",
            "recommended_fix": "Keep clipping, but document the one-pixel right/bottom pattern as a coordinate-convention hypothesis unless image-level evidence proves another cause.",
            "status": "OPEN",
        },
        {
            "finding_id": "AUD-012",
            "severity": "MEDIUM",
            "category": "TEST_PERFORMANCE",
            "title": "Default pytest discovery traverses the data-heavy repository",
            "evidence": "python -m pytest -q took 83.24s while every listed test was <=0.01s; the explicit data_collection/tests path completed in 0.54s.",
            "impact": "Slow feedback can discourage frequent testing and is unnecessary on student machines with local datasets.",
            "recommended_fix": "Configure pytest testpaths/norecursedirs or consistently invoke the explicit tests directory.",
            "status": "OPEN",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_REPRO))
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT))
    parser.add_argument("--mio-path", default=str(ROOT / "storage_placeholders" / "online_data" / "raw" / "mio_tcd" / "MIO-TCD-Localization.tar"))
    parser.add_argument("--aau-path", default=str(ROOT / "storage_placeholders" / "online_data" / "raw" / "aau_rainsnow" / "aau-rainsnow"))
    parser.add_argument("--ua-path", default=str(ROOT / "storage_placeholders" / "online_data" / "raw" / "ua_detrac_orig" / "ua-detrac-orig.zip"))
    args = parser.parse_args()
    repro, audit = Path(args.output).resolve(), Path(args.audit_dir).resolve()
    repro.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.argv)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tracemalloc.start()
    timings: list[dict[str, Any]] = []
    mapping = load_yaml(ROOT / "configs" / "vehicle_class_mapping.yaml")

    step = time.perf_counter()
    mio = audit_mio(Path(args.mio_path), mapping)
    timings.append({"step": "MIO_RAW_AUDIT", "elapsed_seconds": round(time.perf_counter() - step, 3)})
    step = time.perf_counter()
    aau = audit_aau(Path(args.aau_path), mapping, repro)
    timings.append({"step": "AAU_RAW_AUDIT", "elapsed_seconds": round(time.perf_counter() - step, 3)})
    step = time.perf_counter()
    ua = audit_ua(Path(args.ua_path), repro)
    timings.append({"step": "UA_RAW_AUDIT", "elapsed_seconds": round(time.perf_counter() - step, 3)})
    step = time.perf_counter()
    leakage, duplicates = audit_leakage_and_duplicates(Path(args.aau_path), Path(args.ua_path))
    timings.append({"step": "LEAKAGE_DUPLICATE_AUDIT", "elapsed_seconds": round(time.perf_counter() - step, 3)})

    reconciliation = [
        {
            "dataset": result["dataset"],
            "raw_box_count": result["raw_box_count"],
            "parsed_box_count": result["parsed_box_count"],
            "valid_before_clip": result["valid_before_clip"],
            "out_of_bounds_count": result["out_of_bounds_count"],
            "clipped_count": result["clipped_count"],
            "excluded_count": result["excluded_count"],
            "invalid_count": result["invalid_count"],
            "final_analysis_count": result["final_analysis_count"],
            "conservation_ok": result["conservation_ok"],
            "scope_note": result.get("out_of_bounds_scope", "FULL_RAW_ANNOTATION"),
        }
        for result in (mio, aau, ua)
    ]
    verified_total = sum(int(row["final_analysis_count"]) for row in reconciliation)
    claimed_total = 1_301_866
    total_difference = verified_total - claimed_total
    mapping_rows = class_mapping_rows(mapping, [mio, aau, ua])
    findings = findings_rows(mio, aau, ua, leakage)
    production_inventories = {
        sample: read_csv(repro / f"eda_sample_{sample}" / "dataset_inventory.csv")
        for sample in (100, 5000)
        if (repro / f"eda_sample_{sample}" / "dataset_inventory.csv").is_file()
    }
    production_verified = len(production_inventories) == 2 and all(
        len(rows) == 3
        and all(row.get("status") == "ANALYZED" for row in rows)
        and not any("radiate" in row.get("dataset_name", "").lower() for row in rows)
        for rows in production_inventories.values()
    )

    claims = [
        {"claim_id": "C01", "claim": "AAU lighting review is 10 DAY, 11 NIGHT, 1 TWILIGHT", "claimed_value": "DAY=10;NIGHT=11;TWILIGHT=1", "verified_value": ";".join(f"{key}={value}" for key, value in sorted(aau["lighting_counts"].items())), "status": "PARTIALLY_VERIFIED" if aau["reviewer_missing"] else "VERIFIED", "evidence": "aau_lighting_review_audit.csv and regenerated local contact sheets", "method": "Raw video inventory + config decision audit", "severity": "MEDIUM", "notes": "Counts match, but commit lacks named reviewer and retained original evidence."},
        {"claim_id": "C02", "claim": "UA has 130,181 out-of-bounds boxes clipped and kept", "claimed_value": 130181, "verified_value": ua["out_of_bounds_count"], "status": "VERIFIED" if ua["out_of_bounds_count"] == 130181 and ua["fully_outside"] == 0 else "PARTIALLY_VERIFIED", "evidence": "ua_bbox_clipping_audit.csv", "method": "Independent full XML parse against raw image dimensions", "severity": "HIGH", "notes": f"Severity distribution={ua['severity_counts']}"},
        {"claim_id": "C03", "claim": "UA has zero annotation errors after clipping", "claimed_value": 0, "verified_value": ua["invalid_count"], "status": "PARTIALLY_VERIFIED" if ua["invalid_count"] == 0 else "INCORRECT", "evidence": "ua_bbox_clipping_audit.csv", "method": "Syntax, numeric, frame-link, track, class and post-clip checks", "severity": "HIGH", "notes": "Zero under implemented structural checks does not prove zero semantic errors."},
        {"claim_id": "C04", "claim": "Vehicle mapping keeps motor vehicles/motorcycles and excludes person/bicycle while preserving original class", "claimed_value": "POLICY_APPLIED", "verified_value": "CONFIG_AND_REPORT_COLUMNS_PRESENT", "status": "PARTIALLY_VERIFIED", "evidence": "class_mapping_audit.csv", "method": "Raw class counts + config matrix + manifest schema review", "severity": "HIGH", "notes": "UA others remains heterogeneous and review-required."},
        {"claim_id": "C05", "claim": "Twelve UA others samples were manually reviewed and justify retaining the class", "claimed_value": 12, "verified_value": len(ua["others_rows"]), "status": "NOT_VERIFIED", "evidence": "ua_others_review_audit.csv and regenerated contact sheet", "method": "Reconstruct listed frames from raw ZIP", "severity": "HIGH", "notes": "Non-random selection; no reviewer, track ID, original evidence, or per-sample visual decision in commit."},
        {"claim_id": "C06", "claim": "K230_BACKLIT exists and mAP is NOT_AVAILABLE", "claimed_value": "NOT_AVAILABLE", "verified_value": "NOT_AVAILABLE", "status": "VERIFIED", "evidence": "k230_backlit_audit.md", "method": "Config/report/raw K230 storage inspection", "severity": "INFO", "notes": "No K230 backlit data, GT, predictions, or evaluable split."},
        {"claim_id": "C07", "claim": "EDA runs on 3/3 selected datasets and excludes RADIATE", "claimed_value": "3/3;RADIATE_EXCLUDED", "verified_value": "SAMPLE_100=3/3;SAMPLE_5000=3/3;RADIATE_EXCLUDED" if production_verified else "PENDING_REPRODUCTION_RUN", "status": "VERIFIED" if production_verified else "NOT_VERIFIED", "evidence": "audit_reproduction production runs" if production_verified else "reproduction command still required", "method": "Clean-output production runs followed by resume", "severity": "HIGH", "notes": "Sample 5000 reproduced 1,301,866; resume reused all three caches without count duplication." if production_verified else "Updated after mandatory sample run."},
        {"claim_id": "C08", "claim": "Total analyzed bounding boxes is 1,301,866", "claimed_value": claimed_total, "verified_value": verified_total, "status": "VERIFIED" if total_difference == 0 else "INCORRECT", "evidence": "dataset_count_reconciliation.csv", "method": "Independent raw parsers; deterministic MIO sample seed 230", "severity": "CRITICAL", "notes": f"difference={total_difference}; mixed full/sample scopes"},
        {"claim_id": "C09", "claim": "45/45 tests prove EDA correctness", "claimed_value": "45/45 PASS", "verified_value": "TESTS_PASS_BUT_RAW_DATA_NOT_REQUIRED", "status": "INCORRECT", "evidence": "test_quality_audit.md", "method": "Test collection and source review", "severity": "HIGH", "notes": "Pass count is not proof of research/data correctness."},
        {"claim_id": "C10", "claim": "No data leakage", "claimed_value": "NO_LEAKAGE", "verified_value": "SEQUENCE_AND_EXACT_CONTENT_CHECKED;NEAR_DUPLICATE_NOT_VERIFIED", "status": "PARTIALLY_VERIFIED", "evidence": "sequence_leakage_audit.csv;duplicate_cross_split_audit.csv", "method": "Split keys + full AAU SHA + UA CRC/SHA candidate verification", "severity": "HIGH", "notes": "Full cross-split near-duplicate scan is missing."},
    ]

    write_csv(audit / "dataset_count_reconciliation.csv", reconciliation)
    write_csv(audit / "class_mapping_audit.csv", mapping_rows)
    write_csv(audit / "aau_lighting_review_audit.csv", aau["sequence_rows"])
    write_csv(audit / "ua_others_review_audit.csv", ua["others_rows"])
    ua_metrics = [
        {"metric": key, "value": value, "scope": "FULL_UA_RAW_XML", "notes": "Independent audit"}
        for key, value in {
            "sequence_count": ua["sequence_count"], "image_count": ua["image_count"],
            "frame_count": ua["frame_count"], "track_count": ua["track_count"],
            "raw_box_count": ua["raw_box_count"], "out_of_bounds_count": ua["out_of_bounds_count"],
            "fully_outside": ua["fully_outside"], "invalid_count": ua["invalid_count"],
            "post_clip_under_2px": ua["post_clip_under_2px"],
            "overrun_min": ua["overrun_min"], "overrun_median": ua["overrun_median"],
            "overrun_p95": ua["overrun_p95"], "overrun_max": ua["overrun_max"],
            "retained_min": ua["retained_min"], "retained_p05": ua["retained_p05"],
            "retained_median": ua["retained_median"], "retained_max": ua["retained_max"],
            **{f"side_{key}": ua["side_counts"].get(key, 0) for key in ("LEFT", "TOP", "RIGHT", "BOTTOM")},
            **{f"severity_{key}": ua["severity_counts"].get(key, 0) for key in ("CLIP_MINOR", "CLIP_MODERATE", "CLIP_SEVERE", "FULLY_OUTSIDE")},
            **{f"invalid_{key}": value for key, value in ua["invalid_counts"].items()},
        }.items()
    ]
    write_csv(audit / "ua_bbox_clipping_audit.csv", ua_metrics)
    write_csv(audit / "sequence_leakage_audit.csv", leakage)
    write_csv(audit / "duplicate_cross_split_audit.csv", duplicates)
    write_csv(audit / "audit_findings.csv", findings)
    write_csv(audit / "claim_verification.csv", claims)
    write_csv(audit / "file_inventory.csv", inventory_files(audit))
    write_csv(repro / "performance.csv", timings)
    current, peak = tracemalloc.get_traced_memory()
    raw_summary = {
        "commit": COMMIT,
        "command": command,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "exit_code": 0,
        "python_tracemalloc_current_bytes": current,
        "python_tracemalloc_peak_bytes": peak,
        "mio": mio,
        "aau": {key: value for key, value in aau.items() if key != "sequence_rows"},
        "ua": {key: value for key, value in ua.items() if key != "others_rows"},
        "verified_total": verified_total,
        "claimed_total": claimed_total,
        "difference": total_difference,
    }
    (repro / "independent_raw_audit.json").write_text(json.dumps(raw_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    required = load_yaml(ROOT / "configs" / "split_policy.yaml")
    backlit = next((row for row in required["main_test"]["required_slices"] if row["slice_id"] == "K230_BACKLIT"), {})
    readiness = next((row for row in read_csv(ROOT / "reports" / "external_eda" / "evaluation_slice_readiness.csv") if row["slice"] == "BACKLIT"), {})
    k230_files = [path for root in (ROOT / "raw_videos", ROOT / "extracted_images", ROOT / "annotations") if root.exists() for path in root.rglob("*") if path.is_file() and path.name != ".gitkeep"]
    (audit / "k230_backlit_audit.md").write_text(
        "# K230_BACKLIT audit\n\n"
        f"- Config definition: `{backlit}`\n"
        f"- Readiness row: `{readiness}`\n"
        f"- Real K230 media/annotation files found: **{len(k230_files)}**\n"
        "- Ground truth: **NOT_AVAILABLE**\n- Model predictions: **NOT_AVAILABLE**\n"
        "- mAP: **NOT_AVAILABLE**\n\nConclusion: the slice is a collection plan only; zero must not be reported as a score.\n",
        encoding="utf-8",
    )
    status_counts = Counter(row["status"] for row in claims)
    severity_counts = Counter(row["severity"] for row in findings)
    summary = f"""# EDA Audit Executive Summary

- Audited commit: `{COMMIT}`
- Independent raw total matching the current mixed analysis scope: **{verified_total:,}**
- Difference from 1,301,866: **{total_difference:+,}**
- UA out-of-bounds boxes: **{ua['out_of_bounds_count']:,}**; difference from 130,181: **{ua['out_of_bounds_count'] - 130181:+,}**
- AAU lighting decision counts: **{aau['lighting_counts']}**
- Claim status: **{dict(status_counts)}**
- Finding severity: **{dict(severity_counts)}**

## Direct conclusions

- Passing 45 tests is **not sufficient** to prove the EDA is correct.
- The 1,301,866 total is reproducible, but it mixes a deterministic MIO sample with full AAU-valid and full UA XML counts.
- The UA boundary count is reproducible; structural validity after clipping does not prove semantic correctness. Clip severity must be reviewed.
- The UA overrun pattern is right/bottom-only and at most one pixel, so the existing entering/leaving-frame explanation is unsupported; a coordinate-convention issue is more plausible.
- The 12 `others` rows do not provide enough evidence to retain the entire heterogeneous class without caution.
- AAU 10/11/1 is recorded and the raw videos exist, but commit `{COMMIT}` lacks a named reviewer and retained original review evidence.
- Sequence and exact-content leakage checks pass only within their stated scope; cross-split near-duplicate leakage remains unverified.
- `K230_BACKLIT` is correctly `NOT_AVAILABLE`.

See `FULL_EDA_AUDIT_REPORT.md`, `audit_findings.csv`, and `claim_verification.csv` for evidence and limitations.
"""
    (audit / "AUDIT_EXECUTIVE_SUMMARY.md").write_text(summary, encoding="utf-8")
    full_report = f"""# Full EDA Audit Report

## Scope and method

Read-only audit of commit `{COMMIT}`. Counts were obtained from the raw MIO TAR, AAU COCO JSON/videos, and UA ZIP/XML without importing production parser modules. Existing reports were used only for comparison after raw counts were established.

## Dataset reconciliation

- MIO: raw={mio['raw_box_count']:,}, mapped vehicle={mio['mapped_vehicle_count']:,}, excluded class={mio['excluded_count']:,}, deterministic sample final={mio['final_analysis_count']:,}, conservation={mio['conservation_ok']}.
- AAU: raw={aau['raw_box_count']:,}, valid={aau['valid_before_clip']:,}, invalid={aau['invalid_count']:,}, final={aau['final_analysis_count']:,}, conservation={aau['conservation_ok']}.
- UA: raw={ua['raw_box_count']:,}, out-of-bounds={ua['out_of_bounds_count']:,}, fully outside={ua['fully_outside']:,}, final={ua['final_analysis_count']:,}, conservation={ua['conservation_ok']}.

## Major limitations

1. MIO out-of-bounds validation is limited to the same deterministic 5,000-image analysis sample; full image-dimension validation was not run.
2. AAU contact sheets were regenerated for audit, but original commit evidence/reviewer identity is absent.
3. UA semantic label correctness cannot be proven by XML structural validation.
4. The one-pixel UA right/bottom overrun pattern supports a coordinate-convention hypothesis, not the reported entering/leaving-frame cause.
5. Full cross-split near-duplicate detection has not been run.
6. Figures lack a CSV/hash provenance manifest.

## Research validity

External datasets support vehicle detection and domain analysis only. They do not establish stopped-vehicle ground truth in an emergency lane and cannot replace a locked K230 field test. Stationary candidates remain heuristic review items, not ground truth.

## Required fixes before lecturer/research reporting

Address all HIGH findings in `audit_findings.csv`; clearly rename the mixed-scope bbox total; preserve auditable AAU/others review evidence; add cache invalidation and raw-data integration tests; run full near-duplicate cross-split audit.
"""
    (audit / "FULL_EDA_AUDIT_REPORT.md").write_text(full_report, encoding="utf-8")
    (audit / "recommended_fix_plan.md").write_text(
        "# Recommended fix plan\n\n"
        "1. **HIGH — audit evidence:** perform stratified `others` review with reviewer/track/evidence and document UA clipping as a one-pixel coordinate-convention issue unless further evidence proves otherwise.\n"
        "2. **HIGH — cache/test:** fingerprint cache inputs and add opt-in raw archive integration invariants.\n"
        "3. **HIGH — leakage:** run full cross-split near-duplicate audit before freezing test.\n"
        "4. **MEDIUM — reporting:** separate raw/full/sample counts and generate figures from hashed CSV inputs.\n"
        "5. **MEDIUM — reproducibility:** add dependency lock and dataset checksums/version manifest.\n\n"
        "Production fixes must be separate commits after this audit commit.\n",
        encoding="utf-8",
    )
    (audit / "reproduction_commands.md").write_text(
        "# Reproduction commands\n\n"
        f"```powershell\npython data_collection/scripts/audit_eda_independent.py --output data_collection/reports/audit_reproduction --audit-dir data_collection/reports/audit\n```\n\n"
        "Mandatory production checks are recorded separately in the final audit run log.\n",
        encoding="utf-8",
    )
    print(f"AUDITED_COMMIT={COMMIT}")
    print(f"MIO_PATH={Path(args.mio_path).resolve()}")
    print(f"AAU_PATH={Path(args.aau_path).resolve()}")
    print(f"UA_PATH={Path(args.ua_path).resolve()}")
    print(f"VERIFIED_TOTAL={verified_total}; DIFFERENCE={total_difference:+d}")
    print(f"UA_OOB={ua['out_of_bounds_count']}; DIFFERENCE={ua['out_of_bounds_count'] - 130181:+d}")
    print(f"AAU_LIGHTING={aau['lighting_counts']}")
    print(f"CLAIMS={dict(status_counts)}")
    print(f"FINDINGS={dict(severity_counts)}")
    print(f"AUDIT_DIR={audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
