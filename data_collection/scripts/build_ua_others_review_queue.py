"""Build a deterministic, stratified visual-review queue for UA `others`.

The script never changes source annotations. It writes a CSV review queue and
optional contact sheets outside Git so reviewers can make explicit decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from PIL import Image

from external_eda_common import ROOT, read_csv, write_csv


DEFAULT_UA = (
    ROOT
    / "storage_placeholders"
    / "online_data"
    / "raw"
    / "ua_detrac_orig"
    / "ua-detrac-orig.zip"
)
DEFAULT_OUTPUT = ROOT / "reports" / "external_eda" / "ua_others_stratified_review_queue.csv"
DEFAULT_DECISIONS = ROOT / "reports" / "external_eda" / "ua_others_stratified_review_decisions.csv"
DEFAULT_CONTACTS = (
    ROOT
    / "storage_placeholders"
    / "online_data"
    / "contact_sheets"
    / "ua_others_stratified"
)

FIELDS = [
    "sample_id",
    "dataset_name",
    "sequence_id",
    "frame_id",
    "track_id",
    "source_file",
    "weather",
    "camera_state",
    "area_ratio",
    "size_bucket",
    "boundary_status",
    "truncation_ratio",
    "sampling_method",
    "visual_assessment",
    "mapped_class",
    "include_for_training",
    "preserve_original_class",
    "reviewer",
    "review_date",
    "review_status",
    "evidence_path",
    "notes",
]


def _sequence_from_path(name: str) -> str:
    for part in PurePosixPath(name).parts:
        if part.upper().startswith("MVI_"):
            return part
    return Path(name).stem


def _image_dimensions(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[int, int]:
    with Image.open(io.BytesIO(archive.read(info))) as image:
        return image.size


def _size_bucket(area_ratio: float) -> str:
    if area_ratio < 0.001:
        return "SMALL"
    if area_ratio < 0.02:
        return "MEDIUM"
    return "LARGE"


def _rank(candidate: dict[str, Any]) -> str:
    key = f"{candidate['sequence_id']}|{candidate['frame_id']}|{candidate['track_id']}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _select_stratified(candidates: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        groups[
            (
                candidate["weather"],
                candidate["camera_state"],
                candidate["size_bucket"],
                candidate["boundary_status"],
            )
        ].append(candidate)
    for values in groups.values():
        values.sort(key=_rank)

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int, str]] = set()
    sequence_counts: Counter[str] = Counter()
    group_keys = sorted(groups)

    for sequence_cap in (1, 2, 3, 5):
        made_progress = True
        while len(selected) < target_count and made_progress:
            made_progress = False
            for group_key in group_keys:
                for candidate in groups[group_key]:
                    key = (
                        candidate["sequence_id"],
                        candidate["frame_id"],
                        candidate["track_id"],
                    )
                    if key in selected_keys or sequence_counts[candidate["sequence_id"]] >= sequence_cap:
                        continue
                    selected.append(candidate)
                    selected_keys.add(key)
                    sequence_counts[candidate["sequence_id"]] += 1
                    made_progress = True
                    break
                if len(selected) >= target_count:
                    break
        if len(selected) >= target_count:
            break
    return selected


def _make_contact_sheets(
    archive: zipfile.ZipFile,
    selected: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles: list[np.ndarray] = []
    for row in selected:
        array = np.frombuffer(archive.read(row["source_file"]), dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            tiles.append(np.zeros((270, 480, 3), dtype=np.uint8))
            continue
        left, top, right, bottom = row["clipped_xyxy"]
        box_width = max(1.0, right - left)
        box_height = max(1.0, bottom - top)
        pad_x = max(40.0, box_width * 2.0)
        pad_y = max(30.0, box_height * 1.5)
        height, width = image.shape[:2]
        crop_left = max(0, int(left - pad_x))
        crop_top = max(0, int(top - pad_y))
        crop_right = min(width, int(right + pad_x))
        crop_bottom = min(height, int(bottom + pad_y))
        crop = image[crop_top:crop_bottom, crop_left:crop_right].copy()
        if crop.size == 0:
            crop = image.copy()
            crop_left = crop_top = 0
        cv2.rectangle(
            crop,
            (max(0, int(left) - crop_left), max(0, int(top) - crop_top)),
            (max(0, int(right) - crop_left), max(0, int(bottom) - crop_top)),
            (0, 255, 255),
            2,
        )
        tile = cv2.resize(crop, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (480, 30), (0, 0, 0), -1)
        title = (
            f"{row['sample_id']} {row['sequence_id']} f{row['frame_id']} "
            f"{row['size_bucket']} {row['boundary_status']}"
        )
        cv2.putText(tile, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
        tiles.append(tile)

    for page_index in range(0, len(tiles), 12):
        page_tiles = tiles[page_index : page_index + 12]
        canvas = np.zeros((1080, 1440, 3), dtype=np.uint8)
        for tile_index, tile in enumerate(page_tiles):
            row_index, column_index = divmod(tile_index, 3)
            canvas[
                row_index * 270 : (row_index + 1) * 270,
                column_index * 480 : (column_index + 1) * 480,
            ] = tile
        page_number = page_index // 12 + 1
        cv2.imwrite(str(output_dir / f"ua_others_stratified_page_{page_number:02d}.jpg"), canvas)


def build_queue(
    ua_path: Path,
    output: Path,
    contact_dir: Path,
    target_count: int,
    create_contact_sheets: bool,
    decisions_path: Path | None = None,
) -> list[dict[str, Any]]:
    decisions = {
        row["sample_id"]: row
        for row in read_csv(decisions_path)
    } if decisions_path and decisions_path.is_file() else {}
    with zipfile.ZipFile(ua_path) as archive:
        infos = archive.infolist()
        image_infos = [
            info
            for info in infos
            if info.filename.lower().endswith((".jpg", ".jpeg", ".png"))
            and "/detrac-images/" in info.filename.lower()
        ]
        image_by_frame: dict[tuple[str, int], zipfile.ZipInfo] = {}
        first_image_by_sequence: dict[str, zipfile.ZipInfo] = {}
        for info in image_infos:
            sequence = _sequence_from_path(info.filename)
            first_image_by_sequence.setdefault(sequence, info)
            try:
                frame = int(Path(info.filename).stem.lower().replace("img", ""))
            except ValueError:
                continue
            image_by_frame[(sequence, frame)] = info
        dimensions = {
            sequence: _image_dimensions(archive, info)
            for sequence, info in first_image_by_sequence.items()
        }

        candidates: list[dict[str, Any]] = []
        xml_infos = [
            info
            for info in infos
            if info.filename.lower().endswith(".xml")
            and "annotations-xml" in info.filename.lower()
        ]
        for info in sorted(xml_infos, key=lambda item: item.filename):
            root = ET.fromstring(archive.read(info))
            sequence = root.attrib.get("name") or Path(info.filename).stem
            sequence_attributes = root.find("sequence_attribute")
            weather = (
                sequence_attributes.attrib.get("sence_weather", "UNKNOWN")
                if sequence_attributes is not None
                else "UNKNOWN"
            ).upper()
            camera_state = (
                sequence_attributes.attrib.get("camera_state", "UNKNOWN")
                if sequence_attributes is not None
                else "UNKNOWN"
            ).upper()
            width, height = dimensions.get(sequence, (960, 540))
            for frame in root.findall("frame"):
                frame_id = int(frame.attrib.get("num", "0"))
                for target in frame.findall("./target_list/target"):
                    attribute = target.find("attribute")
                    if attribute is None or attribute.attrib.get("vehicle_type") != "others":
                        continue
                    box = target.find("box")
                    if box is None:
                        continue
                    left = float(box.attrib["left"])
                    top = float(box.attrib["top"])
                    box_width = float(box.attrib["width"])
                    box_height = float(box.attrib["height"])
                    right, bottom = left + box_width, top + box_height
                    clipped = (
                        min(max(left, 0.0), float(width)),
                        min(max(top, 0.0), float(height)),
                        min(max(right, 0.0), float(width)),
                        min(max(bottom, 0.0), float(height)),
                    )
                    area_ratio = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1]) / (width * height)
                    source = image_by_frame.get((sequence, frame_id))
                    if source is None:
                        continue
                    candidates.append(
                        {
                            "sequence_id": sequence,
                            "frame_id": frame_id,
                            "track_id": target.attrib.get("id", ""),
                            "source_file": source.filename,
                            "weather": weather,
                            "camera_state": camera_state,
                            "area_ratio": round(area_ratio, 8),
                            "size_bucket": _size_bucket(area_ratio),
                            "boundary_status": "BOUNDARY_CLIPPED" if right > width or bottom > height or left < 0 or top < 0 else "IN_FRAME",
                            "truncation_ratio": attribute.attrib.get("truncation_ratio", ""),
                            "clipped_xyxy": clipped,
                        }
                    )

        selected = _select_stratified(candidates, target_count)
        page_root = contact_dir.relative_to(ROOT).as_posix() if contact_dir.is_relative_to(ROOT) else str(contact_dir)
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(selected, 1):
            page_number = (index - 1) // 12 + 1
            sample_id = f"UA_OTHERS_STRAT_{index:03d}"
            decision = decisions.get(sample_id, {})
            assessment = decision.get("visual_assessment", "PENDING_REVIEW")
            review_status = decision.get("review_status", "PENDING_MANUAL_REVIEW")
            if assessment == "NON_VEHICLE":
                training_decision = "FALSE_REVIEW_REJECT"
            elif assessment == "UNDETERMINED":
                training_decision = "FALSE_PENDING_SECOND_REVIEW"
            elif assessment in {"CONFIRMED_MOTORIZED_VEHICLE", "LIKELY_MOTORIZED_VEHICLE"}:
                training_decision = (
                    "TRUE_DATA_LEAD_APPROVED"
                    if review_status == "DATA_LEAD_SIGNOFF_COMPLETED"
                    else "TRUE_WITH_CAUTION"
                )
            else:
                training_decision = "PENDING_REVIEW"
            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset_name": "UA-DETRAC Original",
                    **{key: value for key, value in candidate.items() if key != "clipped_xyxy"},
                    "sampling_method": "DETERMINISTIC_STRATIFIED_WEATHER_CAMERA_SIZE_BOUNDARY_SEQUENCE_CAP",
                    "visual_assessment": assessment,
                    "mapped_class": "vehicle",
                    "include_for_training": training_decision,
                    "preserve_original_class": True,
                    "reviewer": decision.get("reviewer", ""),
                    "review_date": decision.get("review_date", ""),
                    "review_status": review_status,
                    "evidence_path": f"{page_root}/ua_others_stratified_page_{page_number:02d}.jpg",
                    "notes": decision.get("notes", "Do not use this sample alone to approve all UA others annotations."),
                    "clipped_xyxy": candidate["clipped_xyxy"],
                }
            )
        if create_contact_sheets:
            _make_contact_sheets(archive, rows, contact_dir)

    write_csv(output, rows, FIELDS)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ua-path", default=str(DEFAULT_UA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--contact-sheet-dir", default=str(DEFAULT_CONTACTS))
    parser.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    parser.add_argument("--target-count", type=int, default=60)
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()
    rows = build_queue(
        Path(args.ua_path),
        Path(args.output),
        Path(args.contact_sheet_dir),
        max(1, args.target_count),
        not args.no_contact_sheets,
        Path(args.decisions),
    )
    print(f"UA others stratified review rows: {len(rows)}")
    print(f"Review queue: {Path(args.output).resolve()}")
    print(f"Contact sheets: {Path(args.contact_sheet_dir).resolve() if not args.no_contact_sheets else 'SKIPPED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
