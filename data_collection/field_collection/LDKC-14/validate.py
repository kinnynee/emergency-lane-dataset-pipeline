"""Check completeness and consistency of the LDKC-14 test-only handoff."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


BASE = Path(__file__).resolve().parent


def load_csv(name: str, required: set[str]) -> list[dict[str, str]]:
    with (BASE / name).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        return list(reader)


def number(value: str, label: str, errors: list[str], integer: bool = False) -> float | int | None:
    try:
        result = int(value) if integer else float(value)
    except (TypeError, ValueError):
        errors.append(f"{label}: invalid number {value!r}")
        return None
    if not integer and not math.isfinite(result):
        errors.append(f"{label}: non-finite number {value!r}")
        return None
    return result


def main() -> int:
    manifest = load_csv("video_manifest.csv", {"video", "fps", "so_khung", "chieu_rong", "chieu_cao", "thoi_luong_giay", "loai_video", "xac_nhan_khong_dung", "roi_json", "tap_du_lieu"})
    events = load_csv("su_kien.csv", {"video", "id_xe", "khung_bat_dau", "khung_ket_thuc", "trong_lan", "dieu_kien"})
    boxes = load_csv("khung_bao.csv", {"video", "id_xe", "khung", "vi_tri", "x_min", "y_min", "x_max", "y_max"})
    errors: list[str] = []
    videos: dict[str, dict[str, float | int | str]] = {}
    normal_seconds = 0.0

    for row in manifest:
        name = row["video"].strip()
        if not name or name in videos:
            errors.append(f"video_manifest.csv: empty or duplicate video {name!r}")
            continue
        values = {key: number(row[key], f"{name}.{key}", errors, key in {"so_khung", "chieu_rong", "chieu_cao"}) for key in ("fps", "so_khung", "chieu_rong", "chieu_cao", "thoi_luong_giay")}
        if any(value is None or value <= 0 for value in values.values()):
            errors.append(f"{name}: FPS, frame count, dimensions and duration must be positive")
            continue
        if row["tap_du_lieu"] != "TEST_ONLY":
            errors.append(f"{name}: tap_du_lieu must be TEST_ONLY")
        if not (BASE / name).is_file():
            errors.append(f"{name}: video file not found in LDKC-14 package")
        if row["loai_video"] not in {"su_kien", "binh_thuong_khong_dung"}:
            errors.append(f"{name}: invalid loai_video")
        roi_path = BASE / row["roi_json"]
        try:
            if not isinstance(json.loads(roi_path.read_text(encoding="utf-8-sig")), dict):
                raise ValueError("ROI JSON root must be an object")
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: cannot read valid roi_json {roi_path}: {exc}")
        if row["loai_video"] == "binh_thuong_khong_dung" and row["xac_nhan_khong_dung"] == "co":
            normal_seconds += float(values["thoi_luong_giay"])
        videos[name] = {**values, "loai_video": row["loai_video"]}

    event_keys: set[tuple[str, str, int, int]] = set()
    positive_events: list[tuple[str, str, int, int, set[str]]] = []
    day = night = long_stops = 0
    for row in events:
        name, vehicle = row["video"].strip(), row["id_xe"].strip()
        start = number(row["khung_bat_dau"], f"{name}.khung_bat_dau", errors, True)
        end = number(row["khung_ket_thuc"], f"{name}.khung_ket_thuc", errors, True)
        if not vehicle or name not in videos or start is None or end is None:
            errors.append(f"su_kien.csv: unknown video, empty id_xe or invalid frame for {name!r}")
            continue
        key = (name, vehicle, start, end)
        if key in event_keys:
            errors.append(f"su_kien.csv: duplicate event {key}")
        event_keys.add(key)
        if start < 0 or start > end or end >= videos[name]["so_khung"]:
            errors.append(f"{key}: frame range outside video")
            continue
        if row["trong_lan"] not in {"co", "khong"}:
            errors.append(f"{key}: trong_lan must be co or khong")
        conditions = set(row["dieu_kien"].split("|"))
        if not conditions or not conditions <= {"ngay", "dem", "mua", "nguoc_sang"} or ("ngay" in conditions and "dem" in conditions):
            errors.append(f"{key}: invalid dieu_kien")
        if videos[name]["loai_video"] == "binh_thuong_khong_dung" and row["trong_lan"] == "co":
            errors.append(f"{name}: normal video has a stopped vehicle in lane")
        if row["trong_lan"] == "co":
            duration = (end - start + 1) / float(videos[name]["fps"])
            if duration < 10:
                errors.append(f"{key}: stopped duration {duration:.2f}s is under 10s")
            else:
                positive_events.append((name, vehicle, start, end, conditions))
                day += "ngay" in conditions
                night += "dem" in conditions
                long_stops += 30 <= duration <= 120

    box_index: set[tuple[str, str, int, str]] = set()
    for row in boxes:
        name, vehicle, position = row["video"].strip(), row["id_xe"].strip(), row["vi_tri"].strip()
        frame = number(row["khung"], f"{name}.khung", errors, True)
        coords = [number(row[key], f"{name}.{key}", errors) for key in ("x_min", "y_min", "x_max", "y_max")]
        if name not in videos or not vehicle or frame is None or position not in {"dau", "giua", "cuoi"} or any(v is None for v in coords):
            errors.append(f"khung_bao.csv: invalid row for {name!r}")
            continue
        x1, y1, x2, y2 = coords
        if not (0 <= x1 < x2 <= videos[name]["chieu_rong"] and 0 <= y1 < y2 <= videos[name]["chieu_cao"]):
            errors.append(f"{name}/{vehicle}/{frame}: box outside image or non-positive")
        if not 0 <= frame < videos[name]["so_khung"]:
            errors.append(f"{name}/{vehicle}/{frame}: box frame outside video")
        index_key = (name, vehicle, frame, position)
        if index_key in box_index:
            errors.append(f"khung_bao.csv: duplicate {index_key}")
        box_index.add(index_key)

    for name, vehicle, start, end, _ in positive_events:
        expected = (("dau", start), ("giua", (start + end) // 2), ("cuoi", end))
        for position, frame in expected:
            if (name, vehicle, frame, position) not in box_index:
                errors.append(f"{name}/{vehicle}/{start}-{end}: missing {position} box at frame {frame}")

    if len(positive_events) < 60:
        errors.append(f"Need 60 valid in-lane stops; found {len(positive_events)}")
    if day < 20 or night < 20:
        errors.append(f"Need 20 day and 20 night stops; found day={day}, night={night}")
    if long_stops == 0:
        errors.append("Need at least one 30-120s stop")
    if normal_seconds < 5 * 3600:
        errors.append(f"Need 5h reviewed normal traffic; found {normal_seconds / 3600:.2f}h")

    print(f"Videos: {len(videos)}; valid in-lane stops: {len(positive_events)}; day: {day}; night: {night}; 30-120s: {long_stops}; reviewed normal traffic: {normal_seconds / 3600:.2f}h")
    for error in errors:
        print("ERROR:", error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
