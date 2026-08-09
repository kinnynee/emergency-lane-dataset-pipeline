"""Extract and stage the newly supplied Pexels/Mixkit videos for vehicle annotation.

The assignments are based on a three-point visual review.  They are recorded
as review candidates, never added to a train/validation/test split directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


RAW = Path(r"D:/UMT_EVIDENCE/online_data/raw/pexels")
FRAME_ROOT = Path(r"D:/UMT_EVIDENCE/online_data/frames/user_provided_pexels_20260809_v1")
BATCH_ROOT = Path(r"D:/UMT_EVIDENCE/dataset-v1-full/review_pending/annotation_batches/user_provided_pexels_20260809_v1")


SPECS = [
    ("12641071-uhd_2160_3840_30fps.mp4", "UPX_12641071", "URBAN_ROAD", "TWILIGHT", "train", "Vertical urban wet-road sunset traffic."),
    ("12654544_3840_2160_30fps.mp4", "UPX_12654544", "HIGHWAY", "NIGHT", "val", "Aerial motorway light-trail view at night."),
    ("13095829_3840_2160_24fps.mp4", "UPX_13095829", "URBAN_ROAD", "TWILIGHT", "train", "Urban arterial traffic at sunset."),
    ("13538225_2160_3840_30fps.mp4", "UPX_13538225", "HIGHWAY", "TWILIGHT", "val", "Elevated motorway traffic at sunset."),
    ("140224-774508039_medium.mp4", "UPX_140224", "INTERSECTION", "DAY", "cross_test", "Aerial intersection traffic in daylight."),
    ("14176127_2160_3840_30fps.mp4", "UPX_14176127", "HIGHWAY", "DAY", "train", "Divided multi-lane traffic corridor in daylight."),
    ("14388179-hd_1080_1920_30fps.mp4", "UPX_14388179", "HIGHWAY", "TWILIGHT", "cross_test", "Dashcam highway at sunset."),
    ("15418503_3840_2160_24fps.mp4", "UPX_15418503", "URBAN_ROAD", "TWILIGHT", "val", "Urban boulevard at dusk."),
    ("15477129_1080_1920_30fps.mp4", "UPX_15477129", "URBAN_ROAD", "DAY", "val", "Vertical urban avenue traffic in daylight."),
    ("15571746_1080_1920_30fps.mp4", "UPX_15571746", "URBAN_ROAD", "TWILIGHT", "val", "Urban road sunset traffic."),
    ("mixkit-cars-on-an-avenue-of-a-big-city-during-sunset-40635-hd-ready.mp4", "UPX_MIXKIT_40635", "URBAN_ROAD", "TWILIGHT", "cross_test", "Urban avenue traffic at sunset."),
    ("mixkit-highway-and-roads-from-an-overhead-aerial-perspective-49854-hd-ready.mp4", "UPX_MIXKIT_49854", "HIGHWAY", "DAY", "train", "Overhead highway/road aerial view in daylight."),
    ("mixkit-highway-of-a-city-during-a-sunset-3418-hd-ready.mp4", "UPX_MIXKIT_3418", "HIGHWAY", "TWILIGHT", "cross_test", "Aerial city highway at sunset."),
]


def extract(spec: tuple[str, str, str, str, str, str], execute: bool) -> tuple[dict[str, str], list[dict[str, str]]]:
    file_name, video_id, road_type, lighting, intended_split, review_note = spec
    source = RAW / file_name
    if not source.is_file():
        raise FileNotFoundError(f"Missing supplied video: {source}")
    output_dir = FRAME_ROOT / video_id
    batch_images = BATCH_ROOT / "images"
    if output_dir.exists():
        raise FileExistsError(f"Extraction target exists: {output_dir}")
    if not execute:
        return ({"video_id": video_id, "file_name": file_name, "target_road_type": road_type, "lighting": lighting, "intended_split": intended_split, "review_note": review_note, "extracted_frames": "DRY_RUN"}, [])
    output_dir.mkdir(parents=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(source),
        "-vf", "fps=1,scale=1280:1280:force_original_aspect_ratio=decrease", "-q:v", "3",
        str(output_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(command, check=True)
    frames = sorted(output_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames extracted from {source}")
    source_group = "MIXKIT_USER_PROVIDED_UNVERIFIED" if file_name.startswith("mixkit-") else "PEXELS_USER_PROVIDED_UNVERIFIED"
    video_row = {
        "video_id": video_id,
        "file_name": file_name,
        "source_video": str(source),
        "source_group": source_group,
        "license_status": "USER_PROVIDED_PENDING_SOURCE_PROVENANCE",
        "target_road_type": road_type,
        "lighting": lighting,
        "intended_split": intended_split,
        "extraction_fps": "1",
        "max_dimension": "1280",
        "extracted_frames": str(len(frames)),
        "review_note": review_note,
        "export_status": "EXTRACTED_REVIEW_REQUIRED",
    }
    task_rows: list[dict[str, str]] = []
    for frame_number, frame in enumerate(frames, start=1):
        target_name = f"{video_id}_{frame.name}"
        destination = batch_images / target_name
        if destination.exists():
            raise FileExistsError(f"Batch image collision: {destination}")
        shutil.copy2(frame, destination)
        task_rows.append({
            "task_id": f"{video_id}_{frame_number:06d}",
            "image_file": f"images/{target_name}",
            "expected_yolo_label": f"annotations_pending/{Path(target_name).stem}.txt",
            "annotation_status": "UNASSIGNED",
            "intended_split": intended_split,
            "class_id": "0",
            "class_name": "vehicle",
            "road_type": road_type,
            "lighting": lighting,
            "source_video_id": video_id,
            "source_video": str(source),
            "license_status": "USER_PROVIDED_PENDING_SOURCE_PROVENANCE",
            "review_requirement": "Auto-label all visible motor vehicles; human review before promotion.",
        })
    return video_row, task_rows


def main(execute: bool) -> int:
    if FRAME_ROOT.exists() or BATCH_ROOT.exists():
        raise FileExistsError("Versioned extraction or batch directory already exists; refusing to overwrite")
    all_video_rows: list[dict[str, str]] = []
    all_tasks: list[dict[str, str]] = []
    if execute:
        (BATCH_ROOT / "images").mkdir(parents=True)
        (BATCH_ROOT / "annotations_pending").mkdir()
        (BATCH_ROOT / "metadata").mkdir()
    for spec in SPECS:
        video_row, task_rows = extract(spec, execute)
        all_video_rows.append(video_row)
        all_tasks.extend(task_rows)
    summary = {
        "status": "COMPLETE" if execute else "DRY_RUN",
        "unique_videos": len(SPECS),
        "excluded_duplicate": "12641071-uhd_2160_3840_30fps (1).mp4",
        "frames": len(all_tasks) if execute else "PENDING_EXTRACTION",
        "intended_sequences_by_split": {
            split: sum(row["intended_split"] == split for row in all_video_rows)
            for split in ("train", "val", "cross_test")
        },
    }
    if execute:
        with (BATCH_ROOT / "metadata/video_extraction_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_video_rows[0]))
            writer.writeheader()
            writer.writerows(all_video_rows)
        with (BATCH_ROOT / "metadata/annotation_tasks.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_tasks[0]))
            writer.writeheader()
            writer.writerows(all_tasks)
        (BATCH_ROOT / "metadata/extraction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.execute))
