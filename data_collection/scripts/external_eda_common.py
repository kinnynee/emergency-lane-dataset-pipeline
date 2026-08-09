"""Tiện ích chung cho EDA dữ liệu ngoài, ưu tiên streaming và truy vết."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "external_eda"
FIGURES = DEFAULT_OUTPUT / "figures"
CONFIG_PATH = ROOT / "configs" / "external_eda_config.yaml"
MAPPING_PATH = ROOT / "configs" / "vehicle_class_mapping.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT.parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "NOT_AVAILABLE"
    except (OSError, subprocess.SubprocessError):
        return "NOT_AVAILABLE"


def reservoir_sample(items: Iterable[Any], size: int, seed: int = 230) -> list[Any]:
    """Lấy mẫu reservoir không cần nạp toàn bộ iterable vào RAM."""
    rng = random.Random(seed)
    sample: list[Any] = []
    for index, item in enumerate(items):
        if index < size:
            sample.append(item)
        else:
            replacement = rng.randint(0, index)
            if replacement < size:
                sample[replacement] = item
    return sample


def safe_sequence(path: str) -> str:
    parts = PurePosixPath(path.replace("\\", "/")).parts
    for part in parts:
        if part.upper().startswith("MVI_"):
            return part
    if len(parts) >= 2:
        return parts[-2]
    return Path(path).stem


def image_from_bytes(data: bytes) -> np.ndarray | None:
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def image_quality(image: np.ndarray, file_size: int, source_file: str) -> dict[str, Any]:
    if image is None or image.size == 0:
        return {
            "source_file": source_file,
            "read_status": "CORRUPT_OR_UNREADABLE",
            "assessment_source": "AUTOMATIC_ESTIMATE",
        }
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean = float(gray.mean())
    std = float(gray.std())
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_ratio = float(np.mean(gray < 25))
    bright_ratio = float(np.mean(gray > 240))
    saturation = float(hsv[:, :, 1].mean())
    return {
        "source_file": source_file,
        "read_status": "OK",
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 6) if height else "",
        "file_size_bytes": file_size,
        "mean_brightness": round(mean, 4),
        "brightness_std": round(std, 4),
        "contrast": round(std, 4),
        "laplacian_variance": round(lap, 4),
        "blur_score": round(lap, 4),
        "dark_pixel_ratio": round(dark_ratio, 6),
        "bright_pixel_ratio": round(bright_ratio, 6),
        "mean_saturation": round(saturation, 4),
        "black_suspect": mean < 5,
        "white_suspect": mean > 250,
        "underexposed_suspect": mean < 45,
        "overexposed_suspect": mean > 220,
        "blur_suspect": lap < 60,
        "assessment_source": "AUTOMATIC_ESTIMATE",
    }


def letterbox_box_metrics(
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    target: int = 320,
) -> dict[str, Any]:
    if image_width <= 0 or image_height <= 0:
        return {"box_320_category": "NOT_COMPUTED"}
    scale = min(target / image_width, target / image_height)
    width_320 = width * scale
    height_320 = height * scale
    area_320 = width_320 * height_320
    minimum_side = min(width_320, height_320)
    if minimum_side < 4:
        category = "EXTREMELY_TINY"
    elif minimum_side < 8:
        category = "VERY_SMALL"
    elif minimum_side < 16:
        category = "SMALL"
    else:
        category = "USABLE"
    return {
        "box_width_320": round(width_320, 6),
        "box_height_320": round(height_320, 6),
        "box_area_320": round(area_320, 6),
        "box_320_category": category,
    }


def relative_size_category(area_ratio: float) -> str:
    if area_ratio < 0.001:
        return "SMALL"
    if area_ratio < 0.02:
        return "MEDIUM"
    return "LARGE"


def distance_proxy(area_ratio: float) -> str:
    if area_ratio < 0.001:
        return "FAR"
    if area_ratio < 0.02:
        return "MEDIUM"
    return "NEAR"


def validate_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int | None,
    image_height: int | None,
) -> list[str]:
    issues: list[str] = []
    values = (xmin, ymin, xmax, ymax)
    if not all(math.isfinite(value) for value in values):
        return ["NAN_OR_INFINITY"]
    if xmin < 0 or ymin < 0:
        issues.append("NEGATIVE_COORDINATE")
    if xmin >= xmax or ymin >= ymax:
        issues.append("NON_POSITIVE_SIZE")
    if image_width and xmax > image_width:
        issues.append("X_OUT_OF_BOUNDS")
    if image_height and ymax > image_height:
        issues.append("Y_OUT_OF_BOUNDS")
    return issues


def clip_bbox_to_image(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
) -> tuple[tuple[float, float, float, float], list[str]]:
    """Clip a boundary-crossing box while recording which sides changed.

    Boundary-crossing coordinates may reflect truncation or a source coordinate
    convention. Clip them rather than assuming a semantic cause or dropping the
    object. Call ``validate_bbox`` after clipping to reject only boxes that remain
    malformed or have no visible area.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive before bbox clipping")

    clipped_xmin = min(max(xmin, 0.0), float(image_width))
    clipped_ymin = min(max(ymin, 0.0), float(image_height))
    clipped_xmax = min(max(xmax, 0.0), float(image_width))
    clipped_ymax = min(max(ymax, 0.0), float(image_height))
    adjustments: list[str] = []
    if clipped_xmin != xmin:
        adjustments.append("LEFT_BOUNDARY_CLIPPED")
    if clipped_ymin != ymin:
        adjustments.append("TOP_BOUNDARY_CLIPPED")
    if clipped_xmax != xmax:
        adjustments.append("RIGHT_BOUNDARY_CLIPPED")
    if clipped_ymax != ymax:
        adjustments.append("BOTTOM_BOUNDARY_CLIPPED")
    return (
        (clipped_xmin, clipped_ymin, clipped_xmax, clipped_ymax),
        adjustments,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def perceptual_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    block = dct[:8, :8]
    median = float(np.median(block[1:, :]))
    bits = (block > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_similarity(left: int, right: int, bits: int = 64) -> float:
    distance = (left ^ right).bit_count()
    return round(1.0 - distance / bits, 6)


def aggregate_numeric(
    rows: list[dict[str, Any]], key: str
) -> dict[str, float | int | str]:
    values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    if not values:
        return {"count": 0, "min": "", "mean": "", "median": "", "max": ""}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": round(float(array.min()), 6),
        "mean": round(float(array.mean()), 6),
        "median": round(float(np.median(array)), 6),
        "max": round(float(array.max()), 6),
    }


def stable_split(sequence_id: str) -> str:
    value = int(hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < 0.80:
        return "EXTERNAL_TRAIN"
    if value < 0.90:
        return "EXTERNAL_VALIDATION"
    return "CROSS_DATASET_TEST"


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def checkpoint_done(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return key in payload.get("completed", [])


def add_checkpoint(path: Path, key: str) -> None:
    payload = {"completed": []}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    completed = set(payload.get("completed", []))
    completed.add(key)
    payload["completed"] = sorted(completed)
    json_dump(path, payload)


def count_values(rows: Iterable[dict[str, Any]], key: str) -> Counter[str]:
    return Counter(str(row.get(key, "UNKNOWN") or "UNKNOWN") for row in rows)
