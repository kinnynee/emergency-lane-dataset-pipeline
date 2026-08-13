"""Generate the five evidence-backed figures for the project report.

The source datasets do not contain stopped-vehicle-in-emergency-lane ground
truth, so this tool reports detector quality only. All performance figures in
the report use *only* the locked ``cross_test`` split and are separated by
source domain; aggregated AAU+UA headline metrics are prohibited.
Train/validation summaries may be retained as internal diagnostics, but are
never presented as independent model quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont
import yaml

from ignored_region import bbox_center_is_ignored, region_map_from_ua_annotations


AAU_DATASET = "AAU RainSnow"
UA_DATASET = "UA-DETRAC Original"
MIO_DATASET = "MIO-TCD Localization"
DEVICE_WIDTH = 640.0
DEVICE_HEIGHT = 480.0
IOU_THRESHOLDS = tuple(round(0.50 + step * 0.05, 2) for step in range(10))
PRESENTATION_CONFIDENCE = 0.50
MIN_CONDITION_IMAGES = 30
MIN_CONDITION_BOXES = 100


@dataclass(frozen=True)
class Box:
    image_id: str
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0


@dataclass(frozen=True)
class EvaluationSample:
    image_id: str
    image_path: Path
    width: int
    height: int
    lighting: str
    weather: str
    labels: tuple[Box, ...]
    predictions: tuple[Box, ...]
    dataset: str = ""
    split: str = ""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_yolo_labels(path: Path, image_id: str, width: int, height: int) -> tuple[Box, ...]:
    boxes: list[Box] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, center_x, center_y, box_width, box_height = (float(value) for value in line.split())
        if class_id != 0:
            raise ValueError(f"Expected one vehicle class in {path}, got class {class_id}")
        center_x *= width
        center_y *= height
        box_width *= width
        box_height *= height
        boxes.append(Box(image_id, (
            center_x - box_width / 2,
            center_y - box_height / 2,
            center_x + box_width / 2,
            center_y + box_height / 2,
        )))
    return tuple(boxes)


def box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _match_predictions(samples: Sequence[EvaluationSample], iou_threshold: float, confidence: float = 0.0) -> tuple[list[tuple[Box, bool]], dict[tuple[str, int], bool]]:
    labels_by_image = {sample.image_id: sample.labels for sample in samples}
    matched: dict[tuple[str, int], bool] = {}
    candidates = sorted(
        (box for sample in samples for box in sample.predictions if box.confidence >= confidence),
        key=lambda box: box.confidence,
        reverse=True,
    )
    results: list[tuple[Box, bool]] = []
    for prediction in candidates:
        best_index = -1
        best_iou = iou_threshold
        for index, label in enumerate(labels_by_image[prediction.image_id]):
            if matched.get((prediction.image_id, index)):
                continue
            overlap = box_iou(prediction.xyxy, label.xyxy)
            if overlap >= best_iou:
                best_iou = overlap
                best_index = index
        is_true_positive = best_index >= 0
        if is_true_positive:
            matched[(prediction.image_id, best_index)] = True
        results.append((prediction, is_true_positive))
    return results, matched


def average_precision(samples: Sequence[EvaluationSample], iou_threshold: float) -> float | None:
    ground_truth_count = sum(len(sample.labels) for sample in samples)
    if ground_truth_count == 0:
        return None
    matches, _ = _match_predictions(samples, iou_threshold)
    if not matches:
        return 0.0
    true_positives = np.cumsum([int(is_true_positive) for _, is_true_positive in matches])
    false_positives = np.cumsum([int(not is_true_positive) for _, is_true_positive in matches])
    recall = true_positives / ground_truth_count
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall_envelope = np.concatenate(([0.0], recall, [1.0]))
    precision_envelope = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision_envelope) - 2, -1, -1):
        precision_envelope[index] = max(precision_envelope[index], precision_envelope[index + 1])
    changes = np.where(recall_envelope[1:] != recall_envelope[:-1])[0]
    return float(np.sum((recall_envelope[changes + 1] - recall_envelope[changes]) * precision_envelope[changes + 1]))


def map50_95(samples: Sequence[EvaluationSample]) -> float | None:
    values = [average_precision(samples, threshold) for threshold in IOU_THRESHOLDS]
    available = [value for value in values if value is not None]
    return float(np.mean(available)) if available else None


def _letterbox_box_to_k230(box: Box, width: int, height: int) -> tuple[float, float, float, float]:
    scale = min(DEVICE_WIDTH / width, DEVICE_HEIGHT / height)
    pad_x = (DEVICE_WIDTH - width * scale) / 2.0
    pad_y = (DEVICE_HEIGHT - height * scale) / 2.0
    x1, y1, x2, y2 = box.xyxy
    return x1 * scale + pad_x, y1 * scale + pad_y, x2 * scale + pad_x, y2 * scale + pad_y


def size_group(box: Box, width: int, height: int) -> str:
    x1, y1, x2, y2 = _letterbox_box_to_k230(box, width, height)
    minimum_side = min(x2 - x1, y2 - y1)
    if minimum_side < 25.0:
        return "<25 px"
    if minimum_side < 50.0:
        return "25–49 px"
    return "≥50 px"


def _scene_lookup(scene_metadata: Path) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["dataset_name"], row["sequence_id"]): row
        for row in _read_csv(scene_metadata)
    }


def _ignored_regions_from_run(run_dir: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    config = yaml.safe_load((run_dir / "smoke_config.yaml").read_text(encoding="utf-8")) or {}
    roots = config.get("ua_annotation_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("smoke_config.yaml must list ua_annotation_roots for a total held-out metric")
    result: dict[str, list[tuple[float, float, float, float]]] = {}
    for raw_root in roots:
        for sequence_id, regions in region_map_from_ua_annotations(Path(str(raw_root))).items():
            previous = result.get(sequence_id)
            if previous is not None and previous != regions:
                raise ValueError(f"Conflicting ignored regions for {sequence_id}")
            result[sequence_id] = regions
    return result


def _class_ignore_regions_from_export(run_dir: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """Load bicycle/motorbike ignore boxes emitted by the unified exporter."""
    path = run_dir / "dataset" / "metadata" / "ignored_annotations.csv"
    if not path.is_file():
        return {}
    regions: defaultdict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for row in _read_csv(path):
        try:
            box = tuple(float(row[field]) for field in ("xmin", "ymin", "xmax", "ymax"))
        except (KeyError, ValueError):
            raise ValueError(f"Invalid ignore-region row in {path}") from None
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f"Non-positive ignore region in {path}")
        regions[str(row["image_id"])].append(box)
    return dict(regions)


def _load_full_dataset(run_dir: Path, scene_metadata: Path, weights: Path, device: str) -> list[EvaluationSample]:
    image_rows = _read_csv(run_dir / "dataset" / "metadata" / "images.csv")
    scenes = _scene_lookup(scene_metadata)
    ignored_by_sequence = _ignored_regions_from_run(run_dir)
    ignored_by_image = _class_ignore_regions_from_export(run_dir)
    selected = [
        row for row in image_rows
        if row.get("dataset") in {AAU_DATASET, UA_DATASET, MIO_DATASET}
    ]
    if not selected:
        raise ValueError("No supported dataset images found in the run directory")
    selected.sort(key=lambda row: (row["dataset"], row["image_id"]))
    paths = [run_dir / "dataset" / row["exported_image"] for row in selected]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset image missing: {missing[0]}")
    if not weights.is_file():
        raise FileNotFoundError(f"Weights missing: {weights}")
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ultralytics is required to generate report figures") from exc
    predictions = YOLO(str(weights)).predict(
        source=[str(path) for path in paths],
        imgsz=320,
        conf=0.001,
        iou=0.70,
        max_det=300,
        device=device,
        verbose=False,
    )
    samples: list[EvaluationSample] = []
    for row, path, result in zip(selected, paths, predictions):
        image_id = row["image_id"]
        width, height = int(row["width"]), int(row["height"])
        scene = scenes.get((row["dataset"], row["sequence_id"]))
        if scene is None:
            raise ValueError(f"Missing reviewed scene metadata for {row['dataset']} / {row['sequence_id']}")
        raw_boxes = result.boxes
        prediction_boxes: list[Box] = []
        if raw_boxes is not None:
            coordinates = raw_boxes.xyxy.cpu().tolist()
            confidences = raw_boxes.conf.cpu().tolist()
            classes = raw_boxes.cls.int().cpu().tolist()
            for xyxy, confidence, class_id in zip(coordinates, confidences, classes):
                if class_id == 0:
                    box = tuple(float(value) for value in xyxy)
                    static_ignored = ignored_by_sequence.get(row["sequence_id"], []) if row["dataset"] == UA_DATASET else []
                    if bbox_center_is_ignored(box, [*static_ignored, *ignored_by_image.get(image_id, [])]):
                        continue
                    prediction_boxes.append(Box(image_id, box, float(confidence)))
        label_path = run_dir / "dataset" / row["exported_label"]
        samples.append(EvaluationSample(
            image_id=image_id,
            image_path=path,
            width=width,
            height=height,
            lighting=scene["lighting"],
            weather=scene["weather"],
            labels=_load_yolo_labels(label_path, image_id, width, height),
            predictions=tuple(prediction_boxes),
            dataset=row["dataset"],
            split=row["split"],
        ))
    return samples


def _condition_slices(samples: Sequence[EvaluationSample]) -> dict[str, list[EvaluationSample]]:
    return {
        "DAY": [sample for sample in samples if sample.lighting == "DAY"],
        "NIGHT": [sample for sample in samples if sample.lighting == "NIGHT"],
        "RAIN": [sample for sample in samples if "RAIN" in sample.weather],
    }


def _condition_metrics(samples: Sequence[EvaluationSample], metric_scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, subset in _condition_slices(samples).items():
        image_count = len(subset)
        ground_truth_boxes = sum(len(sample.labels) for sample in subset)
        is_reportable = image_count >= MIN_CONDITION_IMAGES and ground_truth_boxes >= MIN_CONDITION_BOXES
        value = map50_95(subset) if is_reportable else None
        rows.append({
            "condition": condition,
            "image_count": image_count,
            "ground_truth_boxes": ground_truth_boxes,
            "mAP50_95": "" if value is None else f"{value:.6f}",
            "sample_status": "REPORTABLE" if is_reportable else "N/A_INSUFFICIENT_SAMPLE",
            "metric_scope": metric_scope,
        })
    return rows


def _condition_overlap(samples: Sequence[EvaluationSample]) -> dict[str, Any]:
    rain = [sample for sample in samples if "RAIN" in sample.weather]
    night = [sample for sample in samples if sample.lighting == "NIGHT"]
    overlap = [sample for sample in rain if sample.lighting == "NIGHT"]
    rain_images = len(rain)
    night_images = len(night)
    overlap_images = len(overlap)
    return {
        "condition_a": "RAIN",
        "condition_b": "NIGHT",
        "rain_image_count": rain_images,
        "night_image_count": night_images,
        "overlap_image_count": overlap_images,
        "overlap_ground_truth_boxes": sum(len(sample.labels) for sample in overlap),
        "rain_images_also_night_ratio": overlap_images / rain_images if rain_images else None,
        "night_images_also_rain_ratio": overlap_images / night_images if night_images else None,
        "scope": "AAU RainSnow + UA-DETRAC Original cross_test only; MIO-TCD is train-only and excluded",
    }


def _recall_by_size(samples: Sequence[EvaluationSample]) -> list[dict[str, Any]]:
    _, matched = _match_predictions(samples, iou_threshold=0.50, confidence=PRESENTATION_CONFIDENCE)
    totals: defaultdict[str, int] = defaultdict(int)
    detected: defaultdict[str, int] = defaultdict(int)
    for sample in samples:
        for index, label in enumerate(sample.labels):
            group = size_group(label, sample.width, sample.height)
            totals[group] += 1
            detected[group] += int(matched.get((sample.image_id, index), False))
    return [
        {
            "size_group_k230_640x480": group,
            "ground_truth_boxes": totals[group],
            "matched_at_iou_0_50": detected[group],
            "recall_at_confidence_0_50": detected[group] / totals[group] if totals[group] else 0.0,
            "metric_scope": "AAU RainSnow + UA-DETRAC Original cross_test only; MIO-TCD is train-only and excluded; UA ignored-region predictions excluded",
        }
        for group in ("<25 px", "25–49 px", "≥50 px")
    ]


def _recall_by_dataset_size(samples: Sequence[EvaluationSample]) -> list[dict[str, Any]]:
    """Compute recall separately so a large UA slice cannot hide AAU weakness."""
    rows: list[dict[str, Any]] = []
    for dataset in (AAU_DATASET, UA_DATASET):
        subset = [sample for sample in samples if sample.dataset == dataset]
        for row in _recall_by_size(subset):
            rows.append({"dataset": dataset, **row})
    return rows


def _metric_by_dataset(samples: Sequence[EvaluationSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in (AAU_DATASET, UA_DATASET):
        subset = [sample for sample in samples if sample.dataset == dataset]
        if not subset:
            rows.append({
                "dataset": dataset, "image_count": 0, "ground_truth_boxes": 0,
                "mAP50_95": "", "status": "NOT_MEASURED_MISSING_DOMAIN_SAMPLES",
                "confidence": PRESENTATION_CONFIDENCE,
            })
            continue
        value = map50_95(subset)
        rows.append({
            "dataset": dataset,
            "image_count": len(subset),
            "ground_truth_boxes": sum(len(sample.labels) for sample in subset),
            "mAP50_95": "" if value is None else f"{value:.6f}",
            "status": "MEASURED_CROSS_TEST",
            "confidence": PRESENTATION_CONFIDENCE,
        })
    return rows


def _plot_architecture(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 6.8))
    axis.set_axis_off()
    components = [
        (0.04, 0.60, 0.18, 0.20, "Edge 1\nCamera K230\nYOLO + ROI + tracker"),
        (0.34, 0.60, 0.20, 0.20, "Cloud\nCoreIoT broker + rule chain\nAudit log / dashboard"),
        (0.66, 0.60, 0.15, 0.20, "Edge 2\nESP32\nSubscribe + control"),
        (0.86, 0.60, 0.11, 0.20, "LED sign\nCảnh báo"),
        (0.34, 0.22, 0.20, 0.16, "Cloud storage\nTelemetry + events"),
    ]
    for x, y, width, height, text in components:
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.015", linewidth=1.4, edgecolor="#245f8f", facecolor="#eef6fb"))
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=11, weight="medium")
    arrow = dict(arrowstyle="->", linewidth=1.8, color="#245f8f")
    axis.annotate("", xy=(0.34, 0.70), xytext=(0.22, 0.70), arrowprops=arrow)
    axis.text(0.28, 0.84, "MQTT: vehicle_detected / vehicle_stopped", ha="center", va="bottom", fontsize=9)
    axis.annotate("", xy=(0.22, 0.63), xytext=(0.34, 0.63), arrowprops={**arrow, "linestyle": "--"})
    axis.text(0.28, 0.54, "config / watchdog", ha="center", va="top", fontsize=9)
    axis.annotate("", xy=(0.66, 0.70), xytext=(0.54, 0.70), arrowprops=arrow)
    axis.text(0.60, 0.84, "RPC / command", ha="center", va="bottom", fontsize=9)
    axis.text(0.60, 0.51, "Khoảng cách triển khai Edge 1 ↔ Edge 2: 200–500 m", ha="center", va="center", fontsize=9.5, weight="bold", color="#245f8f")
    axis.annotate("", xy=(0.86, 0.70), xytext=(0.81, 0.70), arrowprops=arrow)
    axis.text(0.835, 0.84, "GPIO / serial", ha="center", va="bottom", fontsize=9)
    axis.annotate("", xy=(0.44, 0.38), xytext=(0.44, 0.60), arrowprops=arrow)
    axis.text(0.46, 0.49, "telemetry + audit", ha="left", va="center", fontsize=9)
    axis.text(0.5, 0.06, "Luồng bắt buộc Edge → Cloud → Edge: K230 không giao tiếp trực tiếp với ESP32.", ha="center", va="center", fontsize=12, weight="bold", color="#8a2d2d")
    axis.set_title("A1. Kiến trúc cảnh báo xe dừng làn khẩn cấp", fontsize=16, weight="bold", pad=16)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _format_cell(value: float, column: str) -> str:
    if column == "box_count" or column == "roi_proxy_box_count":
        return f"{int(value):,}"
    return f"{value * 100:.2f}%"


def _plot_dataset_overview(analysis: dict[str, Any], output: Path) -> None:
    rows = analysis["bbox_size_by_dataset"]
    columns = [
        ("box_count", "Tổng\nhộp"),
        ("below_25px_ratio", "Không đạt\n25 px"),
        ("roi_proxy_box_count", "Hộp trong\nROI proxy"),
        ("roi_proxy_below_25px_ratio", "Không đạt\ntrong proxy"),
    ]
    values = np.array([[float(row[key]) for key, _ in columns] for row in rows])
    normalized = np.zeros_like(values, dtype=float)
    for column in range(values.shape[1]):
        maximum = values[:, column].max()
        normalized[:, column] = values[:, column] / maximum if maximum else 0.0
    figure, axis = plt.subplots(figsize=(11, 5.6))
    axis.text(
        0.5,
        -0.34,
        "ROI proxy: tâm bbox thuộc đa giác x=55–98%, y=45–92% khung K230 (vùng phải/phía dưới). "
        "Proxy ưu tiên xe gần/lớn; AAU chỉ có n=2 hộp nên 0,00% không có ý nghĩa thống kê.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.7,
        wrap=True,
        color="#5b2830",
    )
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(columns)), [label for _, label in columns])
    axis.set_yticks(range(len(rows)), [row["dataset"] for row in rows])
    for row_index, row in enumerate(rows):
        for column_index, (key, _) in enumerate(columns):
            color = "white" if normalized[row_index, column_index] > 0.55 else "#102a43"
            axis.text(column_index, row_index, _format_cell(float(row[key]), key), ha="center", va="center", color=color, fontsize=11, weight="medium")
    axis.set_title("A2. Tổng quan bbox sau letterbox K230 640×480", fontsize=15, weight="bold", pad=14)
    axis.set_xlabel("Mỗi cột được chuẩn hoá riêng để biểu thị tương quan trong cột; số trong ô là giá trị gốc")
    figure.colorbar(image, ax=axis, label="Cường độ chuẩn hoá theo cột")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_condition_map(metrics: Sequence[dict[str, Any]], output: Path) -> None:
    display = [row for row in metrics if row["condition"] in {"DAY", "NIGHT", "RAIN"}]
    labels = [row["condition"] for row in display]
    values = [float(row["mAP50_95"]) * 100 if row["mAP50_95"] else None for row in display]
    counts = [int(row["ground_truth_boxes"]) for row in display]
    figure, axis = plt.subplots(figsize=(8.6, 5.2))
    bars = axis.bar(labels, [value or 0.0 for value in values], color=["#3f8f5b", "#355c9a", "#4c87a8"], width=0.62)
    for bar, value, count in zip(bars, values, counts):
        text = f"{value:.1f}%\nn={count} hộp" if value is not None else "N/A\nkhông có mẫu"
        axis.text(bar.get_x() + bar.get_width() / 2, (value or 0.0) + 1.2, text, ha="center", va="bottom", fontsize=10)
    available = [value for value in values if value is not None]
    axis.set_ylim(0, max(100.0, max(available, default=0.0) + 13.0))
    axis.set_ylabel("mAP50–95 (%)")
    axis.set_xlabel("Điều kiện (các nhóm có thể chồng lấp: RAIN theo thời tiết)")
    axis.set_title("A4. mAP50–95 theo điều kiện — AAU held-out", fontsize=15, weight="bold", pad=12)
    axis.set_xlabel("Điều kiện (các nhóm có thể chồng lấp; RAIN theo thời tiết)")
    axis.set_title("A4. mAP50–95 theo điều kiện — toàn bộ dataset", fontsize=15, weight="bold", pad=12)
    axis.text(0.5, -0.26, "Bao gồm train + val + cross-test: chỉ dùng tham khảo, không phải đánh giá độc lập.", transform=axis.transAxes, ha="center", va="top", fontsize=9, color="#7a2e2e")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_cross_test_condition_map(
    metrics: Sequence[dict[str, Any]],
    overlap: dict[str, Any],
    output: Path,
) -> None:
    labels = [str(row["condition"]) for row in metrics]
    values = [float(row["mAP50_95"]) * 100 if row["mAP50_95"] else 0.0 for row in metrics]
    reportable = [row["sample_status"] == "REPORTABLE" for row in metrics]
    counts = [int(row["ground_truth_boxes"]) for row in metrics]
    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    bars = axis.bar(labels, values, color=["#3f8f5b", "#355c9a", "#4c87a8"], width=0.62)
    for bar, value, count, is_reportable in zip(bars, values, counts, reportable):
        text = f"{value:.1f}%\nn={count} hộp" if is_reportable else f"N/A\nn={count} hộp"
        axis.text(bar.get_x() + bar.get_width() / 2, value + 1.2, text, ha="center", va="bottom", fontsize=10)
        if not is_reportable:
            bar.set_hatch("//")
            bar.set_edgecolor("#5b2830")
    axis.set_ylim(0, max(100.0, max(values, default=0.0) + 18.0))
    axis.set_ylabel("mAP50–95 (%)")
    axis.set_xlabel("Điều kiện; RAIN theo thời tiết, NIGHT theo ánh sáng")
    axis.set_title("A4. mAP50–95 theo điều kiện — chỉ cross-test", fontsize=15, weight="bold", pad=12)
    rain_ratio = overlap["rain_images_also_night_ratio"]
    night_ratio = overlap["night_images_also_rain_ratio"]
    overlap_text = (
        f"RAIN∩NIGHT: {overlap['overlap_image_count']} ảnh / {overlap['overlap_ground_truth_boxes']} hộp "
        f"({(rain_ratio or 0) * 100:.1f}% RAIN; {(night_ratio or 0) * 100:.1f}% NIGHT). "
        "Không diễn giải riêng tác động của mưa khi hai nhóm chồng lấp."
    )
    axis.text(0.5, -0.25, overlap_text, transform=axis.transAxes, ha="center", va="top", fontsize=9, color="#7a2e2e", wrap=True)
    axis.text(
        0.5,
        -0.40,
        f"Chỉ công bố lát có ≥{MIN_CONDITION_IMAGES} ảnh và ≥{MIN_CONDITION_BOXES} hộp; lát còn lại ghi N/A.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#374151",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _cross_test_metric(samples: Sequence[EvaluationSample]) -> dict[str, Any]:
    value = map50_95(samples)
    if value is None:
        raise ValueError("Cross-test metric has no ground-truth boxes")
    return {
        "scope": "AAU RainSnow + UA-DETRAC Original cross_test only; MIO-TCD excluded because it is train-only",
        "image_count": len(samples),
        "ground_truth_boxes": sum(len(sample.labels) for sample in samples),
        "mAP50_95": value,
        "ua_ignored_region_policy": "Predictions centered in a UA ignored_region are excluded before matching",
    }


def _plot_total_map(metric: dict[str, Any], output: Path) -> None:
    value = float(metric["mAP50_95"]) * 100
    figure, axis = plt.subplots(figsize=(7.8, 5.2))
    bar = axis.bar(["TỔNG held-out\nAAU + UA-DETRAC"], [value], color="#3d6ca8", width=0.55)[0]
    axis.text(bar.get_x() + bar.get_width() / 2, value + 1.6, f"{value:.1f}%\nn={metric['ground_truth_boxes']:,} hộp\n{metric['image_count']:,} ảnh", ha="center", va="bottom", fontsize=11)
    axis.set_ylim(0, max(100.0, value + 16.0))
    axis.set_ylabel("mAP50–95 (%)")
    axis.set_xlabel("Tập đánh giá độc lập với train")
    axis.set_title("A4. mAP50–95 tổng trên dữ liệu held-out", fontsize=15, weight="bold", pad=12)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _full_dataset_metric(samples: Sequence[EvaluationSample]) -> dict[str, Any]:
    value = map50_95(samples)
    if value is None:
        raise ValueError("Full-dataset reference metric has no ground-truth boxes")
    return {
        "scope": "Full exported dataset: AAU RainSnow + UA-DETRAC + MIO-TCD; train + val + cross_test; reference only",
        "image_count": len(samples),
        "ground_truth_boxes": sum(len(sample.labels) for sample in samples),
        "mAP50_95": value,
        "ua_ignored_region_policy": "Predictions centered in a UA ignored_region are excluded before matching",
    }


def _plot_recall_by_size(rows: Sequence[dict[str, Any]], range_rows: Sequence[dict[str, Any]], output: Path) -> None:
    labels = [str(row["size_group_k230_640x480"]) for row in rows]
    values = [float(row["recall_at_confidence_0_50"]) * 100 for row in rows]
    counts = [int(row["ground_truth_boxes"]) for row in rows]
    figure, (axis, range_axis) = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    figure.text(0.5, 0.01, "Phạm vi: chỉ cross-test AAU RainSnow + UA-DETRAC; MIO-TCD train-only nên bị loại.", ha="center", va="bottom", fontsize=8.7, color="#374151")
    bars = axis.bar(labels, values, color=["#b4534d", "#d69742", "#3f8f5b"], width=0.62)
    for bar, value, count in zip(bars, values, counts):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%\nn={count} hộp", ha="center", va="bottom", fontsize=10)
    axis.axvline(0.5, color="#4b5563", linestyle="--", linewidth=1.1)
    axis.text(0.5, 4, "Ngưỡng guard 25 px", rotation=90, ha="right", va="bottom", fontsize=9, color="#374151")
    axis.set_ylim(0, max(100.0, max(values, default=0.0) + 13.0))
    axis.set_ylabel("Recall @ IoU 0.50, confidence 0.50 (%)")
    axis.set_xlabel("Min(width, height) sau letterbox K230 640×480")
    axis.set_title("Recall theo kích thước bbox", fontsize=14, weight="bold", pad=12)
    axis.grid(axis="y", alpha=0.25)
    heights = [float(row["mounting_height_m"]) for row in range_rows]
    ranges = [float(row["estimated_ground_range_m"]) for row in range_rows]
    range_bars = range_axis.bar([f"{height:.0f} m" for height in heights], ranges, color="#4c8d68", width=0.62)
    for bar, ground_range in zip(range_bars, ranges):
        range_axis.text(bar.get_x() + bar.get_width() / 2, ground_range + 0.35, f"{ground_range:.1f} m", ha="center", va="bottom", fontsize=10)
    range_axis.set_ylim(0, max(ranges, default=0.0) + 4.0)
    range_axis.set_ylabel("Khoảng cách mặt đất ước tính (m)")
    range_axis.set_xlabel("Độ cao lắp camera")
    range_axis.set_title("Guard 25 px → khoảng cách", fontsize=14, weight="bold", pad=12)
    range_axis.grid(axis="y", alpha=0.25)
    range_axis.text(0.5, -0.25, "FOV ngang 90°, xe rộng 1,8 m. Kịch bản minh hoạ, không phải hiệu chuẩn K230.", transform=range_axis.transAxes, ha="center", va="top", fontsize=9, wrap=True)
    figure.suptitle("A5. Recall và quy đổi ngưỡng 25 px", fontsize=16, weight="bold", y=1.02)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_recall_by_dataset_size(rows: Sequence[dict[str, Any]], output: Path) -> None:
    groups = ("<25 px", "25â€“49 px", "â‰¥50 px")
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    for axis, dataset in zip(axes, (AAU_DATASET, UA_DATASET)):
        subset = {str(row["size_group_k230_640x480"]): row for row in rows if row["dataset"] == dataset}
        values = [float(subset[group]["recall_at_confidence_0_50"]) * 100 for group in groups]
        counts = [int(subset[group]["ground_truth_boxes"]) for group in groups]
        bars = axis.bar(groups, values, color=["#b4534d", "#d69742", "#3f8f5b"], width=0.62)
        for bar, value, count in zip(bars, values, counts):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%\nn={count}", ha="center", va="bottom", fontsize=10)
        axis.axvline(0.5, color="#4b5563", linestyle="--", linewidth=1.1)
        axis.set_title(dataset, weight="bold")
        axis.set_xlabel("Min(width, height) sau letterbox K230 640×480")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Recall @ IoU 0.50, confidence 0.50 (%)")
    axes[0].set_ylim(0, 105)
    figure.suptitle("A5. Recall theo domain và kích thước bbox — không gộp AAU + UA", fontsize=15, weight="bold")
    figure.text(0.5, 0.01, "AAU small-bbox được giữ riêng vì không thể dùng kết quả UA để che điểm yếu domain AAU.", ha="center", fontsize=9, color="#7a2e2e")
    figure.tight_layout(rect=(0, 0.05, 1, 0.92))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _diagnostics(samples: Sequence[EvaluationSample]) -> list[dict[str, Any]]:
    _, matched = _match_predictions(samples, 0.50, PRESENTATION_CONFIDENCE)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        relevant_predictions = [box for box in sample.predictions if box.confidence >= PRESENTATION_CONFIDENCE]
        local = EvaluationSample(sample.image_id, sample.image_path, sample.width, sample.height, sample.lighting, sample.weather, sample.labels, tuple(relevant_predictions))
        local_matches, _ = _match_predictions([local], 0.50, PRESENTATION_CONFIDENCE)
        true_positives = sum(int(value) for _, value in local_matches)
        rows.append({
            "sample": sample,
            "true_positives": true_positives,
            "false_negatives": len(sample.labels) - sum(int(matched.get((sample.image_id, index), False)) for index in range(len(sample.labels))),
            "false_positives": len(local_matches) - true_positives,
        })
    return rows


def _draw_box(draw: ImageDraw.ImageDraw, box: Box, scale_x: float, scale_y: float, offset_x: int, offset_y: int, color: str) -> None:
    x1, y1, x2, y2 = box.xyxy
    draw.rectangle((offset_x + x1 * scale_x, offset_y + y1 * scale_y, offset_x + x2 * scale_x, offset_y + y2 * scale_y), outline=color, width=2)


def _example_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _render_examples(samples: Sequence[EvaluationSample], output: Path) -> None:
    diagnostics = _diagnostics(samples)
    successes = sorted(diagnostics, key=lambda row: (row["false_negatives"], row["false_positives"], -row["true_positives"], row["sample"].image_id))[:3]
    failures = sorted(diagnostics, key=lambda row: (-row["false_negatives"], -row["false_positives"], row["true_positives"], row["sample"].image_id))[:3]
    selected = [("THÀNH CÔNG", row) for row in successes] + [("THẤT BẠI / CẦN XEM", row) for row in failures]
    tile_width, tile_height = 360, 260
    canvas = Image.new("RGB", (3 * tile_width, 2 * tile_height + 28), "#202020")
    draw = ImageDraw.Draw(canvas)
    title_font = _example_font(14)
    detail_font = _example_font(13)
    for index, (status, row) in enumerate(selected):
        sample: EvaluationSample = row["sample"]
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        image.thumbnail((tile_width - 14, tile_height - 48))
        x = (index % 3) * tile_width + (tile_width - image.width) // 2
        y = (index // 3) * tile_height + 28
        canvas.paste(image, (x, y))
        scale_x, scale_y = image.width / sample.width, image.height / sample.height
        for label in sample.labels:
            _draw_box(draw, label, scale_x, scale_y, x, y, "#31d36f")
        for prediction in sample.predictions:
            if prediction.confidence >= PRESENTATION_CONFIDENCE:
                _draw_box(draw, prediction, scale_x, scale_y, x, y, "#ff5959")
        color = "#6ee7a4" if status == "THÀNH CÔNG" else "#ff9b9b"
        draw.text(((index % 3) * tile_width + 8, (index // 3) * tile_height + 6), f"{status}: {sample.image_id}", fill=color, font=title_font)
        draw.text(((index % 3) * tile_width + 8, (index // 3 + 1) * tile_height - 20), f"TP={row['true_positives']}  FN={row['false_negatives']}  FP={row['false_positives']}", fill="white", font=detail_font)
    draw.text(
        (10, 2 * tile_height + 6),
        "Ưu tiên giảm FN: bỏ sót có thể không kích hoạt cảnh báo; FP chủ yếu gây giảm tốc không cần thiết.",
        fill="#ffd28a",
        font=detail_font,
    )
    canvas.save(output, quality=94)


def _write_report(
    output: Path,
    analysis: dict[str, Any],
    full_metric: dict[str, Any],
    condition_metrics: Sequence[dict[str, Any]],
    total_metric: dict[str, Any],
    recall_rows: Sequence[dict[str, Any]],
    pipeline_rows: Sequence[dict[str, Any]],
) -> None:
    range_rows = analysis.get("range_scenarios", [])
    range_text = "; ".join(
        f"cao {float(row['mounting_height_m']):.0f} m: {float(row['estimated_ground_range_m']):.1f} m"
        for row in range_rows
    ) or "chưa có"
    proxy = analysis["roi_proxy"]
    proxy_points = ", ".join(f"({x:.2f}, {y:.2f})" for x, y in proxy["polygon_normalized"])
    aau_proxy = next(row for row in analysis["bbox_size_by_dataset"] if row["dataset"] == AAU_DATASET)
    recall_sentence = "; ".join(
        f"{row['size_group_k230_640x480']}: {float(row['recall_at_confidence_0_50']) * 100:.1f}%"
        for row in recall_rows
    )
    condition_sentence = "; ".join(
        f"{row['condition']}: {float(row['mAP50_95']) * 100:.1f}% ({row['image_count']} ảnh, n={row['ground_truth_boxes']} bbox)"
        for row in condition_metrics
    )
    table = "\n".join(
        f"| {row['stage']} | {row['metric']} | {row['value']} | {row['status']} |"
        for row in pipeline_rows
    )
    report = f"""# Hình cho báo cáo đề tài

Chỉ có năm hình có bằng chứng hiện tại được đưa vào thân báo cáo. A3 và A6 vẫn được giữ dưới dạng trạng thái/tables để tránh biến số liệu chưa hợp lệ thành kết luận.

## A1. Sơ đồ kiến trúc hệ thống

![A1 kiến trúc](A1_edge_cloud_edge_architecture.png)

Kết luận: luồng cảnh báo là **Edge → Cloud → Edge**; K230 gửi telemetry lên CoreIoT và ESP32 chỉ nhận lệnh từ CoreIoT, không có kênh K230 → ESP32 trực tiếp.

## A2. Phân bố dataset và ROI proxy

![A2 phân bố dataset](A2_dataset_distribution_roi_proxy.png)

`ROI proxy` là đa giác chuẩn hoá trong khung giả lập K230 640×480 có các đỉnh: {proxy_points}. Một bbox thuộc proxy khi **tâm bbox** nằm trong đa giác. Trạng thái của nó là `{proxy['status']}`: đây là vùng ước lượng phục vụ thống kê dữ liệu công khai, không phải làn khẩn cấp đã hiệu chuẩn từ camera K230.

Kết luận: tỷ lệ fail trong proxy nhỏ hơn toàn ảnh có thể lạc quan vì vùng này ưu tiên đối tượng gần/lớn; đặc biệt AAU chỉ có **{aau_proxy['roi_proxy_box_count']} bbox** trong proxy nên 0,00% không có ý nghĩa thống kê. Khi có ROI K230 đã hiệu chuẩn, phải chạy lại toàn bộ A2.

## A3. Đường cong huấn luyện

Không đưa hình A3 vào thân báo cáo hiện tại. `results.csv` chỉ có mAP ở epoch cuối, trong khi 24 epoch đầu không chạy validation và mAP mặc định không áp dụng lọc `ignored_region` của UA-DETRAC. Đường loss có trong phụ lục `A3_train_loss_supplement.png`, nhưng không được diễn giải là hội tụ/overfit khi chưa có validation metrics theo epoch hợp lệ.

## A4. mAP theo điều kiện trên toàn bộ dataset

![A4 mAP theo điều kiện](A4_map_by_condition_full_dataset.png)

Kết luận: biểu đồ dùng đủ **{full_metric['image_count']:,} ảnh** và **{full_metric['ground_truth_boxes']:,} bbox** của AAU RainSnow + UA-DETRAC + MIO-TCD, gồm train, val và cross-test. {condition_sentence}. Các nhóm DAY/NIGHT/RAIN có thể chồng lấp và 15 ảnh twilight không nằm trong ba cột. Vì có ảnh train, đây chỉ là số liệu tham khảo để so sánh điều kiện, **không phải** mAP đánh giá độc lập. Prediction có tâm nằm trong `ignored_region` UA-DETRAC đã bị loại trước khi matching.

## A5. Recall theo kích thước bbox

![A5 recall theo kích thước và quy đổi mét](A5_recall_by_k230_bbox_size_and_range.png)

Kết luận trong phạm vi held-out AAU + UA: {recall_sentence}. Guard runtime cần `width ≥ 25 AND height ≥ 25` sau letterbox 640×480; nhóm `<25 px` vì thế không phải điều kiện vận hành hợp lệ.

Quy đổi 25 px không phải hằng số vì có phối cảnh. Kịch bản minh hoạ (FOV ngang 90°, xe rộng 1,8 m, chưa đo lens/pitch/lane thực) cho khoảng cách mặt đất: {range_text}. Đây chỉ là ước lượng; số vận hành phải dùng hiệu chuẩn K230 thực tế với bốn điểm mặt đường và `camera_ground_point_m`.

## A6. So sánh độ chính xác qua pipeline

| Điểm đo | Metric | Giá trị | Trạng thái |
|---|---|---:|---|
{table}

Không suy diễn suy giảm INT8 cho đến khi có evaluation cùng một test set ở NNCase simulator và trên board K230.

## A7. Ảnh minh hoạ kết quả

![A7 ví dụ đúng và lỗi](A7_success_and_failure_examples.png)

Xanh lá là nhãn; đỏ là prediction tại confidence 0,50. Kết luận: báo cáo giữ đồng thời ví dụ đúng và lỗi (FN/FP) để phục vụ phân tích sai số, không chỉ chọn ảnh đẹp.

## Giới hạn bắt buộc ghi trong báo cáo

> mAP trong báo cáo này đo khả năng nhận diện `vehicle` trên dữ liệu công khai. Các dataset công khai không có ground truth xe dừng trong làn khẩn cấp; vì vậy chúng không chứng minh đèn cảnh báo bật đúng lúc. Hệ thống cần bộ dữ liệu tự thu bằng K230 đã khoá test, ROI thật và nhãn trạng thái dừng để đánh giá DR/FAR/MTTD tại vị trí triển khai.
"""
    (output / "REPORT_FIGURES.md").write_text(report, encoding="utf-8")


def _write_cross_test_report(
    output: Path,
    analysis: dict[str, Any],
    cross_test_metric: dict[str, Any],
    condition_metrics: Sequence[dict[str, Any]],
    overlap: dict[str, Any],
    recall_rows: Sequence[dict[str, Any]],
    pipeline_rows: Sequence[dict[str, Any]],
) -> None:
    proxy = analysis["roi_proxy"]
    proxy_points = ", ".join(f"({x:.2f}, {y:.2f})" for x, y in proxy["polygon_normalized"])
    aau_proxy = next(row for row in analysis["bbox_size_by_dataset"] if row["dataset"] == AAU_DATASET)
    range_text = "; ".join(
        f"cao {float(row['mounting_height_m']):.0f} m: {float(row['estimated_ground_range_m']):.1f} m"
        for row in analysis.get("range_scenarios", [])
    ) or "chưa có"
    condition_sentence = "; ".join(
        (
            f"{row['condition']}: {float(row['mAP50_95']) * 100:.1f}% "
            f"({row['image_count']} ảnh, n={row['ground_truth_boxes']} bbox)"
            if row["mAP50_95"]
            else f"{row['condition']}: N/A ({row['image_count']} ảnh, n={row['ground_truth_boxes']} bbox)"
        )
        for row in condition_metrics
    )
    recall_sentence = "; ".join(
        f"{row['size_group_k230_640x480']}: {float(row['recall_at_confidence_0_50']) * 100:.1f}%"
        for row in recall_rows
    )
    table = "\n".join(
        f"| {row['stage']} | {row['metric']} | {row['value']} | {row['status']} |"
        for row in pipeline_rows
    )
    report = f"""# Hình cho báo cáo đề tài

Mọi chỉ số hiệu năng trong báo cáo này đều được đo **chỉ trên `cross_test` đã khoá**. Train và validation không được gộp vào mAP, recall hoặc precision công bố.

## A1. Sơ đồ kiến trúc hệ thống

![A1 kiến trúc](A1_edge_cloud_edge_architecture.png)

Luồng cảnh báo là **Edge → Cloud → Edge**: K230 gửi telemetry lên CoreIoT, ESP32 chỉ nhận lệnh từ CoreIoT; không có kênh K230 → ESP32 trực tiếp. Hai trạm Edge được triển khai cách nhau khoảng **200–500 m** để cảnh báo đủ sớm cho xe phía sau phản ứng.

## A2. Phân bố dataset và ROI proxy

![A2 phân bố dataset](A2_dataset_distribution_roi_proxy.png)

`ROI proxy` là đa giác trong khung K230 640×480 có các đỉnh chuẩn hoá: {proxy_points}; tương ứng vùng phải/phía dưới, x=55–98% và y=45–92% khung. Một bbox thuộc proxy khi **tâm bbox** nằm trong đa giác. Đây là ước lượng cho dữ liệu công khai (`{proxy['status']}`), không phải làn khẩn cấp đã hiệu chuẩn từ K230. Vì proxy ưu tiên xe gần/lớn, tỷ lệ không đạt trong proxy có thể lạc quan hơn toàn khung.

Đặc biệt, AAU chỉ có **{aau_proxy['roi_proxy_box_count']} bbox** trong proxy: ô 0,00% đi kèm không có ý nghĩa thống kê và không thể diễn giải là AAU đạt 100%. Khi có ROI K230 đã hiệu chuẩn, phải chạy lại toàn bộ A2.

## A4. mAP theo điều kiện trên cross-test

![A4 mAP theo điều kiện](A4_map_by_condition_cross_test.png)

Cross-test hiện có **{cross_test_metric['image_count']:,} ảnh** và **{cross_test_metric['ground_truth_boxes']:,} bbox**, từ AAU RainSnow và UA-DETRAC. {condition_sentence}. Các lát chỉ được hiển thị mAP khi có ít nhất {MIN_CONDITION_IMAGES} ảnh và {MIN_CONDITION_BOXES} bbox; lát thiếu mẫu phải ghi N/A.

Mức chồng lấn bắt buộc phải nêu: **{overlap['overlap_image_count']} ảnh** ({overlap['overlap_ground_truth_boxes']} bbox) thuộc đồng thời RAIN và NIGHT — bằng **{float(overlap['rain_images_also_night_ratio'] or 0) * 100:.1f}%** số ảnh RAIN và **{float(overlap['night_images_also_rain_ratio'] or 0) * 100:.1f}%** số ảnh NIGHT. Do đó không thể kết luận riêng “mưa tệ hơn đêm” từ lát này.

MIO-TCD là train-only, không có sequence cross-test độc lập nên bị loại khỏi A4 và A5. Đây là giới hạn của tập test hiện tại, không được bù bằng cách đưa MIO train/val vào chỉ số công bố. Các số train+val+cross-test chỉ được giữ dưới `internal_diagnostics/` để chẩn đoán overfit.

## A5. Recall theo kích thước bbox

![A5 recall theo kích thước và quy đổi mét](A5_recall_by_k230_bbox_size_and_range.png)

Trên cùng cross-test AAU + UA: {recall_sentence}. Guard runtime là `width ≥ 25 AND height ≥ 25` sau letterbox 640×480; vì vậy nhóm `<25 px` không phải điều kiện vận hành hợp lệ.

Quy đổi 25 px phụ thuộc phối cảnh. Với kịch bản minh hoạ FOV ngang 90°, xe rộng 1,8 m (chưa đo lens/pitch/lane thực tế), khoảng cách mặt đất ước tính là {range_text}. Đây chỉ là ước lượng; thông số vận hành phải dùng hiệu chuẩn K230 với bốn điểm mặt đường và `camera_ground_point_m`.

## A6. So sánh độ chính xác qua pipeline

| Điểm đo | Metric | Giá trị | Trạng thái |
|---|---|---:|---|
{table}

Không suy diễn suy giảm INT8 cho đến khi có evaluation trên **cùng cross-test** ở NNCase simulator và board K230.

## A7. Ảnh minh hoạ kết quả

![A7 ví dụ đúng và lỗi](A7_success_and_failure_examples.png)

Xanh lá là nhãn; đỏ là prediction tại confidence 0,50. Hình giữ cả ca thành công và thất bại với TP/FN/FP. Với hệ thống cảnh báo an toàn, **FN (bỏ sót) nguy hiểm hơn FP (báo nhầm)**: FN có thể làm đèn không bật cho xe chạy phía sau, còn FP chủ yếu gây giảm tốc không cần thiết. Vì vậy các lỗi FN phải là ưu tiên cải thiện.

## Giới hạn bắt buộc ghi trong báo cáo

> mAP và recall ở đây đo khả năng nhận diện `vehicle` trên cross-test dữ liệu công khai. Các dataset này không có ground truth xe dừng trong làn khẩn cấp; vì vậy chúng không chứng minh đèn cảnh báo bật đúng lúc. Hệ thống cần bộ dữ liệu tự thu bằng K230 đã khoá test, ROI thật và nhãn trạng thái dừng để đánh giá DR/FAR/MTTD tại vị trí triển khai.
"""
    (output / "REPORT_FIGURES.md").write_text(report, encoding="utf-8")


def _write_separated_report(
    output: Path,
    analysis: dict[str, Any],
    domain_metrics: Sequence[dict[str, Any]],
    recall_rows: Sequence[dict[str, Any]],
    pipeline_rows: Sequence[dict[str, Any]],
) -> None:
    """Write the public report without A4 or any AAU+UA aggregate headline."""
    proxy = analysis["roi_proxy"]
    aau_proxy = next(row for row in analysis["bbox_size_by_dataset"] if row["dataset"] == AAU_DATASET)
    aau_small = next(
        row for row in recall_rows
        if row["dataset"] == AAU_DATASET and row["size_group_k230_640x480"] == "<25 px"
    )
    aau_small_recall = float(aau_small["recall_at_confidence_0_50"]) * 100
    metric_rows = "\n".join(
        f"| {row['dataset']} | {row['image_count']} | {row['ground_truth_boxes']} | "
        f"{(float(row['mAP50_95']) * 100 if row['mAP50_95'] else 'NOT_MEASURED')} | {row['status']} |"
        for row in domain_metrics
    )
    pipeline_table = "\n".join(
        f"| {row['stage']} | {row['metric']} | {row['value']} | {row['status']} |"
        for row in pipeline_rows
    )
    report = f"""# Hình cho báo cáo đề tài — bản car-only, confidence 0,50

Chỉ dùng `cross_test` đã khoá. Không có số headline gộp AAU + UA-DETRAC;
không có A4 và không dùng biểu đồ train+val+cross-test trong báo cáo.

## A1. Sơ đồ kiến trúc hệ thống

![A1 kiến trúc](A1_edge_cloud_edge_architecture.png)

## A2. Phân bố dataset và ROI proxy

![A2 phân bố dataset](A2_dataset_distribution_roi_proxy.png)

Chỉ giữ A2; overview heatmap trùng lặp bị loại. ROI proxy chỉ là ước lượng dữ
liệu công khai (`{proxy['status']}`), không phải ROI đã hiệu chuẩn K230. AAU
chỉ có **{aau_proxy['roi_proxy_box_count']} bbox** trong proxy nên phải ghi:
**chưa đủ dữ liệu để đo đáng tin cậy**. Không suy rộng từ MIO hoặc UA-DETRAC.

## Kết quả detector tách riêng theo domain

| Domain | Ảnh cross-test | Bbox | mAP50–95 | Trạng thái |
|---|---:|---:|---:|---|
{metric_rows}

Mọi metric được rerun ở confidence **0,50**. Không dùng số confidence 0,25,
không cộng AAU và UA-DETRAC để che chênh lệch domain.

## A5. Recall theo domain và kích thước bbox

![A5 recall tách domain](A5_recall_by_dataset_size_confidence_050.png)

Điểm yếu phải nêu rõ: AAU ở nhóm `<25 px` có recall **{aau_small_recall:.1f}%**
tại confidence 0,50 (mốc chẩn đoán trước đây xấp xỉ 16,7% phải được đối chiếu
lại bằng đúng release này). Không dùng các tỷ lệ gộp 46,1% / 88,4% / 92,2%.

## A6. Chuỗi chứng cứ từ model đến K230

| Điểm đo | Metric | Giá trị | Trạng thái |
|---|---|---:|---|
{pipeline_table}

Không suy rộng mAP public cross-test thành hiệu năng board/camera thực địa.
DAY/NIGHT/RAIN/BACKLIT chỉ có thể báo cáo sau khi K230 tự quay, gán nhãn,
review và khoá session test.

## A7. Ảnh minh hoạ kết quả

![A7 ví dụ đúng và lỗi](A7_success_and_failure_examples.png)

Xanh lá là nhãn; đỏ là prediction tại confidence 0,50. Giữ cả TP/FN/FP để ưu
tiên giảm FN.

## Chính sách nhãn

`motorcycle`, `motorbike`, `bicycle` không là positive. Chúng được ghi thành
`IGNORE_REGION`, được che trên bản copy train và bị loại khỏi matching khi tâm
prediction nằm trong vùng ignore. Xem `28_car_only_k230_release_policy.md`.
"""
    (output / "REPORT_FIGURES.md").write_text(report, encoding="utf-8")


def _plot_loss_supplement(run_dir: Path, output: Path) -> None:
    rows = _read_csv(run_dir / "results.csv")
    fields = (("train/box_loss", "box loss"), ("train/cls_loss", "cls loss"), ("train/dfl_loss", "dfl loss"))
    figure, axis = plt.subplots(figsize=(8.6, 5.2))
    epochs = [int(row["epoch"]) for row in rows]
    for field, label in fields:
        axis.plot(epochs, [float(row[field]) for row in rows], marker="o", markersize=3.5, label=label)
    axis.set(xlabel="Epoch", ylabel="Train loss", title="Phụ lục A3. Train loss — không diễn giải mAP/overfit")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _generate_report_figures_legacy(run_dir: Path, analysis_json: Path, scene_metadata: Path, output: Path, device: str) -> dict[str, Any]:
    analysis = json.loads(analysis_json.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    _plot_architecture(output / "A1_edge_cloud_edge_architecture.png")
    _plot_dataset_overview(analysis, output / "A2_dataset_distribution_roi_proxy.png")
    _plot_loss_supplement(run_dir, output / "A3_train_loss_supplement.png")
    full_samples = _load_full_dataset(run_dir, scene_metadata, run_dir / "weights" / "best.pt", device)
    heldout_samples = [
        sample for sample in full_samples
        if sample.dataset in {AAU_DATASET, UA_DATASET} and sample.split in {"val", "cross_test"}
    ]
    if not heldout_samples:
        raise ValueError("No held-out AAU/UA samples found in the run directory")
    full_metric = _full_dataset_metric(full_samples)
    condition_metrics = _condition_metrics(full_samples)
    total_metric = _total_metric(heldout_samples)
    recall_rows = _recall_by_size(heldout_samples)
    _write_csv(output / "A4_map_by_condition_full_dataset.csv", list(condition_metrics[0]), condition_metrics)
    _write_csv(output / "A5_recall_by_k230_bbox_size.csv", list(recall_rows[0]), recall_rows)
    _plot_condition_map(condition_metrics, output / "A4_map_by_condition_full_dataset.png")
    _plot_recall_by_size(recall_rows, analysis.get("range_scenarios", []), output / "A5_recall_by_k230_bbox_size_and_range.png")
    _render_examples(heldout_samples, output / "A7_success_and_failure_examples.png")
    pipeline_rows = [
        {
            "stage": "PyTorch float32 trên PC",
            "metric": "Custom mAP50–95, tổng AAU + UA held-out",
            "value": f"{float(total_metric['mAP50_95']) * 100:.2f}%",
            "status": "AVAILABLE",
        },
        {
            "stage": "NNCase/K230 INT8 simulator",
            "metric": "Cùng test set và cùng post-processing",
            "value": "N/A",
            "status": "PENDING_SIMULATOR_EVALUATION",
        },
        {
            "stage": "K230 board INT8",
            "metric": "Cùng test set trên thiết bị",
            "value": "N/A",
            "status": "BLOCKED_BOARD_RUN_REQUIRED",
        },
    ]
    _write_csv(output / "A6_pipeline_accuracy_comparison.csv", list(pipeline_rows[0]), pipeline_rows)
    statuses = [
        {"figure": "A1", "status": "READY", "reason": "Architecture confirmed as Edge-to-Cloud-to-Edge."},
        {"figure": "A2", "status": "READY_WITH_PROXY_LIMITATION", "reason": "Public-data ROI proxy is explicitly defined; AAU proxy n=2."},
        {"figure": "A3", "status": "APPENDIX_ONLY", "reason": "No valid ignored-region-filtered validation metrics per epoch."},
        {"figure": "A4", "status": "READY_FULL_DATASET_REFERENCE_SCOPE", "reason": "Custom mAP50-95 by DAY/NIGHT/RAIN across all exported AAU + UA + MIO images, including train; UA ignored-region predictions filtered."},
        {"figure": "A5", "status": "READY_TOTAL_HELD_OUT_SCOPE", "reason": "Recall at IoU 0.50/confidence 0.50 by K230-letterboxed bbox size plus 25 px range scenarios."},
        {"figure": "A6", "status": "PENDING", "reason": "K230 compile passed; simulator and board accuracy runs are absent."},
        {"figure": "A7", "status": "READY_TOTAL_HELD_OUT_SCOPE", "reason": "Selected success and failure samples from the evaluated held-out data."},
    ]
    _write_csv(output / "report_figure_status.csv", list(statuses[0]), statuses)
    _write_report(output, analysis, full_metric, condition_metrics, total_metric, recall_rows, pipeline_rows)
    summary = {
        "output": str(output.resolve()),
        "evaluation_scope": "A4: full exported dataset (train + val + cross-test); A5/A6/A7: AAU + UA-DETRAC held-out validation + cross-test; one-class vehicle detector; UA ignored-region predictions excluded",
        "full_dataset_metric": full_metric,
        "a4_condition_metrics": condition_metrics,
        "heldout_metric": total_metric,
        "recall_by_size": recall_rows,
        "pipeline_comparison": pipeline_rows,
        "statuses": statuses,
    }
    (output / "report_figure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def generate_report_figures(run_dir: Path, analysis_json: Path, scene_metadata: Path, output: Path, device: str) -> dict[str, Any]:
    """Generate report figures using only independent cross-test measurements."""
    analysis = json.loads(analysis_json.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    _plot_architecture(output / "A1_edge_cloud_edge_architecture.png")
    _plot_dataset_overview(analysis, output / "A2_dataset_distribution_roi_proxy.png")
    _plot_loss_supplement(run_dir, output / "A3_train_loss_supplement.png")

    full_samples = _load_full_dataset(run_dir, scene_metadata, run_dir / "weights" / "best.pt", device)
    cross_test_samples = [sample for sample in full_samples if sample.split == "cross_test"]
    if not cross_test_samples:
        raise ValueError("No cross_test samples found in the run directory")

    cross_test_scope = (
        "Locked cross_test only; AAU RainSnow and UA-DETRAC Original are reported separately; "
        "MIO-TCD is train-only and excluded; thresholded metrics use confidence 0.50"
    )
    domain_metrics = _metric_by_dataset(cross_test_samples)
    recall_rows = _recall_by_dataset_size(cross_test_samples)
    _write_csv(output / "A5_recall_by_dataset_size_confidence_050.csv", list(recall_rows[0]), recall_rows)
    _write_csv(output / "domain_metrics_cross_test_confidence_050.csv", list(domain_metrics[0]), domain_metrics)
    _plot_recall_by_dataset_size(recall_rows, output / "A5_recall_by_dataset_size_confidence_050.png")
    _render_examples(cross_test_samples, output / "A7_success_and_failure_examples.png")
    pipeline_rows = [
        *[
            {
                "stage": "PyTorch float32 on PC",
                "metric": f"mAP50-95 {row['dataset']} cross_test, confidence 0.50",
                "value": f"{float(row['mAP50_95']) * 100:.2f}%" if row["mAP50_95"] else "NOT_MEASURED",
                "status": row["status"],
            }
            for row in domain_metrics
        ],
        {
            "stage": "NNCase/K230 INT8 simulator",
            "metric": "Per-domain cross_test at confidence 0.50",
            "value": "NOT_MEASURED",
            "status": "PENDING_SIMULATOR_EVALUATION",
        },
        {
            "stage": "K230 board INT8",
            "metric": "Locked K230 DAY/NIGHT/RAIN/BACKLIT tests",
            "value": "NOT_MEASURED",
            "status": "BLOCKED_BOARD_RUN_REQUIRED",
        },
    ]
    _write_csv(output / "A6_pipeline_accuracy_comparison.csv", list(pipeline_rows[0]), pipeline_rows)
    statuses = [
        {"figure": "A1", "status": "READY", "reason": "Architecture uses Edge-to-Cloud-to-Edge."},
        {"figure": "A2", "status": "READY_WITH_PROXY_LIMITATION", "reason": "Public-data ROI proxy only; AAU proxy n=2 is not reliable."},
        {"figure": "A3", "status": "APPENDIX_ONLY", "reason": "No valid ignored-region-filtered validation metrics per epoch."},
        {"figure": "A4", "status": "REMOVED", "reason": "Removed: aggregation and duplicate condition presentation are prohibited."},
        {"figure": "A5", "status": "READY_SEPARATED_DOMAIN_SCOPE", "reason": "AAU and UA-DETRAC recall are separate at confidence 0.50."},
        {"figure": "A6", "status": "PENDING", "reason": "Board run and locked K230 sessions are absent."},
        {"figure": "A7", "status": "READY_CROSS_TEST_SCOPE", "reason": "Success and failure samples come from cross_test."},
    ]
    _write_csv(output / "report_figure_status.csv", list(statuses[0]), statuses)
    _write_separated_report(output, analysis, domain_metrics, recall_rows, pipeline_rows)
    summary = {
        "output": str(output.resolve()),
        "evaluation_scope": cross_test_scope,
        "confidence": PRESENTATION_CONFIDENCE,
        "domain_metrics": domain_metrics,
        "recall_by_dataset_size": recall_rows,
        "pipeline_comparison": pipeline_rows,
        "statuses": statuses,
    }
    (output / "report_figure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary

    cross_test_scope = (
        "AAU RainSnow + UA-DETRAC Original cross_test only; MIO-TCD is train-only and excluded; "
        "UA ignored-region predictions excluded"
    )
    cross_test_metric = _cross_test_metric(cross_test_samples)
    condition_metrics = _condition_metrics(cross_test_samples, cross_test_scope)
    overlap = _condition_overlap(cross_test_samples)
    recall_rows = _recall_by_size(cross_test_samples)

    _write_csv(output / "A4_map_by_condition_cross_test.csv", list(condition_metrics[0]), condition_metrics)
    _write_csv(output / "A4_rain_night_overlap_cross_test.csv", list(overlap), [overlap])
    _write_csv(output / "A5_recall_by_k230_bbox_size.csv", list(recall_rows[0]), recall_rows)
    _plot_cross_test_condition_map(condition_metrics, overlap, output / "A4_map_by_condition_cross_test.png")
    _plot_recall_by_size(recall_rows, analysis.get("range_scenarios", []), output / "A5_recall_by_k230_bbox_size_and_range.png")
    _render_examples(cross_test_samples, output / "A7_success_and_failure_examples.png")

    # Preserve train/validation-inclusive figures exclusively for internal
    # diagnostics. They are intentionally not linked from REPORT_FIGURES.md.
    internal_output = output / "internal_diagnostics"
    internal_output.mkdir(parents=True, exist_ok=True)
    internal_scope = (
        "Internal diagnostic only: full exported AAU RainSnow + UA-DETRAC + MIO-TCD "
        "dataset (train + val + cross_test); not an independent performance metric"
    )
    internal_condition_metrics = _condition_metrics(full_samples, internal_scope)
    _write_csv(
        internal_output / "A4_map_by_condition_train_val_cross_test.csv",
        list(internal_condition_metrics[0]),
        internal_condition_metrics,
    )
    _plot_condition_map(internal_condition_metrics, internal_output / "A4_map_by_condition_train_val_cross_test.png")

    pipeline_rows = [
        {
            "stage": "PyTorch float32 trên PC",
            "metric": "Custom mAP50–95, tổng AAU + UA cross-test",
            "value": f"{float(cross_test_metric['mAP50_95']) * 100:.2f}%",
            "status": "AVAILABLE",
        },
        {
            "stage": "NNCase/K230 INT8 simulator",
            "metric": "Cùng cross-test và cùng post-processing",
            "value": "N/A",
            "status": "PENDING_SIMULATOR_EVALUATION",
        },
        {
            "stage": "K230 board INT8",
            "metric": "Cùng cross-test trên thiết bị",
            "value": "N/A",
            "status": "BLOCKED_BOARD_RUN_REQUIRED",
        },
    ]
    _write_csv(output / "A6_pipeline_accuracy_comparison.csv", list(pipeline_rows[0]), pipeline_rows)
    statuses = [
        {"figure": "A1", "status": "READY", "reason": "Architecture includes the 200–500 m Edge 1–Edge 2 deployment distance."},
        {"figure": "A2", "status": "READY_WITH_PROXY_LIMITATION", "reason": "ROI proxy bounds and center rule are explicit; AAU proxy n=2 is flagged as non-inferential."},
        {"figure": "A3", "status": "APPENDIX_ONLY", "reason": "No valid ignored-region-filtered validation metrics per epoch."},
        {"figure": "A4", "status": "READY_CROSS_TEST_WITH_RAIN_NIGHT_CONFOUND", "reason": "mAP50-95 uses cross_test only; RAIN/NIGHT overlap is reported explicitly."},
        {"figure": "A5", "status": "READY_CROSS_TEST_SCOPE", "reason": "Recall uses AAU + UA cross_test only; MIO is train-only and excluded."},
        {"figure": "A6", "status": "PENDING", "reason": "K230 compile passed; simulator and board accuracy runs are absent."},
        {"figure": "A7", "status": "READY_CROSS_TEST_SCOPE", "reason": "Success and failure samples come from cross_test; the caption prioritises FN reduction."},
    ]
    _write_csv(output / "report_figure_status.csv", list(statuses[0]), statuses)
    _write_cross_test_report(output, analysis, cross_test_metric, condition_metrics, overlap, recall_rows, pipeline_rows)
    summary = {
        "output": str(output.resolve()),
        "evaluation_scope": cross_test_scope,
        "cross_test_metric": cross_test_metric,
        "a4_condition_metrics": condition_metrics,
        "a4_rain_night_overlap": overlap,
        "recall_by_size": recall_rows,
        "pipeline_comparison": pipeline_rows,
        "internal_diagnostics": str(internal_output.resolve()),
        "statuses": statuses,
    }
    (output / "report_figure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--scene-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = generate_report_figures(
        args.run_dir.resolve(), args.analysis_json.resolve(), args.scene_metadata.resolve(), args.output.resolve(), args.device
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
