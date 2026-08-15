"""Export MIO-TCD, AAU RainSnow and UA-DETRAC as one one-class YOLO dataset.

The exporter is deliberately conservative: MIO is always train-only, AAU and
UA-DETRAC are split by their complete sequence, and an existing output is
never overwritten. Every retained annotation records its source class; every
rejected annotation records why it was not exported. Two-wheel annotations are
a separate third state: they are retained as ignore regions and are never
silently converted into negative/background training examples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sqlite3
import tarfile
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image

from external_eda_common import ROOT, clip_bbox_to_image, load_yaml, read_csv, safe_sequence, validate_bbox
from ignored_region import IgnoredRegion, mask_image_bytes, parse_ua_ignored_regions
from yolo_export_qc import audit_yolo_dataset, write_report


DEFAULT_MIO = ROOT / "storage_placeholders" / "online_data" / "raw" / "mio_tcd" / "MIO-TCD-Localization.tar"
DEFAULT_AAU = ROOT / "storage_placeholders" / "online_data" / "raw" / "aau_rainsnow" / "aau-rainsnow.zip"
DEFAULT_UA = ROOT / "storage_placeholders" / "online_data" / "raw" / "ua_detrac_orig" / "ua-detrac-orig.zip"
DEFAULT_OUTPUT = ROOT / "dataset_output" / "dataset_v1"
DEFAULT_SPLITS = ROOT / "reports" / "external_eda" / "split_proposal.csv"
DEFAULT_MAPPING = ROOT / "configs" / "vehicle_class_mapping.yaml"

SPLIT_NAMES = {
    "EXTERNAL_TRAIN": "train",
    "EXTERNAL_VALIDATION": "val",
    "CROSS_DATASET_TEST": "cross_test",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

IMAGE_FIELDS = [
    "image_id", "dataset", "split", "sequence_id", "frame_id", "source_image",
    "exported_image", "exported_label", "width", "height", "vehicle_box_count",
    "boundary_clipped_box_count", "ignored_region_mask_applied", "ignored_region_count",
]
ANNOTATION_FIELDS = [
    "image_id", "dataset", "split", "sequence_id", "frame_id", "annotation_id", "track_id",
    "original_class", "mapped_class", "class_id", "source_image", "source_annotation",
    "raw_xmin", "raw_ymin", "raw_xmax", "raw_ymax", "clipped_xmin", "clipped_ymin",
    "clipped_xmax", "clipped_ymax", "clip_applied", "clip_adjustments", "preserve_original_class",
]
REJECTED_FIELDS = [
    "dataset", "image_id", "sequence_id", "frame_id", "annotation_id", "track_id", "original_class",
    "source_image", "source_annotation", "action", "reason",
]
IGNORED_ANNOTATION_FIELDS = [
    "image_id", "dataset", "split", "sequence_id", "frame_id", "annotation_id", "track_id",
    "original_class", "source_image", "source_annotation", "xmin", "ymin", "xmax", "ymax",
    "handling", "reason",
]
SEQUENCE_FIELDS = ["dataset", "sequence_id", "split", "split_source"]
NEGATIVE_FIELDS = [
    "image_id", "dataset", "split", "sequence_id", "frame_id",
    "exported_image", "exported_label", "negative_reason",
]

PILOT_DATASETS = {
    "MIO-TCD Localization",
    "AAU RainSnow",
    "UA-DETRAC Original",
}


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def _image_size(data: bytes, source: str) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unreadable image {source}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Image has invalid dimensions {source}: {width}x{height}")
    return int(width), int(height)


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _yolo_line(box: tuple[float, float, float, float], width: int, height: int) -> str:
    xmin, ymin, xmax, ymax = box
    values = (
        ((xmin + xmax) / 2.0) / width,
        ((ymin + ymax) / 2.0) / height,
        (xmax - xmin) / width,
        (ymax - ymin) / height,
    )
    if not all(0.0 <= value <= 1.0 for value in values) or values[2] <= 0 or values[3] <= 0:
        raise ValueError(f"YOLO value outside valid range: {values}")
    return "0 " + " ".join(f"{value:.8f}" for value in values)


def _split_map(path: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for row in read_csv(path):
        dataset = str(row.get("dataset_name", "")).strip()
        sequence = str(row.get("sequence_id", "")).strip()
        proposed = str(row.get("proposed_split", "")).strip()
        if dataset and sequence and proposed in SPLIT_NAMES:
            key = (dataset, sequence)
            old = result.get(key)
            if old and old != SPLIT_NAMES[proposed]:
                raise ValueError(f"Sequence has multiple split assignments: {dataset}/{sequence}")
            result[key] = SPLIT_NAMES[proposed]
    if not result:
        raise ValueError(f"No usable split assignments in {path}")
    return result


def _normalise_source_path(value: str) -> str:
    """Return a case-insensitive, archive-independent path representation."""
    return value.replace("\\", "/").strip().lstrip("./").casefold()


def _truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


class _Selection:
    """Exact source-image selection for a versioned subset release.

    The source archives occasionally add one outer directory.  A selected
    source is therefore matched exactly first, then by an unambiguous suffix.
    Ambiguous suffixes are rejected instead of silently exporting a different
    image than the manifest requested.
    """

    def __init__(self, manifest_path: Path, subset: str, rows: list[dict[str, str]], proposal_override_used: bool):
        self.manifest_path = manifest_path
        self.subset = subset
        self.rows = rows
        self.proposal_override_used = proposal_override_used
        self._remaining: dict[str, set[str]] = defaultdict(set)
        self._counts: Counter[str] = Counter()
        for row in rows:
            dataset = str(row["dataset_name"])
            source = _normalise_source_path(str(row["source_file"]))
            self._remaining[dataset].add(source)
            self._counts[dataset] += 1

    @classmethod
    def load(
        cls,
        manifest_path: Path,
        subset: str,
        allow_proposal: bool,
        datasets: set[str] | None = None,
    ) -> "_Selection":
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Selection manifest not found: {manifest_path}")
        rows = [
            row for row in read_csv(manifest_path)
            if str(row.get("target_subset", "")).strip() == subset
            and (datasets is None or str(row.get("dataset_name", "")).strip() in datasets)
        ]
        if not rows:
            source_scope = f" and datasets {sorted(datasets)!r}" if datasets else ""
            raise ValueError(f"Selection manifest contains no rows for subset {subset!r}{source_scope}")
        invalid = [
            row for row in rows
            if str(row.get("dataset_name", "")).strip() not in PILOT_DATASETS
            or not str(row.get("source_file", "")).strip()
        ]
        if invalid:
            raise ValueError(f"Selection manifest has {len(invalid)} row(s) with an unsupported dataset or empty source_file")
        keys = [
            (str(row["dataset_name"]).strip(), _normalise_source_path(str(row["source_file"])))
            for row in rows
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Selection manifest contains duplicate dataset/source_file rows")
        proposed_rows = [row for row in rows if not _truthy(row.get("selected", ""))]
        if proposed_rows and not allow_proposal:
            raise ValueError(
                "Selection rows are still PROPOSAL_ONLY. Review and mark selected=TRUE, "
                "or pass --allow-proposal for the explicitly provisional pilot release."
            )
        return cls(manifest_path, subset, rows, proposal_override_used=bool(proposed_rows))

    def _match(self, dataset: str, source_image: str) -> str | None:
        source = _normalise_source_path(source_image)
        candidates = self._remaining.get(dataset, set())
        if source in candidates:
            return source
        suffix_matches = [
            candidate for candidate in candidates
            if source.endswith("/" + candidate) or candidate.endswith("/" + source)
        ]
        if len(suffix_matches) > 1:
            raise ValueError(
                f"Ambiguous selection source match for {dataset}: {source_image}; "
                f"matches={len(suffix_matches)}"
            )
        return suffix_matches[0] if suffix_matches else None

    def includes(self, dataset: str, source_image: str) -> bool:
        return self._match(dataset, source_image) is not None

    def consume(self, dataset: str, source_image: str) -> None:
        matched = self._match(dataset, source_image)
        if matched is None:
            raise AssertionError(f"Tried to consume unselected image: {dataset}/{source_image}")
        self._remaining[dataset].remove(matched)

    def assert_complete(self, writer: "_DatasetWriter") -> None:
        missing = [
            f"{dataset}:{source}"
            for dataset, sources in sorted(self._remaining.items())
            for source in sorted(sources)
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"{len(missing)} selected source image(s) were not found in the supplied archives: {preview}")
        if writer.counts["exported_images"] != len(self.rows):
            raise AssertionError(
                f"Subset export count mismatch: expected={len(self.rows)}, "
                f"exported={writer.counts['exported_images']}"
            )
        actual = Counter()
        for dataset, counts in writer.dataset_counts.items():
            actual[dataset] = counts["exported_images"]
        if actual != self._counts:
            raise AssertionError(f"Subset source allocation mismatch: expected={dict(self._counts)}, actual={dict(actual)}")

    def metadata(self) -> dict[str, Any]:
        digest = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        return {
            "selection_manifest": str(self.manifest_path.resolve()),
            "selection_manifest_sha256": digest,
            "selection_subset": self.subset,
            "selection_image_count": len(self.rows),
            "selection_images_by_dataset": dict(sorted(self._counts.items())),
            "selection_proposal_override_used": self.proposal_override_used,
        }

    def write_snapshot(self, output: Path) -> None:
        target = output / "metadata" / "selection_manifest.csv"
        target.write_text(self.manifest_path.read_text(encoding="utf-8-sig"), encoding="utf-8")


class _AssetReader:
    """Read dataset assets from either an extracted directory or a ZIP file."""

    def __init__(self, path: Path):
        self.path = path
        self._archive: zipfile.ZipFile | None = None
        self._members: dict[str, str] = {}

    def __enter__(self) -> "_AssetReader":
        if self.path.is_dir():
            # A UA source is often supplied as an evidence root that also
            # contains prior exports.  Restrict discovery to UA's known
            # top-level directories so a re-export never re-scans (or treats
            # as source) hundreds of thousands of files from an old/new
            # dataset release.
            ua_roots = [
                self.path / "DETRAC-Images",
                self.path / "DETRAC-Train-Annotations-XML",
                self.path / "DETRAC-Test-Annotations-XML",
            ]
            roots = [root for root in ua_roots if root.is_dir()] or [self.path]
            for root in roots:
                for item in root.rglob("*"):
                    if item.is_file():
                        relative = item.relative_to(self.path).as_posix()
                        self._members[relative] = relative
        elif self.path.is_file() and self.path.suffix.lower() == ".zip":
            self._archive = zipfile.ZipFile(self.path)
            self._members = {item.filename.replace("\\", "/"): item.filename for item in self._archive.infolist() if not item.is_dir()}
        else:
            raise FileNotFoundError(f"Dataset source must be an extracted directory or ZIP: {self.path}")
        return self

    def __exit__(self, *_: object) -> None:
        if self._archive:
            self._archive.close()

    def find(self, requested: str) -> str:
        normalized = requested.replace("\\", "/").lstrip("/")
        if normalized in self._members:
            return normalized
        candidates = [name for name in self._members if name.endswith("/" + normalized)]
        if len(candidates) != 1:
            raise FileNotFoundError(f"Could not uniquely locate dataset asset {requested}; matches={len(candidates)}")
        return candidates[0]

    def read(self, requested: str) -> bytes:
        member = self.find(requested)
        if self._archive:
            return self._archive.read(self._members[member])
        return (self.path / self._members[member]).read_bytes()

    def names(self) -> list[str]:
        return list(self._members)


class _AnnotationStore:
    """Disk-backed annotation grouping so full MIO export does not exhaust RAM."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="unified_yolo_")
        self._connection = sqlite3.connect(Path(self._temporary.name) / "annotations.sqlite3")
        self._connection.execute("CREATE TABLE annotations (image_key TEXT NOT NULL, payload TEXT NOT NULL)")
        self._connection.execute("CREATE INDEX annotations_by_image ON annotations(image_key)")

    def __enter__(self) -> "_AnnotationStore":
        return self

    def __exit__(self, *_: object) -> None:
        self._connection.close()
        self._temporary.cleanup()

    def add(self, image_key: str | int, annotation: dict[str, Any]) -> None:
        self._connection.execute(
            "INSERT INTO annotations (image_key, payload) VALUES (?, ?)",
            (str(image_key), json.dumps(annotation, ensure_ascii=False)),
        )

    def keys(self) -> set[str]:
        self._connection.commit()
        return {str(row[0]) for row in self._connection.execute("SELECT DISTINCT image_key FROM annotations")}

    def for_image(self, image_key: str | int) -> list[dict[str, Any]]:
        self._connection.commit()
        return [json.loads(str(row[0])) for row in self._connection.execute("SELECT payload FROM annotations WHERE image_key = ?", (str(image_key),))]


class _DatasetWriter:
    def __init__(self, output: Path, mapping: dict[str, Any]):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output is not empty: {output}. Choose a new dataset version; no files were changed.")
        self.output = output
        self.mapping = mapping
        self.output.mkdir(parents=True, exist_ok=True)
        for split in SPLIT_NAMES.values():
            (output / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        metadata = output / "metadata"
        metadata.mkdir(exist_ok=True)
        self._stack = ExitStack()
        self._image_writer = csv.DictWriter(self._stack.enter_context((metadata / "images.csv").open("w", encoding="utf-8", newline="")), fieldnames=IMAGE_FIELDS, extrasaction="ignore")
        self._annotation_writer = csv.DictWriter(self._stack.enter_context((metadata / "annotations.csv").open("w", encoding="utf-8", newline="")), fieldnames=ANNOTATION_FIELDS, extrasaction="ignore")
        self._rejected_writer = csv.DictWriter(self._stack.enter_context((metadata / "rejected_annotations.csv").open("w", encoding="utf-8", newline="")), fieldnames=REJECTED_FIELDS, extrasaction="ignore")
        self._ignored_writer = csv.DictWriter(self._stack.enter_context((metadata / "ignored_annotations.csv").open("w", encoding="utf-8", newline="")), fieldnames=IGNORED_ANNOTATION_FIELDS, extrasaction="ignore")
        self._negative_writer = csv.DictWriter(self._stack.enter_context((metadata / "negative_samples.csv").open("w", encoding="utf-8", newline="")), fieldnames=NEGATIVE_FIELDS, extrasaction="ignore")
        self._image_writer.writeheader()
        self._annotation_writer.writeheader()
        self._rejected_writer.writeheader()
        self._ignored_writer.writeheader()
        self._negative_writer.writeheader()
        self.counts: Counter[str] = Counter()
        self.dataset_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.images_by_split: Counter[str] = Counter()
        self.boxes_by_split: Counter[str] = Counter()
        self._image_ids: set[str] = set()
        self.sequence_splits: dict[tuple[str, str], str] = {}

    def close(self) -> None:
        self._stack.close()

    def register_sequence(self, dataset: str, sequence_id: str, split: str, split_source: str) -> None:
        key = (dataset, sequence_id)
        prior = self.sequence_splits.get(key)
        if prior and prior != split:
            raise ValueError(f"Sequence leakage: {dataset}/{sequence_id} assigned to {prior} and {split}")
        self.sequence_splits[key] = split

    def reject(self, annotation: dict[str, Any], reason: str, action: str = "EXCLUDE") -> None:
        dataset = str(annotation.get("dataset", ""))
        self.counts["input_annotations"] += 1
        self.counts["rejected_annotations"] += 1
        self.dataset_counts[dataset]["input_annotations"] += 1
        self.dataset_counts[dataset]["rejected_annotations"] += 1
        self._rejected_writer.writerow({
            "dataset": dataset,
            "image_id": annotation.get("image_id", ""),
            "sequence_id": annotation.get("sequence_id", ""),
            "frame_id": annotation.get("frame_id", ""),
            "annotation_id": annotation.get("annotation_id", ""),
            "track_id": annotation.get("track_id", ""),
            "original_class": annotation.get("original_class", ""),
            "source_image": annotation.get("source_image", ""),
            "source_annotation": annotation.get("source_annotation", ""),
            "action": action,
            "reason": reason,
        })

    def add_image(
        self,
        *,
        dataset: str,
        mapping_key: str,
        image_id: str,
        split: str,
        sequence_id: str,
        frame_id: str,
        source_image: str,
        image_bytes: bytes,
        annotations: Iterable[dict[str, Any]],
        exclusions: set[tuple[str, str]] | None = None,
        ignored_regions: Iterable[IgnoredRegion] | None = None,
        mask_ignored_regions: bool = False,
    ) -> None:
        if split not in SPLIT_NAMES.values():
            raise ValueError(f"Unsupported output split {split}")
        if image_id in self._image_ids:
            raise ValueError(f"Duplicate exported image ID {image_id}")
        self._image_ids.add(image_id)
        self.counts["input_images"] += 1
        self.dataset_counts[dataset]["input_images"] += 1
        width, height = _image_size(image_bytes, source_image)
        suffix = Path(source_image).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension for {source_image}")
        ignored_region_list = list(ignored_regions or [])
        class_ignore_regions: list[IgnoredRegion] = []
        lines: list[str] = []
        seen_yolo_lines: set[str] = set()
        clipped_count = 0
        rules = self.mapping.get(mapping_key, {})
        for annotation in annotations:
            row = dict(annotation)
            row.update(dataset=dataset, image_id=image_id, sequence_id=sequence_id, frame_id=frame_id, source_image=source_image)
            self.counts["input_annotations"] += 1
            self.dataset_counts[dataset]["input_annotations"] += 1
            original_class = str(row.get("original_class", "")).strip()
            rule = rules.get(original_class)
            if rule is None or not rule.get("include", False):
                if rule is not None and rule.get("handling") == "IGNORE_REGION":
                    coords = tuple(_as_float(row.get(key)) for key in ("xmin", "ymin", "xmax", "ymax"))
                    if any(value is None for value in coords):
                        self.counts["rejected_annotations"] += 1
                        self.dataset_counts[dataset]["rejected_annotations"] += 1
                        self._rejected_writer.writerow({**row, "action": "EXCLUDE_INVALID_IGNORE_REGION", "reason": "NON_NUMERIC_OR_NONFINITE_BOX"})
                        continue
                    xmin, ymin, xmax, ymax = coords  # type: ignore[misc]
                    if xmin >= xmax or ymin >= ymax:
                        self.counts["rejected_annotations"] += 1
                        self.dataset_counts[dataset]["rejected_annotations"] += 1
                        self._rejected_writer.writerow({**row, "action": "EXCLUDE_INVALID_IGNORE_REGION", "reason": "NON_POSITIVE_SIZE"})
                        continue
                    clipped, _ = clip_bbox_to_image(xmin, ymin, xmax, ymax, width, height)
                    issues = validate_bbox(*clipped, width, height)
                    if issues:
                        self.counts["rejected_annotations"] += 1
                        self.dataset_counts[dataset]["rejected_annotations"] += 1
                        self._rejected_writer.writerow({**row, "action": "EXCLUDE_INVALID_IGNORE_REGION", "reason": "|".join(issues)})
                        continue
                    class_ignore_regions.append(clipped)
                    self.counts["ignored_annotations"] += 1
                    self.dataset_counts[dataset]["ignored_annotations"] += 1
                    self._ignored_writer.writerow({
                        **row, "xmin": clipped[0], "ymin": clipped[1], "xmax": clipped[2], "ymax": clipped[3],
                        "handling": "IGNORE_REGION", "reason": "NON_TARGET_TWO_WHEEL_OR_BICYCLE",
                    })
                    continue
                self.counts["rejected_annotations"] += 1
                self.dataset_counts[dataset]["rejected_annotations"] += 1
                self._rejected_writer.writerow({**row, "action": "EXCLUDE_CLASS", "reason": "CLASS_NOT_INCLUDED"})
                continue
            track_id = str(row.get("track_id", ""))
            if exclusions and (sequence_id, track_id) in exclusions:
                self.counts["rejected_annotations"] += 1
                self.dataset_counts[dataset]["rejected_annotations"] += 1
                self._rejected_writer.writerow({**row, "action": "EXCLUDE_TRACK", "reason": "DATA_LEAD_TRACK_EXCLUSION"})
                continue
            coords = tuple(_as_float(row.get(key)) for key in ("xmin", "ymin", "xmax", "ymax"))
            if any(value is None for value in coords):
                self.counts["rejected_annotations"] += 1
                self.dataset_counts[dataset]["rejected_annotations"] += 1
                self._rejected_writer.writerow({**row, "action": "EXCLUDE_INVALID", "reason": "NON_NUMERIC_OR_NONFINITE_BOX"})
                continue
            xmin, ymin, xmax, ymax = coords  # type: ignore[misc]
            if xmin >= xmax or ymin >= ymax:
                self.counts["rejected_annotations"] += 1
                self.dataset_counts[dataset]["rejected_annotations"] += 1
                self._rejected_writer.writerow({**row, "action": "EXCLUDE_INVALID", "reason": "NON_POSITIVE_SIZE"})
                continue
            clipped, adjustments = clip_bbox_to_image(xmin, ymin, xmax, ymax, width, height)
            issues = validate_bbox(*clipped, width, height)
            if issues:
                self.counts["rejected_annotations"] += 1
                self.dataset_counts[dataset]["rejected_annotations"] += 1
                self._rejected_writer.writerow({**row, "action": "EXCLUDE_INVALID_AFTER_CLIP", "reason": "|".join(issues)})
                continue
            yolo_line = _yolo_line(clipped, width, height)
            if yolo_line in seen_yolo_lines:
                self.counts["rejected_annotations"] += 1
                self.dataset_counts[dataset]["rejected_annotations"] += 1
                self._rejected_writer.writerow({**row, "action": "EXCLUDE_DUPLICATE_BBOX", "reason": "DUPLICATE_YOLO_BBOX_AFTER_NORMALIZATION"})
                continue
            seen_yolo_lines.add(yolo_line)
            lines.append(yolo_line)
            if adjustments:
                clipped_count += 1
                self.counts["clipped_boxes"] += 1
                self.dataset_counts[dataset]["clipped_boxes"] += 1
            self.counts["exported_boxes"] += 1
            self.dataset_counts[dataset]["exported_boxes"] += 1
            self.boxes_by_split[split] += 1
            self._annotation_writer.writerow({
                "image_id": image_id, "dataset": dataset, "split": split, "sequence_id": sequence_id,
                "frame_id": frame_id, "annotation_id": row.get("annotation_id", ""), "track_id": track_id,
                "original_class": original_class, "mapped_class": "vehicle", "class_id": 0,
                "source_image": source_image, "source_annotation": row.get("source_annotation", ""),
                "raw_xmin": xmin, "raw_ymin": ymin, "raw_xmax": xmax, "raw_ymax": ymax,
                "clipped_xmin": clipped[0], "clipped_ymin": clipped[1], "clipped_xmax": clipped[2], "clipped_ymax": clipped[3],
                "clip_applied": bool(adjustments), "clip_adjustments": "|".join(adjustments), "preserve_original_class": True,
            })
        ignored_region_list.extend(class_ignore_regions)
        ignored_region_mask_applied = bool(split == "train" and mask_ignored_regions and ignored_region_list)
        if ignored_region_mask_applied:
            image_bytes = mask_image_bytes(image_bytes, ignored_region_list, source_image)
            self.counts["ignored_region_masked_images"] += 1
            self.counts["ignored_region_count"] += len(ignored_region_list)
            self.dataset_counts[dataset]["ignored_region_masked_images"] += 1
            self.dataset_counts[dataset]["ignored_region_count"] += len(ignored_region_list)
        image_target = self.output / "images" / split / f"{image_id}{suffix}"
        label_target = self.output / "labels" / split / f"{image_id}.txt"
        image_target.write_bytes(image_bytes)
        label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self._image_writer.writerow({
            "image_id": image_id, "dataset": dataset, "split": split, "sequence_id": sequence_id,
            "frame_id": frame_id, "source_image": source_image,
            "exported_image": image_target.relative_to(self.output).as_posix(),
            "exported_label": label_target.relative_to(self.output).as_posix(),
            "width": width, "height": height, "vehicle_box_count": len(lines), "boundary_clipped_box_count": clipped_count,
            "ignored_region_mask_applied": ignored_region_mask_applied,
            "ignored_region_count": len(ignored_region_list),
        })
        if not lines:
            self._negative_writer.writerow({
                "image_id": image_id, "dataset": dataset, "split": split,
                "sequence_id": sequence_id, "frame_id": frame_id,
                "exported_image": image_target.relative_to(self.output).as_posix(),
                "exported_label": label_target.relative_to(self.output).as_posix(),
                "negative_reason": "NO_RETAINED_POSITIVE_ANNOTATIONS_IGNORE_REGIONS_MASKED" if class_ignore_regions else "NO_RETAINED_VEHICLE_ANNOTATIONS",
            })
        self.counts["exported_images"] += 1
        self.dataset_counts[dataset]["exported_images"] += 1
        self.images_by_split[split] += 1

    def write_summary(
        self,
        sources: dict[str, str],
        split_source: Path,
        mapping_source: Path,
        selection_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.close()
        if self.counts["input_annotations"] != self.counts["exported_boxes"] + self.counts["rejected_annotations"] + self.counts["ignored_annotations"]:
            raise AssertionError("Annotation count reconciliation failed before summary")
        metadata = self.output / "metadata"
        with (metadata / "sequence_splits.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SEQUENCE_FIELDS)
            writer.writeheader()
            for (dataset, sequence_id), split in sorted(self.sequence_splits.items()):
                writer.writerow({"dataset": dataset, "sequence_id": sequence_id, "split": split, "split_source": str(split_source)})
        # Data files are complete at this point.  Recount the materialized
        # output before creating any derived metadata; exporter counters are
        # never used as evidence of the final release contents.
        recount = audit_yolo_dataset(self.output, require_summary=False)
        actual_images = int(recount["image_count_actual"])
        actual_boxes = int(recount["box_count_actual"])
        if self.counts["exported_images"] != actual_images:
            raise AssertionError(
                f"Written image count differs from exporter state: writer={self.counts['exported_images']}, actual={actual_images}"
            )
        if self.counts["exported_boxes"] != actual_boxes:
            raise AssertionError(
                f"Written box count differs from exporter state: writer={self.counts['exported_boxes']}, actual={actual_boxes}"
            )
        counts = Counter(self.counts)
        counts["input_images"] = actual_images
        counts["exported_images"] = actual_images
        counts["exported_boxes"] = actual_boxes
        counts["rejected_annotations"] = self.counts["rejected_annotations"]
        counts["ignored_annotations"] = self.counts["ignored_annotations"]
        dataset_counts = {name: dict(sorted(values.items())) for name, values in sorted(self.dataset_counts.items())}
        for name, actual in recount["output_counts_by_dataset"].items():
            dataset_counts.setdefault(name, {}).update(actual)
        summary = {
            "format": "unified-yolo-v2", "target_class": "vehicle", "class_id": 0,
            "preserve_original_class": True, "sources": sources,
            "split_source": str(split_source.resolve()), "mapping_source": str(mapping_source.resolve()),
            "summary_generated_from": "final_output_recount",
            "counts": dict(sorted(counts.items())),
            "images_by_split": {split: count for split, count in recount["images_by_split"].items() if count},
            "boxes_by_split": {split: count for split, count in recount["boxes_by_split"].items() if count},
            "counts_by_dataset": dataset_counts,
            "sequence_count": len(self.sequence_splits),
        }
        if selection_metadata:
            summary["selection"] = selection_metadata
        # Ultralytics resolves `path` relative to the process working directory,
        # not reliably relative to this YAML file.  A resolved path makes each
        # exported release trainable without requiring callers to `cd` first.
        dataset_yaml = {"path": self.output.resolve().as_posix(), "train": "images/train", "val": "images/val", "test": "images/cross_test", "names": {0: "vehicle"}}
        (self.output / "data.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
        (metadata / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        qc_report = audit_yolo_dataset(self.output, require_summary=True)
        write_report(qc_report, metadata / "qc_report.json")
        if qc_report["status"] != "PASS":
            raise RuntimeError(f"Export QC failed: {qc_report['errors']}")
        summary["validated_export"] = True
        summary["qc_report"] = "metadata/qc_report.json"
        (metadata / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary


def _aau_split(split_map: dict[tuple[str, str], str], sequence: str) -> str:
    split = split_map.get(("AAU RainSnow", sequence))
    if not split:
        raise ValueError(f"AAU sequence is missing a split assignment: {sequence}")
    return split


def _ua_split(split_map: dict[tuple[str, str], str], sequence: str) -> str:
    split = split_map.get(("UA-DETRAC Original", sequence))
    if not split:
        raise ValueError(f"UA-DETRAC sequence is missing a split assignment: {sequence}")
    return split


def _export_mio(writer: _DatasetWriter, mio_path: Path, selection: _Selection | None = None) -> None:
    if not mio_path.is_file():
        raise FileNotFoundError(f"MIO-TCD Localization TAR not found: {mio_path}")
    with tarfile.open(mio_path, "r:*") as archive:
        images: dict[str, tarfile.TarInfo] = {}
        gt_info: tarfile.TarInfo | None = None
        for member in archive:
            name = member.name.replace("\\", "/")
            lower = name.lower()
            if member.isfile() and Path(name).suffix.lower() in IMAGE_SUFFIXES and "/train/" in lower:
                image_id = Path(name).stem
                if image_id in images:
                    raise ValueError(f"MIO has duplicate train image ID {image_id}")
                images[image_id] = member
            elif member.isfile() and lower.endswith("gt_train.csv"):
                gt_info = member
        if not images or gt_info is None:
            raise RuntimeError("MIO archive must contain train images and gt_train.csv")
        with _AnnotationStore() as rows:
            handle = archive.extractfile(gt_info)
            if handle is None:
                raise RuntimeError("Could not read MIO gt_train.csv")
            for line_number, row in enumerate(csv.reader(io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")), start=1):
                if len(row) < 6:
                    writer.reject({"dataset": "MIO-TCD Localization", "annotation_id": line_number, "source_annotation": gt_info.name}, "MALFORMED_ROW")
                    continue
                image_key = row[0].strip()
                rows.add(image_key, {
                    "annotation_id": line_number, "original_class": row[1].strip(), "xmin": row[2], "ymin": row[3], "xmax": row[4], "ymax": row[5],
                    "source_annotation": gt_info.name,
                })
            sequence = "MIO_NO_SEQUENCE_TRAIN_ONLY"
            writer.register_sequence("MIO-TCD Localization", sequence, "train", "TRAIN_ONLY_NO_SEQUENCE_METADATA")
            for missing_id in sorted(rows.keys() - set(images)):
                for annotation in rows.for_image(missing_id):
                    source = f"MIO-TCD-Localization/train/{missing_id}.jpg"
                    if selection is None or selection.includes("MIO-TCD Localization", source):
                        writer.reject({"dataset": "MIO-TCD Localization", "image_id": f"MIO_{_safe_id(missing_id)}", "sequence_id": sequence, **annotation}, "MISSING_SOURCE_IMAGE")
            for source_id, member in sorted(images.items()):
                source = member.name.replace("\\", "/")
                if selection is not None and not selection.includes("MIO-TCD Localization", source):
                    continue
                binary = archive.extractfile(member)
                if binary is None:
                    raise RuntimeError(f"Could not read MIO image {source}")
                writer.add_image(dataset="MIO-TCD Localization", mapping_key="mio_tcd", image_id=f"MIO_{_safe_id(source_id)}", split="train", sequence_id=sequence, frame_id=source_id, source_image=source, image_bytes=binary.read(), annotations=rows.for_image(source_id), mask_ignored_regions=True)
                if selection is not None:
                    selection.consume("MIO-TCD Localization", source)


def _export_aau(writer: _DatasetWriter, aau_path: Path, split_map: dict[tuple[str, str], str], selection: _Selection | None = None) -> None:
    with _AssetReader(aau_path) as source:
        json_name = next((name for name in source.names() if name.lower().endswith("aaurainsnow-rgb.json")), None)
        if not json_name:
            raise RuntimeError("AAU source must contain aauRainSnow-rgb.json")
        payload = json.loads(source.read(json_name).decode("utf-8-sig"))
        categories = {int(row["id"]): str(row["name"]) for row in payload.get("categories", [])}
        images = {int(row["id"]): row for row in payload.get("images", [])}
        with _AnnotationStore() as rows:
            for annotation in payload.get("annotations", []):
                image_key = int(annotation.get("image_id", -1))
                bbox = annotation.get("bbox") or []
                coords: dict[str, Any] = {}
                if len(bbox) == 4:
                    left, top, box_width, box_height = (_as_float(value) for value in bbox)
                    coords = {"xmin": bbox[0], "ymin": bbox[1], "xmax": left + box_width if left is not None and box_width is not None else "", "ymax": top + box_height if top is not None and box_height is not None else ""}
                rows.add(image_key, {"annotation_id": annotation.get("id", ""), "original_class": categories.get(int(annotation.get("category_id", -1)), f"UNKNOWN_CATEGORY_{annotation.get('category_id', '')}"), "source_annotation": json_name, **coords})
            for missing_id in sorted(rows.keys() - {str(key) for key in images}):
                for annotation in rows.for_image(missing_id):
                    writer.reject({"dataset": "AAU RainSnow", "image_id": f"AAU_{missing_id}", **annotation}, "MISSING_IMAGE_METADATA")
            for image_key, meta in sorted(images.items()):
                file_name = str(meta.get("file_name", "")).replace("\\", "/")
                sequence = Path(file_name).parent.name
                if not file_name or not sequence:
                    raise ValueError(f"AAU image has no usable sequence path: image_id={image_key}")
                if selection is not None and not selection.includes("AAU RainSnow", file_name):
                    continue
                split = _aau_split(split_map, sequence)
                writer.register_sequence("AAU RainSnow", sequence, split, "split_proposal.csv")
                writer.add_image(dataset="AAU RainSnow", mapping_key="aau_rainsnow", image_id=f"AAU_{image_key}", split=split, sequence_id=sequence, frame_id=str(image_key), source_image=file_name, image_bytes=source.read(file_name), annotations=rows.for_image(image_key), mask_ignored_regions=True)
                if selection is not None:
                    selection.consume("AAU RainSnow", file_name)


def _export_ua(
    writer: _DatasetWriter,
    ua_path: Path,
    split_map: dict[tuple[str, str], str],
    mapping: dict[str, Any],
    selection: _Selection | None = None,
    mask_ignored_regions: bool = True,
) -> None:
    exclusions = {
        (str(row.get("sequence_id", "")), str(row.get("track_id", "")))
        for row in mapping.get("ua_detrac", {}).get("others", {}).get("track_exclusions", [])
        if row.get("action") == "EXCLUDE_NON_VEHICLE_TRACK"
    }
    with _AssetReader(ua_path) as source:
        by_frame: dict[tuple[str, int], str] = {}
        for name in source.names():
            if Path(name).suffix.lower() not in IMAGE_SUFFIXES or "/detrac-images/" not in name.lower():
                continue
            try:
                frame_id = int(Path(name).stem.lower().replace("img", ""))
            except ValueError:
                continue
            by_frame[(safe_sequence(name), frame_id)] = name
        xml_names = sorted(name for name in source.names() if name.lower().endswith(".xml") and "annotations-xml" in name.lower())
        if not xml_names:
            raise RuntimeError("UA source contains no annotation XML files")
        for xml_name in xml_names:
            root = ET.fromstring(source.read(xml_name))
            sequence = root.attrib.get("name") or Path(xml_name).stem
            split = _ua_split(split_map, sequence)
            ignored_regions = parse_ua_ignored_regions(root)
            for frame in root.findall("frame"):
                frame_id = int(frame.attrib.get("num", "0"))
                source_image = by_frame.get((sequence, frame_id))
                annotations: list[dict[str, Any]] = []
                for target in frame.findall("./target_list/target"):
                    box = target.find("box")
                    attribute = target.find("attribute")
                    base = {"annotation_id": f"{frame_id}:{target.attrib.get('id', '')}", "track_id": str(target.attrib.get("id", "")), "original_class": str(attribute.attrib.get("vehicle_type", "MISSING")) if attribute is not None else "MISSING", "source_annotation": xml_name}
                    if box is None:
                        annotations.append(base)
                    else:
                        left, top, box_width, box_height = (box.attrib.get(key, "") for key in ("left", "top", "width", "height"))
                        left_value, top_value, width_value, height_value = (_as_float(value) for value in (left, top, box_width, box_height))
                        annotations.append({**base, "xmin": left, "ymin": top, "xmax": left_value + width_value if left_value is not None and width_value is not None else "", "ymax": top_value + height_value if top_value is not None and height_value is not None else ""})
                if source_image is None:
                    for annotation in annotations:
                        writer.reject({"dataset": "UA-DETRAC Original", "image_id": f"UA_{sequence}_{frame_id:05d}", "sequence_id": sequence, "frame_id": frame_id, **annotation}, "MISSING_SOURCE_IMAGE")
                    continue
                if selection is not None and not selection.includes("UA-DETRAC Original", source_image):
                    continue
                writer.register_sequence("UA-DETRAC Original", sequence, split, "split_proposal.csv")
                writer.add_image(
                    dataset="UA-DETRAC Original", mapping_key="ua_detrac",
                    image_id=f"UA_{sequence}_{frame_id:05d}", split=split, sequence_id=sequence,
                    frame_id=str(frame_id), source_image=source_image, image_bytes=source.read(source_image),
                    annotations=annotations, exclusions=exclusions, ignored_regions=ignored_regions,
                    mask_ignored_regions=mask_ignored_regions,
                )
                if selection is not None:
                    selection.consume("UA-DETRAC Original", source_image)


def export_unified_yolo(
    mio_path: Path | None,
    aau_path: Path | None,
    ua_path: Path | None,
    output: Path,
    split_path: Path,
    mapping_path: Path,
    selection_manifest: Path | None = None,
    selection_subset: str | None = None,
    allow_proposal_selection: bool = False,
    selection_datasets: set[str] | None = None,
    mask_ua_ignored_regions: bool = True,
) -> dict[str, Any]:
    """Run the complete export and return the reconciliation summary."""
    mapping = load_yaml(mapping_path)
    if not mapping.get("preserve_original_class"):
        raise ValueError("vehicle_class_mapping.yaml must set preserve_original_class=true")
    split_map = _split_map(split_path)
    if (selection_manifest is None) != (selection_subset is None):
        raise ValueError("selection_manifest and selection_subset must be supplied together")
    if not any((mio_path, aau_path, ua_path)):
        raise ValueError("At least one source archive must be supplied")
    selection = (
        _Selection.load(selection_manifest, selection_subset, allow_proposal_selection, selection_datasets)
        if selection_manifest is not None and selection_subset is not None
        else None
    )
    writer = _DatasetWriter(output, mapping)
    try:
        if mio_path is not None:
            _export_mio(writer, mio_path, selection)
        if aau_path is not None:
            _export_aau(writer, aau_path, split_map, selection)
        if ua_path is not None:
            _export_ua(writer, ua_path, split_map, mapping, selection, mask_ua_ignored_regions)
        if selection is not None:
            selection.assert_complete(writer)
            selection.write_snapshot(output)
        return writer.write_summary(
            {
                name: str(path.resolve())
                for name, path in {
                    "mio_tcd": mio_path,
                    "aau_rainsnow": aau_path,
                    "ua_detrac": ua_path,
                }.items()
                if path is not None
            },
            split_path,
            mapping_path,
            selection.metadata() if selection is not None else None,
        )
    except Exception:
        writer.close()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mio-path", type=Path, default=DEFAULT_MIO)
    parser.add_argument("--aau-path", type=Path, default=DEFAULT_AAU)
    parser.add_argument("--ua-path", type=Path, default=DEFAULT_UA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--mapping-path", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--selection-manifest", type=Path, help="CSV manifest used to select an exact subset")
    parser.add_argument("--selection-subset", help="Value of target_subset to export from --selection-manifest")
    parser.add_argument("--allow-proposal-selection", action="store_true", help="Explicitly allow rows whose selected column is not TRUE")
    parser.add_argument("--no-mask-ua-ignored-regions", action="store_true", help="Diagnostic only: do not black-mask UA ignored regions in training images")
    args = parser.parse_args()
    summary = export_unified_yolo(
        args.mio_path.resolve(), args.aau_path.resolve(), args.ua_path.resolve(), args.output.resolve(),
        args.split_path.resolve(), args.mapping_path.resolve(), args.selection_manifest.resolve() if args.selection_manifest else None,
        args.selection_subset, args.allow_proposal_selection, mask_ua_ignored_regions=not args.no_mask_ua_ignored_regions,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
