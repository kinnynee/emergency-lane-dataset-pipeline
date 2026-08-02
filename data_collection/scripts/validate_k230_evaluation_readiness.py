"""Validate real K230 evaluation sessions without inventing missing mAP values."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from external_eda_common import ROOT, read_csv, write_csv


DEFAULT_MANIFEST = ROOT / "planning" / "k230_evaluation_sessions.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "external_eda" / "evaluation_slice_readiness.csv"
REQUIRED_SLICES = ("DAY", "NIGHT", "BACKLIT", "RAIN")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
OUTPUT_FIELDS = [
    "metric",
    "slice",
    "source",
    "status",
    "current_evaluable_sequences",
    "current_map",
    "required_action",
    "leakage_rule",
]


def _bool(value: Any) -> bool:
    return str(value).strip().upper() in {"TRUE", "1", "YES"}


def _resolve(root: Path, value: str) -> Path | None:
    text = value.strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else (root / path).resolve()


def _pair_status(images_dir: Path | None, labels_dir: Path | None) -> tuple[bool, str, int]:
    if images_dir is None or labels_dir is None:
        return False, "MISSING_IMAGE_OR_LABEL_PATH", 0
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return False, "IMAGE_OR_LABEL_DIRECTORY_NOT_FOUND", 0
    image_stems = {
        path.stem for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    label_stems = {path.stem for path in labels_dir.glob("*.txt") if path.is_file()}
    if not image_stems:
        return False, "NO_IMAGES", 0
    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems
    if missing_labels:
        return False, f"MISSING_LABELS={len(missing_labels)}", len(image_stems)
    if orphan_labels:
        return False, f"ORPHAN_LABELS={len(orphan_labels)}", len(image_stems)
    return True, "PAIRS_OK", len(image_stems)


def validate_sessions(manifest: Path, project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qualifying: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostics: list[dict[str, Any]] = []
    seen_session_slice: set[tuple[str, str]] = set()
    for row in read_csv(manifest):
        session_id = str(row.get("session_id", "")).strip()
        slice_name = str(row.get("slice", "")).strip().upper()
        source = str(row.get("source", "")).strip()
        split = str(row.get("split", "")).strip()
        ground_truth_status = str(row.get("ground_truth_status", "")).strip().upper()
        reason: list[str] = []
        key = (session_id, slice_name)
        if not session_id:
            reason.append("MISSING_SESSION_ID")
        if slice_name not in REQUIRED_SLICES:
            reason.append("INVALID_SLICE")
        if key in seen_session_slice:
            reason.append("DUPLICATE_SESSION_SLICE")
        seen_session_slice.add(key)
        if source != "K230_SELF_RECORDED":
            reason.append("NOT_K230_SELF_RECORDED")
        if split != "MAIN_K230_TEST":
            reason.append("NOT_MAIN_K230_TEST")
        if not _bool(row.get("locked", "")):
            reason.append("SESSION_NOT_LOCKED")
        if ground_truth_status != "APPROVED":
            reason.append("GROUND_TRUTH_NOT_APPROVED")
        images_dir = _resolve(project_root, str(row.get("images_dir", "")))
        labels_dir = _resolve(project_root, str(row.get("labels_dir", "")))
        pairs_ok, pair_reason, image_count = _pair_status(images_dir, labels_dir)
        if not pairs_ok:
            reason.append(pair_reason)
        prediction_path = _resolve(project_root, str(row.get("predictions_path", "")))
        predictions_ready = bool(prediction_path and prediction_path.is_file())
        diagnostic = {
            "session_id": session_id,
            "slice": slice_name,
            "qualified": not reason,
            "image_count": image_count,
            "predictions_ready": predictions_ready,
            "reason": "|".join(reason) if reason else "READY",
        }
        diagnostics.append(diagnostic)
        if not reason:
            qualifying[slice_name].append(diagnostic)

    rows: list[dict[str, Any]] = []
    for slice_name in REQUIRED_SLICES:
        sessions = qualifying.get(slice_name, [])
        ready_predictions = sum(bool(row["predictions_ready"]) for row in sessions)
        if not sessions:
            status = "BLOCKED_MISSING_DATA" if slice_name == "BACKLIT" else "PENDING_COLLECTION"
            action = (
                "Record, label, review and lock a dedicated elevated-camera BACKLIT K230 session."
                if slice_name == "BACKLIT"
                else f"Collect, label, review and lock at least one independent K230 {slice_name} session."
            )
        elif ready_predictions < len(sessions):
            status = "READY_FOR_INFERENCE"
            action = "Run the frozen model on every qualified session and register prediction files."
        else:
            status = "READY_FOR_MAP_CALCULATION"
            action = "Calculate and record mAP from approved ground truth and frozen-model predictions."
        rows.append(
            {
                "metric": "mAP",
                "slice": slice_name,
                "source": "K230_SELF_RECORDED",
                "status": status,
                "current_evaluable_sequences": len(sessions),
                "current_map": "NOT_AVAILABLE",
                "required_action": action,
                "leakage_rule": "Keep entire locked session in MAIN_K230_TEST and out of train/validation.",
            }
        )
    return rows, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"K230 session manifest not found: {manifest}")
    rows, diagnostics = validate_sessions(manifest, ROOT.parent)
    write_csv(args.output.resolve(), rows, OUTPUT_FIELDS)
    if args.diagnostics:
        write_csv(
            args.diagnostics.resolve(),
            diagnostics,
            ["session_id", "slice", "qualified", "image_count", "predictions_ready", "reason"],
        )
    for row in rows:
        print(
            f"{row['slice']}: {row['status']} sequences={row['current_evaluable_sequences']} "
            f"mAP={row['current_map']}"
        )
    if args.require_ready and any(row["status"] != "READY_FOR_MAP_CALCULATION" for row in rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
