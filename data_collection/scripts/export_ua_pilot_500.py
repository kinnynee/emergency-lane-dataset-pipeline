"""Create an atomic 500-image UA-DETRAC-only training pilot from local evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

from export_unified_yolo import DEFAULT_MAPPING, DEFAULT_SPLITS, export_unified_yolo
from validate_ua_pilot_500 import DEFAULT_CONFIG, validate_ua_pilot_500


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reports" / "external_eda" / "selected_data_manifest.csv"
DEFAULT_OUTPUT = ROOT / "dataset_output" / "dataset-v0.1" / "PILOT_500_UA_ONLY"
UA_DATASET = "UA-DETRAC Original"


def export_ua_pilot_500(
    ua_path: Path,
    output: Path,
    selection_manifest: Path,
    split_path: Path,
    mapping_path: Path,
    config_path: Path,
    allow_proposal_selection: bool,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Release output already exists: {output}. Choose a new version; no files were changed.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    try:
        export_summary = export_unified_yolo(
            None, None, ua_path.resolve(), staging.resolve(), split_path.resolve(), mapping_path.resolve(),
            selection_manifest.resolve(), "DATASET_V1_1500", allow_proposal_selection, {UA_DATASET},
        )
        validation = validate_ua_pilot_500(staging, config_path.resolve())
        report_path = staging / "metadata" / "pilot_validation_report.json"
        report_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if validation["status"] != "PASS":
            raise RuntimeError("UA pilot validation failed; staging output was not published")
        staging.replace(output)
        return {
            "status": "PASS",
            "output": str(output.resolve()),
            "export_summary": export_summary,
            "validation_report": str((output / "metadata" / "pilot_validation_report.json").resolve()),
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ua-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--mapping-path", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-proposal-selection", action="store_true")
    args = parser.parse_args()
    try:
        result = export_ua_pilot_500(
            args.ua_path, args.output, args.selection_manifest, args.split_path,
            args.mapping_path, args.config, args.allow_proposal_selection,
        )
    except (OSError, RuntimeError, ValueError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
