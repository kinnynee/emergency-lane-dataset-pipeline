"""Atomically export and validate the provisional dataset-v0.1 / PILOT_500.

The command only publishes an output after both the generic YOLO checks and
the exact 500-image release contract pass.  It never overwrites an existing
release directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

from export_unified_yolo import (
    DEFAULT_AAU,
    DEFAULT_MAPPING,
    DEFAULT_MIO,
    DEFAULT_SPLITS,
    DEFAULT_UA,
    export_unified_yolo,
)
from validate_pilot_500 import DEFAULT_CONFIG, validate_pilot_500


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reports" / "external_eda" / "selected_data_manifest.csv"
DEFAULT_OUTPUT = ROOT / "dataset_output" / "dataset-v0.1" / "PILOT_500"


def export_pilot_500(
    mio_path: Path,
    aau_path: Path,
    ua_path: Path,
    output: Path,
    selection_manifest: Path,
    split_path: Path,
    mapping_path: Path,
    config_path: Path,
    allow_proposal_selection: bool,
) -> dict[str, object]:
    """Create a validated release in a staging directory, then publish it."""
    if output.exists():
        raise FileExistsError(f"Release output already exists: {output}. Choose a new version; no files were changed.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    try:
        export_summary = export_unified_yolo(
            mio_path.resolve(), aau_path.resolve(), ua_path.resolve(), staging.resolve(),
            split_path.resolve(), mapping_path.resolve(), selection_manifest.resolve(), "PILOT_500",
            allow_proposal_selection,
        )
        validation = validate_pilot_500(staging, config_path.resolve())
        report_path = staging / "metadata" / "pilot_validation_report.json"
        report_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if validation["status"] != "PASS":
            raise RuntimeError("PILOT_500 validation failed; staging output was not published")
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
    parser.add_argument("--mio-path", type=Path, default=DEFAULT_MIO)
    parser.add_argument("--aau-path", type=Path, default=DEFAULT_AAU)
    parser.add_argument("--ua-path", type=Path, default=DEFAULT_UA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--mapping-path", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-proposal-selection",
        action="store_true",
        help="Required while the committed PILOT_500 selection rows are still marked selected=FALSE",
    )
    args = parser.parse_args()
    try:
        result = export_pilot_500(
            args.mio_path, args.aau_path, args.ua_path, args.output, args.selection_manifest,
            args.split_path, args.mapping_path, args.config, args.allow_proposal_selection,
        )
    except (OSError, RuntimeError, ValueError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
