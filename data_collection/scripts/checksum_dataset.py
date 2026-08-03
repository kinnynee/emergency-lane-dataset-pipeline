"""Create or verify a deterministic checksum manifest for immutable raw data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    REPO_ROOT
    / "data_collection"
    / "storage_placeholders"
    / "online_data"
    / "raw"
    / "ua_detrac_orig"
    / "ua-detrac-orig.zip"
)
DEFAULT_BASELINE = (
    REPO_ROOT
    / "data_collection"
    / "reports"
    / "external_eda"
    / "baseline_checksums.csv"
)
FIELDS = ["relative_path", "size_bytes", "algorithm", "checksum"]


def _files(dataset: Path) -> tuple[Path, list[Path]]:
    if dataset.is_file():
        return dataset.parent, [dataset]
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {dataset}")
    files = sorted(
        (path for path in dataset.rglob("*") if path.is_file() and not path.is_symlink()),
        key=lambda path: path.relative_to(dataset).as_posix().casefold(),
    )
    if not files:
        raise ValueError(f"Dataset directory contains no files: {dataset}")
    return dataset, files


def _digest(path: Path, algorithm: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scan(dataset: Path, algorithm: str) -> list[dict[str, str | int]]:
    root, files = _files(dataset)
    rows: list[dict[str, str | int]] = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(root).as_posix()
        print(f"[{index}/{len(files)}] {algorithm.upper()} {relative}", flush=True)
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "algorithm": algorithm,
                "checksum": _digest(path, algorithm),
            }
        )
    return rows


def write_baseline(path: Path, rows: Iterable[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def read_baseline(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Checksum baseline not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or list(rows[0]) != FIELDS:
        raise ValueError(f"Invalid checksum baseline schema: {path}")
    return rows


def compare(
    expected_rows: Iterable[dict[str, str]],
    actual_rows: Iterable[dict[str, str | int]],
) -> dict[str, object]:
    expected = {row["relative_path"]: row for row in expected_rows}
    actual = {str(row["relative_path"]): row for row in actual_rows}
    missing = sorted(set(expected) - set(actual))
    added = sorted(set(actual) - set(expected))
    changed = sorted(
        path
        for path in set(expected) & set(actual)
        if expected[path]["checksum"].lower() != str(actual[path]["checksum"]).lower()
        or int(expected[path]["size_bytes"]) != int(actual[path]["size_bytes"])
        or expected[path]["algorithm"].lower() != str(actual[path]["algorithm"]).lower()
    )
    return {
        "status": "PASS" if not (missing or added or changed) else "FAIL",
        "expected_files": len(expected),
        "actual_files": len(actual),
        "missing": missing,
        "added": added,
        "changed": changed,
    }


def set_read_only(dataset: Path) -> int:
    _, files = _files(dataset)
    for path in files:
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--algorithm", choices=("sha256", "md5"), default="sha256")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--set-read-only", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    baseline = args.baseline.resolve()
    if args.verify:
        expected = read_baseline(baseline)
        algorithms = {row["algorithm"].lower() for row in expected}
        if len(algorithms) != 1:
            raise ValueError("Baseline must use exactly one checksum algorithm")
        algorithm = algorithms.pop()
        actual = scan(dataset, algorithm)
        result = compare(expected, actual)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "PASS" else 2

    rows = scan(dataset, args.algorithm)
    write_baseline(baseline, rows)
    if args.set_read_only:
        protected = set_read_only(dataset)
        print(f"Set read-only: {protected} file(s)")
    total_bytes = sum(int(row["size_bytes"]) for row in rows)
    print(
        json.dumps(
            {
                "status": "BASELINE_CREATED",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": len(rows),
                "total_bytes": total_bytes,
                "algorithm": args.algorithm,
                "baseline": str(baseline),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
