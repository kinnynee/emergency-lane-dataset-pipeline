"""Create and verify portable SHA-256 manifests for a file or directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator


MANIFEST_FIELDS = ("relative_path", "size_bytes", "sha256")
VERIFY_REPORT_FIELDS = (
    "relative_path",
    "status",
    "expected_size_bytes",
    "actual_size_bytes",
    "expected_sha256",
    "actual_sha256",
)
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root_for(target: Path) -> Path:
    target = target.resolve()
    if not target.exists():
        raise FileNotFoundError(f"Input path does not exist: {target}")
    return target if target.is_dir() else target.parent


def _files(target: Path, excluded: set[Path] | None = None) -> Iterable[Path]:
    target = target.resolve()
    excluded = {path.resolve() for path in excluded or set()}
    if target.is_file():
        if target not in excluded:
            yield target
        return
    if not target.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {target}")
    for directory, directory_names, file_names in os.walk(target):
        directory_names.sort()
        for file_name in sorted(file_names):
            path = Path(directory) / file_name
            if path.resolve() not in excluded:
                yield path


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def build_manifest(target: Path, manifest: Path) -> dict[str, int | str]:
    """Write ``relative_path,size_bytes,sha256`` records for every input file.

    A manifest saved inside the input directory is deliberately excluded, so a
    rebuild is stable and never attempts to checksum the manifest itself.
    """
    target, manifest = target.resolve(), manifest.resolve()
    if target == manifest:
        raise ValueError("The manifest path must not be the same as the input path")
    root = _root_for(target)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for path in _files(target, {manifest}):
            writer.writerow({
                "relative_path": _relative(path, root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
            file_count += 1
    return {
        "status": "CREATED",
        "input": str(target),
        "manifest": str(manifest),
        "file_count": file_count,
    }


def _safe_relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    candidate = Path(*path.parts)
    if not value or "\\" in value or path.is_absolute() or candidate.is_absolute() or candidate.drive or ".." in path.parts:
        raise ValueError(f"Unsafe relative_path in manifest: {value!r}")
    return candidate


def _manifest_records(manifest: Path, seen: set[str] | None = None) -> Iterator[dict[str, str]]:
    seen = seen if seen is not None else set()
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(MANIFEST_FIELDS):
            raise ValueError(f"Manifest must have columns: {', '.join(MANIFEST_FIELDS)}")
        for record in reader:
            relative = record.get("relative_path") or ""
            _safe_relative_path(relative)
            if relative in seen:
                raise ValueError(f"Duplicate manifest path: {relative}")
            seen.add(relative)
            size = record.get("size_bytes") or ""
            if not size.isdigit():
                raise ValueError(f"Invalid size_bytes for {relative}")
            digest = record.get("sha256") or ""
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
                raise ValueError(f"Invalid SHA-256 for {relative}")
            yield record


def verify_manifest(
    target: Path,
    manifest: Path,
    report: Path | None = None,
    check_extra: bool = False,
    collect_results: bool = True,
) -> tuple[dict[str, int | str], list[dict[str, str]]]:
    """Compare input files against a manifest using bounded working memory."""
    target, manifest = target.resolve(), manifest.resolve()
    root = _root_for(target)
    results: list[dict[str, str]] = []
    expected_paths: set[str] = set()
    counts = {"MATCH": 0, "MISSING": 0, "MODIFIED": 0, "UNEXPECTED": 0}
    report_handle = None
    report_writer = None
    if report:
        report = report.resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report_handle = report.open("w", encoding="utf-8", newline="")
        report_writer = csv.DictWriter(report_handle, fieldnames=VERIFY_REPORT_FIELDS)
        report_writer.writeheader()

    def record_result(result: dict[str, str]) -> None:
        counts[result["status"]] += 1
        if collect_results:
            results.append(result)
        if report_writer:
            report_writer.writerow(result)

    try:
        for record in _manifest_records(manifest, expected_paths):
            relative = record["relative_path"]
            actual_path = root / _safe_relative_path(relative)
            expected_size, expected_hash = record["size_bytes"], record["sha256"]
            if not actual_path.is_file():
                record_result({
                    "relative_path": relative, "status": "MISSING", "expected_size_bytes": expected_size,
                    "actual_size_bytes": "", "expected_sha256": expected_hash, "actual_sha256": "",
                })
                continue
            actual_size = str(actual_path.stat().st_size)
            actual_hash = sha256_file(actual_path)
            status = "MATCH" if actual_size == expected_size and actual_hash == expected_hash else "MODIFIED"
            record_result({
                "relative_path": relative, "status": status, "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size, "expected_sha256": expected_hash, "actual_sha256": actual_hash,
            })

        if check_extra:
            excluded = {manifest}
            if report:
                excluded.add(report)
            for path in _files(target, excluded):
                relative = _relative(path, root)
                if relative not in expected_paths:
                    record_result({
                        "relative_path": relative, "status": "UNEXPECTED", "expected_size_bytes": "",
                        "actual_size_bytes": str(path.stat().st_size), "expected_sha256": "", "actual_sha256": sha256_file(path),
                    })
    finally:
        if report_handle:
            report_handle.close()

    if collect_results:
        results.sort(key=lambda result: result["relative_path"])
    failed_count = counts["MISSING"] + counts["MODIFIED"] + counts["UNEXPECTED"]
    summary: dict[str, int | str] = {
        "status": "PASS" if failed_count == 0 else "FAIL",
        "input": str(target),
        "manifest": str(manifest),
        "checked_files": sum(counts.values()),
        "matched_files": counts["MATCH"],
        "missing_files": counts["MISSING"],
        "modified_files": counts["MODIFIED"],
        "unexpected_files": counts["UNEXPECTED"],
    }
    return summary, results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a SHA-256 manifest")
    create.add_argument("--input", type=Path, required=True, help="File or directory to checksum")
    create.add_argument("--manifest", type=Path, required=True, help="CSV manifest to create")

    verify = subparsers.add_parser("verify", help="Verify files against a SHA-256 manifest")
    verify.add_argument("--input", type=Path, required=True, help="Original file or directory")
    verify.add_argument("--manifest", type=Path, required=True, help="CSV manifest to verify")
    verify.add_argument("--report", type=Path, help="Optional CSV verification report")
    verify.add_argument("--check-extra", action="store_true", help="Also fail when unlisted files are found")

    args = parser.parse_args()
    try:
        if args.command == "create":
            print(json.dumps(build_manifest(args.input, args.manifest), ensure_ascii=False, indent=2))
            return 0
        summary, _ = verify_manifest(args.input, args.manifest, args.report, args.check_extra, collect_results=False)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["status"] == "PASS" else 1
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
