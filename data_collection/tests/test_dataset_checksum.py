from __future__ import annotations

import csv
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from dataset_checksum import build_manifest, sha256_file, verify_manifest


def test_checksum_manifest_detects_a_modified_file(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    nested = dataset / "images"
    nested.mkdir(parents=True)
    source = nested / "frame.txt"
    source.write_text("original", encoding="utf-8")
    manifest = tmp_path / "dataset_checksums.csv"

    summary = build_manifest(dataset, manifest)
    assert summary["file_count"] == 1
    assert len(sha256_file(source)) == 64

    passed, rows = verify_manifest(dataset, manifest)
    assert passed["status"] == "PASS"
    assert rows[0]["status"] == "MATCH"

    source.write_text("changed", encoding="utf-8")
    failed, rows = verify_manifest(dataset, manifest)
    assert failed["status"] == "FAIL"
    assert failed["modified_files"] == 1
    assert rows[0]["status"] == "MODIFIED"


def test_checksum_manifest_can_report_unexpected_files(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "first.txt").write_text("one", encoding="utf-8")
    manifest = tmp_path / "checksums.csv"
    build_manifest(dataset, manifest)
    (dataset / "second.txt").write_text("two", encoding="utf-8")
    report = tmp_path / "verify.csv"

    summary, rows = verify_manifest(dataset, manifest, report, check_extra=True)

    assert summary["status"] == "FAIL"
    assert summary["unexpected_files"] == 1
    assert any(row["relative_path"] == "second.txt" and row["status"] == "UNEXPECTED" for row in rows)
    with report.open(encoding="utf-8", newline="") as handle:
        assert {row["status"] for row in csv.DictReader(handle)} == {"MATCH", "UNEXPECTED"}


def test_checksum_rejects_a_manifest_path_that_escapes_the_input(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = tmp_path / "checksums.csv"
    manifest.write_text(
        "relative_path,size_bytes,sha256\n../outside.txt,1," + "0" * 64 + "\n",
        encoding="utf-8",
    )

    try:
        verify_manifest(dataset, manifest)
    except ValueError as exc:
        assert "Unsafe relative_path" in str(exc)
    else:
        raise AssertionError("Expected an unsafe manifest path to be rejected")


def test_checksum_does_not_overwrite_the_input_file_with_a_manifest(tmp_path: Path) -> None:
    source = tmp_path / "dataset.zip"
    source.write_bytes(b"dataset")

    try:
        build_manifest(source, source)
    except ValueError as exc:
        assert "must not be the same" in str(exc)
    else:
        raise AssertionError("Expected the manifest/input path collision to be rejected")

    assert source.read_bytes() == b"dataset"
