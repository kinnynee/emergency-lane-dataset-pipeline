"""Tests for immutable raw-dataset checksum baselines."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from checksum_dataset import compare, read_baseline, scan, write_baseline  # noqa: E402


def test_checksum_baseline_is_deterministic_and_verifiable(tmp_path: Path) -> None:
    dataset = tmp_path / "raw"
    dataset.mkdir()
    (dataset / "b.bin").write_bytes(b"beta")
    (dataset / "a.bin").write_bytes(b"alpha")
    baseline = tmp_path / "baseline_checksums.csv"

    first = scan(dataset, "sha256")
    second = scan(dataset, "sha256")
    assert first == second
    assert [row["relative_path"] for row in first] == ["a.bin", "b.bin"]

    write_baseline(baseline, first)
    assert compare(read_baseline(baseline), second)["status"] == "PASS"


def test_checksum_verify_reports_changed_missing_and_added(tmp_path: Path) -> None:
    dataset = tmp_path / "raw"
    dataset.mkdir()
    (dataset / "changed.bin").write_bytes(b"before")
    (dataset / "missing.bin").write_bytes(b"remove me")
    baseline = tmp_path / "baseline_checksums.csv"
    write_baseline(baseline, scan(dataset, "sha256"))

    (dataset / "changed.bin").write_bytes(b"after")
    (dataset / "missing.bin").unlink()
    (dataset / "added.bin").write_bytes(b"new")
    result = compare(read_baseline(baseline), scan(dataset, "sha256"))

    assert result["status"] == "FAIL"
    assert result["changed"] == ["changed.bin"]
    assert result["missing"] == ["missing.bin"]
    assert result["added"] == ["added.bin"]
