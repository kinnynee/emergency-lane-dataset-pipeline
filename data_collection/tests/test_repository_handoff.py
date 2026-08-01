"""Smoke tests for a clean Git handoff without local dataset files."""

from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO_ROOT / "data_collection" / "reports" / "external_eda"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_required_handoff_files_are_present() -> None:
    required_paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "data_collection" / "README.md",
        REPO_ROOT / "data_collection" / "configs" / "split_policy.yaml",
        REPO_ROOT / "data_collection" / "docs" / "21_external_dataset_eda_methodology.md",
        REPO_ROOT / "data_collection" / "docs" / "25_eda_distribution_quality_split.md",
        REPORT_ROOT / "executive_summary.md",
        REPORT_ROOT / "quality_audit_summary.csv",
        REPORT_ROOT / "split_validation_summary.csv",
        REPORT_ROOT / "k230_holdout_plan.csv",
    ]

    missing = [str(path.relative_to(REPO_ROOT)) for path in required_paths if not path.is_file()]
    assert not missing, f"Missing repository handoff files: {missing}"


def test_split_validation_has_no_failed_checks() -> None:
    rows = _read_csv(REPORT_ROOT / "split_validation_summary.csv")
    assert rows, "Split validation summary must not be empty"

    failed = [row for row in rows if row.get("status", "").strip().upper() == "FAIL"]
    assert not failed, f"Split validation contains failed checks: {failed}"


def test_quality_summary_covers_all_selected_datasets() -> None:
    rows = _read_csv(REPORT_ROOT / "quality_audit_summary.csv")
    datasets = {
        (row.get("dataset") or row.get("dataset_name") or "").strip().lower()
        for row in rows
    }

    assert any("mio" in name for name in datasets)
    assert any("aau" in name for name in datasets)
    assert any("detrac" in name for name in datasets)


def test_root_readme_explains_reproduction_and_review_status() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    required_text = [
        "python -m pytest data_collection/tests -q",
        "executive_summary.md",
        "quality_audit_summary.csv",
        "split_validation_summary.csv",
        "PROPOSAL_ONLY",
        "REVIEW_REQUIRED",
        "RADIATE is excluded",
    ]

    missing = [text for text in required_text if text not in readme]
    assert not missing, f"README is missing handoff guidance: {missing}"
