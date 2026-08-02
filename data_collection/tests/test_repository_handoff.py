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
        REPO_ROOT / "data_collection" / "configs" / "aau_sequence_lighting_review.yaml",
        REPO_ROOT / "data_collection" / "docs" / "21_external_dataset_eda_methodology.md",
        REPO_ROOT / "data_collection" / "docs" / "25_eda_distribution_quality_split.md",
        REPO_ROOT / "data_collection" / "docs" / "26_supervisor_feedback_corrections.md",
        REPORT_ROOT / "executive_summary.md",
        REPORT_ROOT / "quality_audit_summary.csv",
        REPORT_ROOT / "split_validation_summary.csv",
        REPORT_ROOT / "k230_holdout_plan.csv",
        REPORT_ROOT / "evaluation_slice_readiness.csv",
        REPORT_ROOT / "ua_others_sample_review.csv",
        REPORT_ROOT / "figure_provenance.csv",
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


def test_tracked_reports_reflect_supervisor_corrections() -> None:
    conditions = _read_csv(REPORT_ROOT / "condition_distribution.csv")
    aau_lighting = {
        row["value"]: int(row["count"])
        for row in conditions
        if row["dataset_name"] == "AAU RainSnow" and row["condition"] == "lighting"
    }
    assert aau_lighting == {"DAY": 10, "NIGHT": 11, "TWILIGHT": 1}
    assert all("AUTOMATIC" not in row["assessment_source"] for row in conditions if row["dataset_name"] == "AAU RainSnow" and row["condition"] == "lighting")

    quality = _read_csv(REPORT_ROOT / "quality_audit_summary.csv")
    ua = next(row for row in quality if row["dataset_name"] == "UA-DETRAC Original")
    assert int(ua["invalid_annotations_unique"]) == 0
    assert int(ua["boundary_clipped_bbox_count"]) > 0

    holdout = _read_csv(REPORT_ROOT / "k230_holdout_plan.csv")
    backlit = next(row for row in holdout if row["slice_id"] == "K230_BACKLIT")
    assert backlit["value"] == "BACKLIT"
    assert backlit["status"] == "PENDING_COLLECTION"


def test_reports_use_audited_bbox_scope_and_neutral_ua_cause() -> None:
    root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    methodology = (
        REPO_ROOT / "data_collection" / "docs" / "25_eda_distribution_quality_split.md"
    ).read_text(encoding="utf-8")
    executive = (REPORT_ROOT / "executive_summary.md").read_text(encoding="utf-8")

    assert "analysis-scope sum" in root_readme
    assert "không phải full-raw total" in executive
    assert "khác quy ước tọa độ/off-by-one" in methodology
    assert "chưa có đủ bằng chứng để kết luận" in methodology


def test_every_tracked_figure_has_verified_csv_provenance() -> None:
    rows = _read_csv(REPORT_ROOT / "figure_provenance.csv")
    tracked_figures = {
        path.relative_to(REPORT_ROOT).as_posix()
        for path in (REPORT_ROOT / "figures").glob("*.png")
    }

    assert {row["figure_path"] for row in rows} == tracked_figures
    assert all(row["status"] == "VERIFIED_CSV_SOURCE" for row in rows)
    assert all(row["source_csvs"] and row["source_sha256s"] for row in rows)
    assert all(len(row["figure_sha256"]) == 64 for row in rows)
