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
        REPO_ROOT / "data_collection" / "requirements-data-lock.txt",
        REPO_ROOT / "data_collection" / "planning" / "k230_backlit_collection_protocol.csv",
        REPO_ROOT / "data_collection" / "planning" / "k230_evaluation_sessions.csv",
        REPO_ROOT / "data_collection" / "docs" / "21_external_dataset_eda_methodology.md",
        REPO_ROOT / "data_collection" / "docs" / "25_eda_distribution_quality_split.md",
        REPO_ROOT / "data_collection" / "docs" / "26_supervisor_feedback_corrections.md",
        REPO_ROOT / "data_collection" / "docs" / "27_training_export_and_k230_readiness.md",
        REPO_ROOT / "data_collection" / "scripts" / "export_ua_detrac_yolo.py",
        REPO_ROOT / "data_collection" / "scripts" / "validate_k230_evaluation_readiness.py",
        REPO_ROOT / "data_collection" / "scripts" / "evaluate_yolo_slices.py",
        REPO_ROOT / "data_collection" / "scripts" / "run_full_ua_detrac_replay.py",
        REPO_ROOT / "data_collection" / "scripts" / "checksum_dataset.py",
        REPORT_ROOT / "executive_summary.md",
        REPORT_ROOT / "quality_audit_summary.csv",
        REPORT_ROOT / "split_validation_summary.csv",
        REPORT_ROOT / "k230_holdout_plan.csv",
        REPORT_ROOT / "evaluation_slice_readiness.csv",
        REPORT_ROOT / "ua_others_sample_review.csv",
        REPORT_ROOT / "ua_others_stratified_review_queue.csv",
        REPORT_ROOT / "ua_others_stratified_review_decisions.csv",
        REPORT_ROOT / "ua_others_track_exclusions.csv",
        REPORT_ROOT / "ua_others_data_lead_review.md",
        REPORT_ROOT / "aau_lighting_data_lead_review.md",
        REPORT_ROOT / "ua_yolo_export_smoke_test.md",
        REPORT_ROOT / "baseline_checksums.csv",
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

    aau_config = (
        REPO_ROOT / "data_collection" / "configs" / "aau_sequence_lighting_review.yaml"
    ).read_text(encoding="utf-8")
    assert 'reviewer: "CODEX_ACTING_DATA_LEAD"' in aau_config
    assert 'review_status: "DATA_LEAD_SIGNOFF_COMPLETED"' in aau_config

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


def test_ua_others_review_is_stratified_and_not_overclaimed() -> None:
    rows = _read_csv(REPORT_ROOT / "ua_others_stratified_review_queue.csv")
    assessments = {}
    for row in rows:
        assessments[row["visual_assessment"]] = assessments.get(row["visual_assessment"], 0) + 1

    assert len(rows) == 60
    assert len({row["sequence_id"] for row in rows}) >= 40
    assert {row["boundary_status"] for row in rows} == {"BOUNDARY_CLIPPED", "IN_FRAME"}
    assert len({row["weather"] for row in rows}) >= 4
    assert assessments == {
        "CONFIRMED_MOTORIZED_VEHICLE": 58,
        "NON_VEHICLE": 2,
    }
    assert all(row["review_status"] == "DATA_LEAD_SIGNOFF_COMPLETED" for row in rows)
    assert all(row["preserve_original_class"] == "TRUE" for row in rows)
    assert all(
        row["include_for_training"] == "FALSE_REVIEW_REJECT"
        for row in rows
        if row["visual_assessment"] == "NON_VEHICLE"
    )
    assert all(
        row["include_for_training"] == "TRUE_DATA_LEAD_APPROVED"
        for row in rows
        if row["visual_assessment"] == "CONFIRMED_MOTORIZED_VEHICLE"
    )
    exclusions = _read_csv(REPORT_ROOT / "ua_others_track_exclusions.csv")
    assert exclusions == [
        {
            "dataset_name": "UA-DETRAC Original",
            "sequence_id": "MVI_40172",
            "track_id": "79",
            "original_class": "others",
            "action": "EXCLUDE_NON_VEHICLE_TRACK",
            "excluded_bbox_count": "201",
            "reviewer": "CODEX_ACTING_DATA_LEAD",
            "review_date": "2026-08-02",
            "review_status": "DATA_LEAD_SIGNOFF_COMPLETED",
            "reason": "Stationary roadside bus-stop or advertising structure with people; not a motor vehicle. Exclude the complete track, not only sampled frames.",
        }
    ]


def test_k230_backlit_protocol_cannot_fake_a_missing_score() -> None:
    protocol = _read_csv(
        REPO_ROOT / "data_collection" / "planning" / "k230_backlit_collection_protocol.csv"
    )
    evaluation = next(row for row in protocol if row["protocol_id"] == "BL009")
    split_lock = next(row for row in protocol if row["protocol_id"] == "BL008")

    assert evaluation["status"] == "NOT_AVAILABLE"
    assert "ground truth and model predictions" in evaluation["acceptance_criteria"]
    assert split_lock["target"] == "MAIN_K230_TEST_ONLY"
    assert all(row["status"] != "COMPLETED" for row in protocol)


def test_reproduction_dependencies_are_exactly_locked() -> None:
    lock = (
        REPO_ROOT / "data_collection" / "requirements-data-lock.txt"
    ).read_text(encoding="utf-8").splitlines()
    requirements = [line for line in lock if line and not line.startswith("#")]

    assert requirements
    assert all("==" in requirement for requirement in requirements)
    assert not any(">=" in requirement or "<=" in requirement for requirement in requirements)


def test_tracked_reports_do_not_expose_machine_absolute_paths() -> None:
    report_files = list(REPORT_ROOT.glob("*.csv")) + list(REPORT_ROOT.glob("*.md"))
    report_files.append(REPO_ROOT / "data_collection" / "reports" / "near_duplicate_frame_report.md")
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in report_files
        if "C:\\UMTLab\\k230" in path.read_text(encoding="utf-8", errors="ignore")
    ]

    assert not offenders, f"Machine-specific paths remain in tracked reports: {offenders}"


def test_smoke_config_is_portable_and_matches_500_image_run() -> None:
    config_path = REPO_ROOT / "data_collection" / "configs" / "yolo11n_320_smoke_v2_500.yaml"
    runner_path = REPO_ROOT / "data_collection" / "scripts" / "run_yolo11n_smoke_v2.py"
    assert config_path.is_file()
    assert not (config_path.parent / "yolo11n_320_smoke_v2_750.yaml").exists()
    combined = config_path.read_text(encoding="utf-8") + runner_path.read_text(encoding="utf-8")
    assert "train_images: 500" in combined
    assert "750 train images" not in combined
    assert "D:/UMT_EVIDENCE" not in combined
    assert "D:\\UMT_EVIDENCE" not in combined
    assert "--source-dataset" in combined
    assert "--ua-annotation-root" in combined
