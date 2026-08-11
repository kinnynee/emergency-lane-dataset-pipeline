from __future__ import annotations

import inspect
import json
import csv
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_external_dataset_eda as runner
from create_balanced_subset_plan import create_plan
from external_eda_common import reservoir_sample
from inspect_aau_rainsnow import inspect_aau


def test_empty_statistics_remain_empty() -> None:
    summary = runner._summary_numeric([], "value")
    assert summary == {"count": 0, "min": "", "mean": "", "median": "", "max": ""}


def test_reservoir_sample_does_not_materialize_generator_first() -> None:
    consumed = []

    def values():
        for value in range(100):
            consumed.append(value)
            yield value

    sample = reservoir_sample(values(), 5)
    assert len(sample) == 5
    assert len(consumed) == 100


def test_no_destructive_file_calls_in_eda_scripts() -> None:
    for path in SCRIPTS.glob("*external*.py"):
        source = path.read_text(encoding="utf-8")
        assert ".unlink(" not in source
        assert "rmtree(" not in source


def test_selection_and_split_are_proposal_only_by_default() -> None:
    parser = runner.build_parser()
    args = parser.parse_args([])
    assert args.apply_selection is False
    assert args.apply_split is False
    empty_result = {
        "dataset_name": "MIO-TCD Localization",
        "bbox_samples": [],
        "quality_rows": [],
    }
    plans, manifest, splits = create_plan([empty_result])
    assert all(row["apply_status"] == "PROPOSAL_ONLY" for row in plans)
    assert manifest == []
    assert splits == []


def test_archive_inspectors_process_one_image_variable_at_a_time() -> None:
    source = inspect.getsource(__import__("inspect_mio_tcd").inspect_mio)
    assert "image = image_from_bytes(data)" in source
    assert "images_in_memory" not in source


def test_resume_cache_reuses_completed_result(tmp_path: Path) -> None:
    cache = tmp_path / "result.json"
    expected = {"dataset_name": "UA-DETRAC Original", "status": "ANALYZED"}
    runner._save_cache(cache, expected)
    assert runner._load_cache(cache) == expected


def test_resume_cache_rejects_stale_fingerprint(tmp_path: Path) -> None:
    cache = tmp_path / "result.json"
    expected = {"dataset_name": "UA-DETRAC Original", "status": "ANALYZED"}
    first_identity = {"fingerprint": "first", "payload": {"source": "version-1"}}
    second_identity = {"fingerprint": "second", "payload": {"source": "version-2"}}

    runner._save_cache(cache, expected, first_identity)

    assert runner._load_cache(cache, first_identity) == expected
    assert runner._load_cache(cache, second_identity) is None
    assert runner._load_cache(cache) == expected


def test_aud_002_is_closed_only_with_cache_fingerprint_regression_coverage() -> None:
    findings = Path(__file__).resolve().parents[1] / "reports" / "audit" / "audit_findings.csv"
    with findings.open(encoding="utf-8-sig", newline="") as handle:
        audit_002 = next(row for row in csv.DictReader(handle) if row["finding_id"] == "AUD-002")

    assert audit_002["status"] == "CLOSED"
    assert "fingerprint" in audit_002["evidence"].lower()


def test_figure_provenance_hashes_declared_csv_source(tmp_path: Path) -> None:
    source = tmp_path / "dataset_inventory.csv"
    source.write_text("dataset_name,image_count\nExample,1\n", encoding="utf-8")
    figure = tmp_path / "figures" / "01_images_by_dataset.png"
    figure.parent.mkdir()
    figure.write_bytes(b"deterministic-figure-placeholder")

    rows = runner._write_figure_provenance(tmp_path, [figure])

    assert rows[0]["status"] == "VERIFIED_CSV_SOURCE"
    assert rows[0]["source_csvs"] == "dataset_inventory.csv"
    assert len(rows[0]["figure_sha256"]) == 64
    assert "dataset_inventory.csv:" in rows[0]["source_sha256s"]


def test_aau_without_annotations_does_not_create_fake_error(tmp_path: Path) -> None:
    payload = {"images": [], "annotations": [], "categories": []}
    (tmp_path / "aauRainSnow-rgb.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = inspect_aau(
        tmp_path,
        sample_size=10,
        full_scan=False,
        skip_images=True,
        progress=False,
    )
    assert result["annotation_row_count"] == 0
    assert result["invalid_annotations"] == []


def test_ua_bbox_sample_is_streaming_reservoir_not_first_n() -> None:
    source = inspect.getsource(__import__("inspect_ua_detrac").inspect_ua_detrac)
    assert "bbox_rng.randint(0, bbox_seen - 1)" in source
    assert "bbox_samples[replacement] = bbox_sample" in source
