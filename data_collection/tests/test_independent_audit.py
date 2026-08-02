"""Regression checks for the independent EDA audit evidence.

The raw-archive test is opt-in because a complete run parses more than 1.6
million annotations and is intentionally separate from the fast handoff suite.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = REPO_ROOT / "data_collection" / "reports" / "audit"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_independent_count_evidence_is_conserved() -> None:
    """Tracked audit evidence must retain the independently verified totals."""

    rows = _read_csv(AUDIT_ROOT / "dataset_count_reconciliation.csv")
    assert {row["dataset"] for row in rows} == {
        "MIO-TCD Localization",
        "AAU RainSnow",
        "UA-DETRAC Original",
    }
    assert all(row["conservation_ok"].lower() == "true" for row in rows)
    assert sum(int(row["final_analysis_count"]) for row in rows) == 1_301_866

    ua_metrics = {
        row["metric"]: row["value"]
        for row in _read_csv(AUDIT_ROOT / "ua_bbox_clipping_audit.csv")
    }
    assert int(ua_metrics["out_of_bounds_count"]) == 130_181
    assert float(ua_metrics["overrun_max"]) == 1.0
    assert int(ua_metrics["side_LEFT"]) == 0
    assert int(ua_metrics["side_TOP"]) == 0


@pytest.mark.skipif(
    os.environ.get("K230_RUN_RAW_AUDIT_TESTS") != "1",
    reason="Set K230_RUN_RAW_AUDIT_TESTS=1 for the multi-minute raw archive audit.",
)
def test_independent_raw_archive_audit(tmp_path: Path) -> None:
    """Opt-in integration test that does not import production parser code."""

    output = tmp_path / "reproduction"
    audit = tmp_path / "audit"
    command = [
        sys.executable,
        str(REPO_ROOT / "data_collection" / "scripts" / "audit_eda_independent.py"),
        "--output",
        str(output),
        "--audit-dir",
        str(audit),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True, timeout=1_200)
    result = json.loads((output / "independent_raw_audit.json").read_text(encoding="utf-8"))

    assert result["verified_total"] == 1_301_866
    assert result["difference"] == 0
    assert result["ua"]["out_of_bounds_count"] == 130_181
    assert result["ua"]["invalid_count"] == 0
