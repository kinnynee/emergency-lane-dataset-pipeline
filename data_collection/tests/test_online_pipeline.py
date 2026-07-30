"""Kiểm thử không phá dữ liệu cho các guard quan trọng của pipeline online."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from download_approved_sources import eligible
from online_common import safe_name, sha256
from propose_dataset_split import split_for
from validate_source_license import source_issues


class OnlinePipelineTests(unittest.TestCase):
    """Các kiểm thử guard: license, URL, checksum, split và không ghi đè."""

    def test_unverified_license_is_blocked(self) -> None:
        self.assertIn("license_verified không phải TRUE", source_issues({"license_verified": "UNVERIFIED"}))

    def test_empty_url_is_not_eligible(self) -> None:
        self.assertFalse(eligible({"download_status": "APPROVED", "license_verified": "TRUE", "permission_status": "APPROVED", "direct_download_url": ""}))

    def test_checksum_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"k230")
            self.assertEqual(sha256(path), sha256(path))

    def test_name_rule_removes_spaces_and_special_characters(self) -> None:
        self.assertEqual(safe_name("Video 01!.MP4"), "video_01_mp4")

    def test_same_video_has_same_split(self) -> None:
        self.assertEqual(split_for("VID_A"), split_for("VID_A"))

    def test_dry_run_guard_does_not_create_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "would_be_downloaded.mp4"
            self.assertFalse(target.exists())

    def test_report_has_no_sample_video_claim(self) -> None:
        report = (Path(__file__).resolve().parents[1] / "reports/online_source_report.md").read_text(encoding="utf-8")
        self.assertNotIn("Video đã tải: 1", report)


if __name__ == "__main__":
    unittest.main()
