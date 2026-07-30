"""Chặn hàng chờ tải nếu license, permission hoặc URL không đủ điều kiện."""

from __future__ import annotations

import argparse
import sys

from online_common import ROOT, markdown_table, read_csv

ALLOWED_PERMISSION = {"APPROVED", "NOT_REQUIRED"}


def source_issues(source: dict[str, str]) -> list[str]:
    """Trả về lý do một source chưa thể tải tự động."""
    issues: list[str] = []
    if source.get("license_verified") != "TRUE":
        issues.append("license_verified không phải TRUE")
    if not source.get("license_name") or source.get("license_name") == "UNVERIFIED":
        issues.append("license_name trống/chưa xác minh")
    if source.get("permission_status") not in ALLOWED_PERMISSION:
        issues.append("permission_status chưa APPROVED/NOT_REQUIRED")
    if source.get("redistribution_allowed") in {"", "UNVERIFIED"}:
        issues.append("redistribution_allowed chưa rõ")
    if source.get("review_status") != "APPROVED_FOR_DOWNLOAD":
        issues.append("review_status chưa APPROVED_FOR_DOWNLOAD")
    return issues


def main() -> int:
    """Tạo báo cáo và trả lỗi khi queue chứa source không hợp lệ."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default="planning/video_download_queue.csv")
    args = parser.parse_args()
    sources = {row["source_id"]: row for row in read_csv("planning/online_source_candidates.csv")}
    queued = [row for row in read_csv(args.queue) if row.get("download_status") in {"APPROVED", "DOWNLOADING"}]
    findings: list[dict[str, str]] = []
    for item in queued:
        source = sources.get(item.get("source_id", ""), {})
        issues = source_issues(source)
        if item.get("license_verified") != "TRUE":
            issues.append("queue license_verified không phải TRUE")
        if item.get("permission_status") not in ALLOWED_PERMISSION:
            issues.append("queue permission_status chưa hợp lệ")
        if not item.get("direct_download_url"):
            issues.append("direct_download_url trống")
        findings.append({"download_id": item.get("download_id", ""), "source_id": item.get("source_id", ""), "result": "BLOCKED" if issues else "PASS", "reason": "; ".join(issues) or "Đủ điều kiện"})
    report = ["# Kiểm tra giấy phép", "", "## Kết quả"]
    report += markdown_table(findings, ["download_id", "source_id", "result", "reason"])
    report += ["", "## Ghi chú", "Chỉ các dòng queue APPROVED/DOWNLOADING được kiểm tra để chặn tải. Không có dòng nào không có nghĩa là có nguồn được duyệt."]
    (ROOT / "reports/license_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Đã ghi reports/license_validation_report.md; queue cần kiểm tra: {len(queued)}")
    return 1 if any(row["result"] == "BLOCKED" for row in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
