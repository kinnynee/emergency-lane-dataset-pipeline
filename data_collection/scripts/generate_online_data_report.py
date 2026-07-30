"""Tạo EDA chỉ từ registry/metadata thực tế, không chèn số liệu mẫu."""

from __future__ import annotations

from collections import Counter

from online_common import ROOT, read_csv


def count(rows: list[dict[str, str]], field: str) -> str:
    """Tóm tắt phân bố hoặc báo chưa có."""
    values = Counter(row.get(field) or "CHƯA GHI" for row in rows)
    return "\n".join(f"- {key}: {value}" for key, value in sorted(values.items())) if values else "CHƯA CÓ DỮ LIỆU."


def main() -> int:
    """Sinh online_source_report.md và daily_report_draft.md."""
    sources = read_csv("planning/online_source_candidates.csv")
    videos = read_csv("planning/video_quality_review.csv")
    images = read_csv("metadata/images.csv")
    approved_sources = sum(row.get("review_status") in {"APPROVED_FOR_DOWNLOAD", "APPROVED_INTERNAL_ONLY", "DOWNLOADED", "PROCESSED"} for row in sources)
    verified = sum(row.get("license_verified") == "TRUE" for row in sources)
    permission = sum(row.get("review_status") == "NEEDS_PERMISSION" for row in sources)
    rejected = sum(row.get("review_status") == "REJECTED" for row in sources)
    lines = ["# Báo cáo dữ liệu Internet", "", "## Nguồn", f"- Tổng nguồn registry: {len(sources)}", f"- License đã xác minh: {verified}", f"- Cần xin phép: {permission}", f"- Đã duyệt: {approved_sources}", f"- Bị loại: {rejected}", "", "## Video", f"- Tổng video metadata: {len(videos)}", f"- Tổng thời lượng: {sum(float(row.get('duration_seconds') or 0) for row in videos):.2f} giây" if videos else "- Tổng thời lượng: CHƯA CÓ DỮ LIỆU", "", "## Phân bố camera", count(videos, "camera_type"), "", "## Ánh sáng/thời tiết", count(videos, "lighting_condition"), count(videos, "weather_condition"), "", "## Frame", f"- Tổng frame metadata: {len(images)}", f"- Near-duplicate đã đánh dấu: {sum((row.get('duplicate_status') or '').startswith('NEAR') for row in images)}", "", "## Cần review", "- Nguồn có legal_risk HIGH/UNKNOWN: " + ", ".join(row["source_id"] for row in sources if row.get("legal_risk") in {"HIGH", "UNKNOWN"}) or "- CHƯA CÓ", "- Xem planning/dataset_condition_gap.csv để biết khoảng trống điều kiện."]
    (ROOT / "reports/online_source_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    daily = ["# [SV1 - ngày 29/07/2026]", "", "1. Hôm nay làm:", "- Chạy báo cáo registry/metadata online.", "", "2. Kết quả/bằng chứng:", f"- Số nguồn tìm được: {len(sources)}.", f"- Số nguồn được duyệt: {approved_sources if approved_sources else 'CHƯA CÓ'}.", f"- Số video tải: {len(videos) if videos else 'CHƯA CÓ'}.", f"- Số frame đã tách: {len(images) if images else 'CHƯA CÓ'}.", "- Link báo cáo: reports/online_source_report.md.", "", "3. Vướng mắc/cần hỗ trợ:", f"- Nguồn chưa xác minh/cần xin phép: {permission if permission else 'CHƯA CÓ'}.", "- Vị trí quay K230 chưa có evidence trong metadata.", "", "4. Ngày mai:", "- Review license/permission và chỉ đưa source đủ điều kiện vào queue."]
    (ROOT / "reports/daily_report_draft.md").write_text("\n".join(daily) + "\n", encoding="utf-8")
    print("Đã ghi reports/online_source_report.md và reports/daily_report_draft.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
