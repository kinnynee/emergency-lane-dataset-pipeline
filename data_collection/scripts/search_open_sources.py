"""Xuất các truy vấn cần chạy thủ công; không crawl hoặc tạo URL giả."""

from __future__ import annotations

import argparse

from online_common import ROOT, read_csv


def main() -> int:
    """In truy vấn, không thực hiện tải/crawl tự động."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", help="Lọc theo query_group")
    args = parser.parse_args()
    rows = read_csv("planning/online_search_queries.csv")
    if args.group:
        rows = [row for row in rows if row.get("query_group") == args.group]
    lines = ["# Danh sách truy vấn nguồn online", "", "Môi trường/script này không tự crawl hay tải nguồn; reviewer tìm trên website chính thức rồi ghi evidence vào registry.", ""]
    lines += [f"- [{row['query_id']}] ({row['language']}) {row['query_text']}" for row in rows]
    (ROOT / "reports/search_queries_to_run.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã xuất {len(rows)} truy vấn vào reports/search_queries_to_run.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
