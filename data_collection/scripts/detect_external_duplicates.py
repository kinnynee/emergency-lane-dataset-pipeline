"""Đề xuất nhóm trùng trên mẫu; không xóa file."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from external_eda_common import hamming_similarity


def detect_duplicates(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_index = 1
    records = [record for result in results for record in result.get("image_records", [])]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_name: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_hash[str(record.get("sha256", ""))].append(record)
        by_name[(record["dataset_name"], Path(record["source_file"]).name.lower())].append(record)
    for values in by_hash.values():
        if len(values) < 2:
            continue
        group_id = f"DUP_{group_index:05d}"
        group_index += 1
        for index, record in enumerate(values):
            rows.append(
                {
                    "duplicate_group_id": group_id,
                    "dataset_name": record["dataset_name"],
                    "sequence_name": record["sequence_name"],
                    "file_path": record["source_file"],
                    "duplicate_type": "EXACT_SHA256",
                    "similarity_score": 1.0,
                    "recommended_keep": index == 0,
                    "recommended_action": "KEEP_ONE_AFTER_MANUAL_REVIEW",
                    "review_status": "PENDING",
                }
            )
    for values in by_name.values():
        distinct = {record["source_file"] for record in values}
        if len(distinct) < 2:
            continue
        group_id = f"DUP_{group_index:05d}"
        group_index += 1
        for index, record in enumerate(values):
            rows.append(
                {
                    "duplicate_group_id": group_id,
                    "dataset_name": record["dataset_name"],
                    "sequence_name": record["sequence_name"],
                    "file_path": record["source_file"],
                    "duplicate_type": "DUPLICATE_BASENAME",
                    "similarity_score": "",
                    "recommended_keep": index == 0,
                    "recommended_action": "VERIFY_CONTENT",
                    "review_status": "PENDING",
                }
            )
    by_sequence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["sequence_name"] != "SEQUENCE_NOT_PROVIDED":
            by_sequence[(record["dataset_name"], record["sequence_name"])].append(record)
    for values in by_sequence.values():
        values.sort(key=lambda row: row["source_file"])
        for left, right in zip(values, values[1:]):
            similarity = hamming_similarity(int(left["phash"]), int(right["phash"]))
            if similarity < 0.95 or left.get("sha256") == right.get("sha256"):
                continue
            group_id = f"DUP_{group_index:05d}"
            group_index += 1
            for index, record in enumerate((left, right)):
                rows.append(
                    {
                        "duplicate_group_id": group_id,
                        "dataset_name": record["dataset_name"],
                        "sequence_name": record["sequence_name"],
                        "file_path": record["source_file"],
                        "duplicate_type": "NEAR_DUPLICATE_PHASH_CONSECUTIVE",
                        "similarity_score": similarity,
                        "recommended_keep": index == 0,
                        "recommended_action": "TEMPORAL_DOWNSAMPLE_AFTER_REVIEW",
                        "review_status": "PENDING",
                    }
                )
    return rows


__all__ = ["detect_duplicates"]
