"""Kiểm tra leakage theo sequence; không tự áp dụng split."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def detect_leakage(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        dataset = result["dataset_name"]
        sequence_splits: dict[str, set[str]] = defaultdict(set)
        for row in result.get("sequences", []):
            split = str(row.get("split", "NOT_PROVIDED"))
            sequence = str(row.get("sequence_name", ""))
            if sequence and split not in {"", "NOT_PROVIDED", "UNKNOWN"}:
                sequence_splits[sequence].add(split)
        for sequence, splits in sorted(sequence_splits.items()):
            if len(splits) > 1:
                rows.append(
                    {
                        "dataset_name": dataset,
                        "sequence_name": sequence,
                        "leakage_type": "SAME_SEQUENCE_MULTIPLE_SPLITS",
                        "splits": "|".join(sorted(splits)),
                        "severity": "CRITICAL",
                        "evidence": "Sequence metadata",
                        "recommended_action": "KEEP_SEQUENCE_IN_ONE_SPLIT",
                        "review_status": "PENDING",
                    }
                )
    return rows


def assert_sequence_split(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(str(row["dataset_name"]), str(row["sequence_id"]))].add(str(row["proposed_split"]))
    conflicts = [key for key, splits in grouped.items() if len(splits) > 1]
    if conflicts:
        raise ValueError(f"Sequence xuất hiện ở nhiều split: {conflicts[:5]}")


__all__ = ["assert_sequence_split", "detect_leakage"]
