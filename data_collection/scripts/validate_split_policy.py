"""Audit proposal split theo sequence và giữ riêng main test K230."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def build_split_audit(
    splits: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    validations: list[dict[str, Any]] = []

    def check(
        check_id: str,
        status: str,
        evidence: str,
        required_action: str,
    ) -> None:
        validations.append(
            {
                "check_id": check_id,
                "status": status,
                "evidence": evidence,
                "required_action": required_action,
                "apply_status": str(policy.get("apply_status", "PROPOSAL_ONLY")),
            }
        )

    sequence_splits: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in splits:
        sequence_splits[(str(row["dataset_name"]), str(row["sequence_id"]))].add(
            str(row["proposed_split"])
        )
    conflicts = [key for key, values in sequence_splits.items() if len(values) > 1]
    check(
        "SEQUENCE_EXCLUSIVE",
        "PASS" if not conflicts else "FAIL",
        f"conflicting_sequence_groups={len(conflicts)}",
        "KEEP_EACH_SEQUENCE_IN_ONE_SPLIT" if conflicts else "NONE",
    )

    split_by_sequence = {
        (str(row["dataset_name"]), str(row["sequence_id"])): str(row["proposed_split"])
        for row in splits
    }
    file_splits: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in manifest:
        key = (str(row["dataset_name"]), str(row["sequence_id"]))
        proposed_split = split_by_sequence.get(key)
        if proposed_split:
            file_splits[(str(row["dataset_name"]), str(row["source_file"]))].add(proposed_split)
    file_conflicts = [key for key, values in file_splits.items() if len(values) > 1]
    check(
        "SOURCE_FILE_EXCLUSIVE",
        "PASS" if not file_conflicts else "FAIL",
        f"source_files_in_multiple_splits={len(file_conflicts)}",
        "REBUILD_MANIFEST_BY_GROUP" if file_conflicts else "NONE",
    )

    non_sequence_units = Counter(str(row.get("split_unit", "UNKNOWN")) for row in splits)
    check(
        "NO_RANDOM_FRAME_SPLIT",
        "PASS" if set(non_sequence_units) <= {"SEQUENCE_ID", "DATASET_TRAIN_ONLY_NO_SEQUENCE"} else "FAIL",
        ", ".join(f"{key}={value}" for key, value in sorted(non_sequence_units.items())),
        "USE_SEQUENCE_VIDEO_SESSION_GROUPS_ONLY",
    )

    cross = [row for row in splits if row.get("proposed_split") == "CROSS_DATASET_TEST"]
    cross_datasets = {str(row["dataset_name"]) for row in cross}
    minimum_datasets = int(policy.get("cross_test_requirements", {}).get("minimum_datasets", 2))
    check(
        "CROSS_DATASET_SOURCE_COVERAGE",
        "PASS" if len(cross_datasets) >= minimum_datasets else "FAIL",
        f"datasets={sorted(cross_datasets)}",
        "ADD_INDEPENDENT_DATASET_SEQUENCES" if len(cross_datasets) < minimum_datasets else "NONE",
    )

    validation_datasets = {
        str(row["dataset_name"])
        for row in splits
        if row.get("proposed_split") == "EXTERNAL_VALIDATION"
    }
    expected_validation_datasets = {
        str(row["dataset_name"])
        for row in splits
        if row.get("evaluation_eligible")
    }
    missing_validation_datasets = sorted(expected_validation_datasets - validation_datasets)
    check(
        "VALIDATION_DATASET_COVERAGE",
        "PASS" if not missing_validation_datasets else "PARTIAL",
        f"observed={sorted(validation_datasets)}; missing={missing_validation_datasets}",
        "ADD_VALIDATION_SEQUENCE_FOR_MISSING_DATASET" if missing_validation_datasets else "NONE",
    )

    required_roads = set(policy.get("cross_test_requirements", {}).get("required_road_types", []))
    observed_roads = {str(row.get("road_type", "UNKNOWN")) for row in cross}
    missing_roads = sorted(required_roads - observed_roads)
    check(
        "CROSS_ROAD_TYPE_COVERAGE",
        "PASS" if not missing_roads else "PARTIAL",
        f"observed={sorted(observed_roads)}; missing={missing_roads}",
        "COLLECT_OR_SOURCE_MISSING_ROAD_TYPES" if missing_roads else "NONE",
    )

    required_lighting = set(policy.get("cross_test_requirements", {}).get("required_lighting", []))
    observed_lighting = {str(row.get("lighting", "UNKNOWN")) for row in cross}
    missing_lighting = sorted(required_lighting - observed_lighting)
    check(
        "CROSS_LIGHTING_COVERAGE",
        "PASS" if not missing_lighting else "PARTIAL",
        f"observed={sorted(observed_lighting)}; missing={missing_lighting}",
        "ADD_MISSING_LIGHTING_SEQUENCE" if missing_lighting else "NONE",
    )

    mio_rows = [row for row in splits if row.get("dataset_name") == "MIO-TCD Localization"]
    mio_safe = bool(mio_rows) and all(
        row.get("proposed_split") == "EXTERNAL_TRAIN" and not row.get("evaluation_eligible")
        for row in mio_rows
    )
    check(
        "MIO_NO_SEQUENCE_TRAIN_ONLY",
        "PASS" if mio_safe else "FAIL",
        f"rows={len(mio_rows)}; splits={sorted({str(row.get('proposed_split')) for row in mio_rows})}",
        "KEEP_MIO_OUT_OF_VALIDATION_AND_TEST" if not mio_safe else "NONE",
    )

    fixed_missing: list[str] = []
    cross_keys = {(str(row["dataset_name"]), str(row["sequence_id"])) for row in cross}
    for dataset, sequences in policy.get("fixed_cross_test_sequences", {}).items():
        for sequence in sequences:
            if (str(dataset), str(sequence)) not in cross_keys:
                fixed_missing.append(f"{dataset}:{sequence}")
    check(
        "FIXED_CROSS_TEST_MEMBERSHIP",
        "PASS" if not fixed_missing else "FAIL",
        f"missing={fixed_missing}",
        "RESTORE_REVIEWED_CROSS_TEST_SEQUENCE" if fixed_missing else "NONE",
    )

    main_test = policy.get("main_test", {})
    check(
        "K230_MAIN_TEST_RESERVED",
        "PENDING" if main_test.get("status") == "PLACEHOLDER_PENDING_COLLECTION" else "PASS",
        f"source={main_test.get('source')}; status={main_test.get('status')}",
        "COLLECT_K230_AND_LOCK_AFTER_DATA_LEAD_APPROVAL",
    )
    check(
        "MAIN_TEST_LOCKED",
        "PASS" if main_test.get("locked") else "PENDING",
        f"locked={bool(main_test.get('locked'))}",
        "LOCK_MANIFEST_AFTER_K230_COLLECTION_AND_REVIEW",
    )

    manifest_files: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in manifest:
        split = split_by_sequence.get((str(row["dataset_name"]), str(row["sequence_id"])), "UNKNOWN")
        manifest_files[(str(row["dataset_name"]), split, str(row["sequence_id"]))].add(
            str(row.get("source_file", ""))
        )
    sequence_counts: Counter[tuple[str, str]] = Counter()
    image_counts: Counter[tuple[str, str]] = Counter()
    for row in splits:
        sequence_counts[(str(row["dataset_name"]), str(row["proposed_split"]))] += 1
    for (dataset, split, _sequence), files in manifest_files.items():
        image_counts[(dataset, split)] += len(files)
    dataset_totals: Counter[str] = Counter()
    for (dataset, _split), count in sequence_counts.items():
        dataset_totals[dataset] += count
    ratios = policy.get("external_ratios", {})
    overrides = policy.get("dataset_overrides", {})
    distribution = [
        {
            "dataset_name": dataset,
            "proposed_split": split,
            "sequence_or_group_count": count,
            "actual_sequence_ratio": round(count / dataset_totals[dataset], 8),
            "target_sequence_ratio": "" if dataset in overrides else ratios.get(split, ""),
            "unique_candidate_images_in_manifest": image_counts[(dataset, split)],
            "apply_status": str(policy.get("apply_status", "PROPOSAL_ONLY")),
            "distribution_scope": "UNIQUE_CANDIDATES_ACROSS_ALL_SUBSET_SCENARIOS",
        }
        for (dataset, split), count in sorted(sequence_counts.items())
    ]

    holdout = [
        {
            "split_name": main_test.get("split_name", "MAIN_K230_TEST"),
            "source": main_test.get("source", "K230_SELF_RECORDED"),
            "slice_id": row.get("slice_id", ""),
            "dimension": row.get("dimension", ""),
            "value": row.get("value", ""),
            "minimum_sessions": row.get("minimum_sessions", 1),
            "current_sessions": 0,
            "status": "PENDING_COLLECTION",
            "locked": bool(main_test.get("locked", False)),
            "notes": "Kế hoạch test; chưa phải số liệu đã thu.",
        }
        for row in main_test.get("required_slices", [])
    ]
    return validations, distribution, holdout


__all__ = ["build_split_audit"]
