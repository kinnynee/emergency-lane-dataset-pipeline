import argparse
import csv
from pathlib import Path

HEADER = [
    "dataset_name",
    "sequence_id",
    "road_type",
    "weather",
    "lighting",
    "camera_view",
    "traffic_density",
    "mean_vehicles_per_image",
    "weather_source",
    "lighting_source",
    "camera_view_source",
    "traffic_density_source",
    "manual_review_status",
    "evidence",
]


def load_csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sequence_scene_metadata.csv for an external dataset from the current project metadata."
    )
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data_collection" / "reports" / "external_eda" / "sequence_scene_metadata.csv",
        help="Path to the current project's sequence_scene_metadata.csv",
    )
    parser.add_argument(
        "--external-splits",
        type=Path,
        default=Path(r"D:/UMT_EVIDENCE/dataset-v1-full/metadata/sequence_splits.csv"),
        help="Path to the external dataset's metadata/sequence_splits.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"D:/UMT_EVIDENCE/dataset-v1-full/metadata/sequence_scene_metadata.csv"),
        help="Path where the generated sequence_scene_metadata.csv will be written",
    )
    parser.add_argument(
        "--warn-missing",
        action="store_true",
        help="Print warnings for sequences that exist in external splits but are missing in current metadata",
    )
    args = parser.parse_args()

    source_rows = load_csv_dict(args.source_metadata)
    source_index: dict[tuple[str, str], dict[str, str]] = {
        (row.get("dataset_name", ""), row.get("sequence_id", "")): row
        for row in source_rows
    }

    external_splits = load_csv_dict(args.external_splits)
    joined_rows: list[dict[str, str]] = []
    missing: list[tuple[str, str]] = []

    for row in external_splits:
        dataset = row.get("dataset", "")
        sequence_id = row.get("sequence_id", "")
        key = (dataset, sequence_id)
        metadata_row = source_index.get(key)
        if metadata_row is not None:
            joined_rows.append({k: metadata_row.get(k, "") for k in HEADER})
        else:
            joined_rows.append(
                {
                    "dataset_name": dataset,
                    "sequence_id": sequence_id,
                    "road_type": "UNKNOWN",
                    "weather": "UNKNOWN",
                    "lighting": "UNKNOWN",
                    "camera_view": "UNKNOWN",
                    "traffic_density": "UNKNOWN",
                    "mean_vehicles_per_image": "",
                    "weather_source": "UNKNOWN",
                    "lighting_source": "UNKNOWN",
                    "camera_view_source": "UNKNOWN",
                    "traffic_density_source": "UNKNOWN",
                    "manual_review_status": "MISSING_METADATA",
                    "evidence": "",
                }
            )
            missing.append(key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(joined_rows)

    print(f"Wrote {len(joined_rows)} rows to {args.output}")
    if missing:
        print(f"{len(missing)} sequences were missing from source metadata.")
        if args.warn_missing:
            for dataset, sequence in missing:
                print(f"MISSING: {dataset}, {sequence}")


if __name__ == "__main__":
    main()
