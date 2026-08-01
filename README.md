# Emergency Lane Dataset Pipeline

Data engineering and EDA pipeline for building a vehicle-detection dataset for the K230 emergency-lane monitoring project.

## Current status

As of 2026-08-01, the repository contains the collection plan, dataset governance documents, validation scripts, EDA summaries, quality audits, and proposed train/validation/cross-test splits.

External datasets currently in scope:

- **MIO-TCD Localization**: vehicle images and bounding boxes; currently train-only because the available metadata does not provide reliable sequence groups.
- **AAU RainSnow**: adverse-weather road sequences; proposed sequence split is 18 train / 2 validation / 2 cross-test sequences.
- **UA-DETRAC**: elevated traffic-camera sequences; proposed sequence split is 71 train / 21 validation / 8 cross-test sequences.
- **RADIATE is excluded** because its forward-facing camera viewpoint does not match the elevated K230 deployment viewpoint.

EDA currently covers 12,198 sampled images and 1,171,685 bounding boxes. The proposed cross-test set covers `HIGHWAY`, `INTERSECTION`, and `URBAN_ROAD`; `EMERGENCY_LANE_LIKE` is still missing. Quality gates remain `REVIEW_REQUIRED`, so the data is not yet marked train-ready.

All dataset selections and splits are `PROPOSAL_ONLY`. The final K230 holdout is still pending collection and must remain locked from training after it is created.

## Repository map

- [`data_collection/README.md`](data_collection/README.md): detailed workflow and directory guide.
- [`data_collection/configs/split_policy.yaml`](data_collection/configs/split_policy.yaml): leakage-safe sequence split policy.
- [`data_collection/docs/21_external_dataset_eda_methodology.md`](data_collection/docs/21_external_dataset_eda_methodology.md): EDA methodology.
- [`data_collection/docs/25_eda_distribution_quality_split.md`](data_collection/docs/25_eda_distribution_quality_split.md): distribution, quality, road-type, and split audit notes.
- [`data_collection/reports/external_eda/executive_summary.md`](data_collection/reports/external_eda/executive_summary.md): main EDA handoff summary.
- [`data_collection/reports/external_eda/quality_audit_summary.csv`](data_collection/reports/external_eda/quality_audit_summary.csv): dataset-level quality gates.
- [`data_collection/reports/external_eda/split_validation_summary.csv`](data_collection/reports/external_eda/split_validation_summary.csv): leakage and split validation results.
- [`data_collection/reports/external_eda/k230_holdout_plan.csv`](data_collection/reports/external_eda/k230_holdout_plan.csv): planned road-type coverage for the future K230 holdout.

## Quick verification

Tests and tracked summary reports do not require the large local datasets:

```powershell
python -m pip install -r data_collection/requirements-data.txt
python -m pytest data_collection/tests -q
```

## Reproduce the full EDA

Large images, videos, downloaded archives, extraction caches, and row-level EDA tables are intentionally excluded from Git. Place the source datasets under `data_collection/storage_placeholders/online_data/raw/`, following the paths documented in [`data_collection/README.md`](data_collection/README.md), then run:

```powershell
python data_collection/scripts/run_external_dataset_eda.py --sample-size 5000 --resume --skip-contact-sheets
```

Generated row-level files such as `bbox_samples.csv`, `image_quality_samples.csv`, `invalid_annotations.csv`, and `duplicate_groups.csv` are local artifacts. The compact aggregate reports needed for review are tracked in Git.

## Remaining decisions

- Collect and lock a representative K230 test set, including an `EMERGENCY_LANE_LIKE` road type.
- Review blurred/dark images, suspected duplicates, and invalid annotations listed by the local EDA queues.
- Confirm the final class mapping and split proposal with the Data Lead before training.
- Add real stopped-vehicle-in-ROI examples; the current external data does not provide sufficient ground truth for this target condition.
