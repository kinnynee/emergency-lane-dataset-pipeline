# Emergency Lane Dataset Pipeline

Data engineering and EDA pipeline for building a vehicle-detection dataset for the K230 emergency-lane monitoring project.

## Current status

As of 2026-08-02, the repository contains the collection plan, dataset governance documents, validation scripts, EDA summaries, quality audits, and proposed train/validation/cross-test splits.

External datasets currently in scope:

- **MIO-TCD Localization**: vehicle images and bounding boxes; currently train-only because the available metadata does not provide reliable sequence groups.
- **AAU RainSnow**: adverse-weather road sequences; proposed sequence split is 18 train / 2 validation / 2 cross-test sequences.
- **UA-DETRAC**: elevated traffic-camera sequences; proposed sequence split is 71 train / 21 validation / 8 cross-test sequences.
- **RADIATE is excluded** because its forward-facing camera viewpoint does not match the elevated K230 deployment viewpoint.

EDA currently covers 12,198 sampled images and an **analysis-scope sum** of 1,301,866 bounding boxes. This is not a full-raw-data total: MIO uses a deterministic 5,000-image sample, while AAU and UA-DETRAC use broader/full annotation scopes. AAU lighting was manually reviewed and signed off by the acting Data Lead for all 22 sequences (`DAY=10`, `NIGHT=11`, `TWILIGHT=1`). UA-DETRAC has 130,181 right/bottom boundary overruns that are clipped and kept; the observed at-most-one-pixel pattern is documented as a coordinate-convention hypothesis, not as proof that vehicles enter or leave the frame. A real-media YOLO smoke export verified 200 image/label pairs and retained 31 clipped boxes. The proposed cross-test set covers `HIGHWAY`, `INTERSECTION`, and `URBAN_ROAD`; `EMERGENCY_LANE_LIKE` and the dedicated K230 `BACKLIT` recording are still missing. Quality gates remain `REVIEW_REQUIRED`, so the data is not yet marked train-ready.

`UA-DETRAC:others` is approved for mapping to `vehicle` with `original_class` preserved. Data Lead review covered all 74 unique `others` tracks: 73 are motor vehicles, while `MVI_40172 / track 79` is a non-vehicle roadside structure. The rejected track contains 201 boxes and must be excluded during export; see `ua_others_data_lead_review.md` and `ua_others_track_exclusions.csv`.

`K230_BACKLIT` remains `NOT_AVAILABLE` until real elevated-camera data, reviewed ground truth, and predictions exist. The dedicated acquisition and split-lock requirements are in `data_collection/planning/k230_backlit_collection_protocol.csv`; missing data is never converted to a zero score.

All dataset selections and splits are `PROPOSAL_ONLY`. The final K230 holdout is still pending collection and must remain locked from training after it is created.

## Repository map

- [`data_collection/README.md`](data_collection/README.md): detailed workflow and directory guide.
- [`data_collection/configs/split_policy.yaml`](data_collection/configs/split_policy.yaml): leakage-safe sequence split policy.
- [`data_collection/docs/21_external_dataset_eda_methodology.md`](data_collection/docs/21_external_dataset_eda_methodology.md): EDA methodology.
- [`data_collection/docs/25_eda_distribution_quality_split.md`](data_collection/docs/25_eda_distribution_quality_split.md): distribution, quality, road-type, and split audit notes.
- [`data_collection/docs/26_supervisor_feedback_corrections.md`](data_collection/docs/26_supervisor_feedback_corrections.md): corrections for lighting labels, boundary boxes, class mapping, and the missing backlit slice.
- [`data_collection/docs/27_training_export_and_k230_readiness.md`](data_collection/docs/27_training_export_and_k230_readiness.md): production YOLO export and K230 slice-readiness workflow.
- [`data_collection/docs/28_unified_export_and_condition_frames.md`](data_collection/docs/28_unified_export_and_condition_frames.md): unified MIO/AAU/UA YOLO export, QC, condition-frame extraction, and YOLO11n-320 baseline commands.
- [`data_collection/reports/external_eda/executive_summary.md`](data_collection/reports/external_eda/executive_summary.md): main EDA handoff summary.
- [`data_collection/reports/external_eda/quality_audit_summary.csv`](data_collection/reports/external_eda/quality_audit_summary.csv): dataset-level quality gates.
- [`data_collection/reports/external_eda/split_validation_summary.csv`](data_collection/reports/external_eda/split_validation_summary.csv): leakage and split validation results.
- [`data_collection/reports/external_eda/k230_holdout_plan.csv`](data_collection/reports/external_eda/k230_holdout_plan.csv): planned road-type coverage for the future K230 holdout.
- [`data_collection/reports/external_eda/evaluation_slice_readiness.csv`](data_collection/reports/external_eda/evaluation_slice_readiness.csv): readiness of the required DAY/NIGHT/BACKLIT/RAIN mAP slices.
- [`data_collection/reports/external_eda/ua_yolo_export_smoke_test.md`](data_collection/reports/external_eda/ua_yolo_export_smoke_test.md): real-media evidence that clipped boxes reach YOLO labels.

## Quick verification

Tests and tracked summary reports do not require the large local datasets:

```powershell
python -m pip install -r data_collection/requirements-data-lock.txt
python -m pytest data_collection/tests -q
```

`requirements-data-lock.txt` ghi đúng phiên bản môi trường Windows/Python 3.12 đã dùng cho lần EDA được kiểm chứng. File requirements dạng khoảng phiên bản vẫn dùng cho phát triển; tái lập báo cáo nghiên cứu nên dùng lock.

## Reproduce the full EDA

Large images, videos, downloaded archives, extraction caches, and row-level EDA tables are intentionally excluded from Git. Place the source datasets under `data_collection/storage_placeholders/online_data/raw/`, following the paths documented in [`data_collection/README.md`](data_collection/README.md), then run:

```powershell
python data_collection/scripts/run_external_dataset_eda.py --sample-size 5000 --resume --skip-contact-sheets
```

Generated row-level files such as `bbox_samples.csv`, `image_quality_samples.csv`, `invalid_annotations.csv`, and `duplicate_groups.csv` are local artifacts. The compact aggregate reports needed for review are tracked in Git.

## Remaining decisions

- Collect and lock a representative K230 test set, including an `EMERGENCY_LANE_LIKE` road type.
- Review blurred/dark images, suspected duplicates, and malformed annotations listed by the EDA queues.
- Approve the final split proposal before training; AAU lighting and UA-DETRAC `others` reviews are already signed off.
- Add real stopped-vehicle-in-ROI examples; the current external data does not provide sufficient ground truth for this target condition.
