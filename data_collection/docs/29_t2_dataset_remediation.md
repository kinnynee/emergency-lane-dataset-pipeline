# T2 dataset remediation and evaluation gate

## Release status

`dataset-v1-full` is not an evaluation-valid release until the T2 remediation
has been applied and its report has passed.  In particular, do not publish a
headline mAP, recall, or slice metric that includes `UPX_*` pseudo-labels in
`val` or `cross_test`.

Run the evidence-only audit first:

```powershell
python data_collection/scripts/remediate_t2_dataset.py `
  --dataset "D:\UMT_EVIDENCE\dataset-v1-full" `
  --report data_collection/reports/t2_dataset_remediation_plan.json
```

The expected production plan has 84 UPX images in `val`, 81 in
`cross_test`, and 66 unregistered `ONLINE_HW_DAY_*` images.  Any other count
is a stop condition: inspect the report and regenerate the plan rather than
applying an old one.

After review, the in-place remediation is intentionally explicit:

```powershell
python data_collection/scripts/remediate_t2_dataset.py `
  --dataset "D:\UMT_EVIDENCE\dataset-v1-full" `
  --apply --confirm-apply-t2
```

The command creates metadata backups, moves the 165 UPX evaluation image/label
pairs to `review_pending/t2_upx_auto_label_evaluation_20260816`, and leaves
the original release media recoverable.  The default `data.yaml` continues to
use only `images/train`, `images/val`, and `images/cross_test`, so the
quarantined files cannot be selected by an ordinary YOLO validation run.

## Label and mapping policy

`UPX_*` is recorded as `AUTO_YOLO11N_COCO_2026_08_09`.  It is a pseudo-label,
not a human-reviewed ground-truth label.  The promotion evidence identifies a
COCO motor-vehicle aggregation and no recorded human reviewer, approval, or
cross-check.  Its inference confidence threshold is not recorded in the
available batch metadata; min/max confidences in output chunks are not a
replacement for that threshold.

The car-only release policy excludes `motorcycle`, `motorbike`, and `bicycle`
from positive class 0 and requires them to be handled as ignore regions.  A
COCO `motor_vehicle` aggregate is therefore ambiguous for this policy.  UPX
may only re-enter training after either a class-preserving re-export that
creates ignore regions, or documented human review.  It must not re-enter an
evaluation split with auto-labels.

`configs/vehicle_class_mapping.yaml` now encodes this as
`online_auto_label.COCO_motor_vehicle = QUARANTINE_PENDING_CLASS_PRESERVING_REEXPORT`;
future ingestion must fail closed instead of mapping that aggregate directly
to class 0.

The remediation adds these fields to `metadata/images.csv`:

| Field | Meaning |
| --- | --- |
| `label_source` | Origin of the label, including the exact UPX generator identifier. |
| `label_review_status` | What review evidence exists; it never implies review when none was recorded. |
| `evaluation_eligibility` | Whether the item can influence evaluation. |
| `provenance_status` | Whether source/permission evidence is complete. |
| `remediation_note` | Human-readable reason and release handling. |

Use `exported_image` as the join key for this version.  `image_id` is not a
safe filename key for legacy `ONLINE_*` and `UPX_*` records.

## ONLINE_HW backfill

The T2 auditor confirmed the missing frames are all accompanied by YOLO label
files.  Their scenes must be registered from the existing sequence metadata
as highway day footage, but their source, licence, and label review are set to
pending rather than inferred.  The audit captures the exact filenames, image
dimensions, and label counts in
`metadata/t2_online_hw_metadata_backfill.csv`.

Visual review found genuine highway scenes in the batch, including a highway
bridge, a front-dashcam motorway sequence, and a night aerial motorway
sequence.  Some sampled frames are near-black or have no usable roadway view;
these need image-quality review before any K230 test seed is created.

## Coverage and documentation

- MIO-TCD is intentionally train-only in the current release.  Validation
  metrics therefore do not measure the source that supplies more than half of
  the training images; this limitation must be stated with every result.
- The release records five sources in practice: MIO-TCD, UA-DETRAC, AAU
  RainSnow, legacy ONLINE_DATA, and user-provided Pexels/Mixkit video.
- Pexels' current licence allows free use and modification, but prohibits,
  among other things, redistribution of unaltered media on competing stock or
  wallpaper services.  Record the source URL, downloader/date, author where
  supplied, licence snapshot URL, and the particular video ID before a dataset
  redistribution decision.  See the official [Pexels licence](https://www.pexels.com/license/).
- The automated outlier-box queue is review-only.  Do not remove labels merely
  because one box contains several others; buses seen through windows are a
  known false-positive pattern for that heuristic.

Generate a review queue (for the known AAU validation issue first) with:

```powershell
python data_collection/scripts/qc_outlier_boxes.py `
  --dataset "D:\UMT_EVIDENCE\dataset-v1-full" `
  --splits val --prefix AAU_ `
  --output data_collection/reports/t2_aau_outlier_box_review.csv
```
