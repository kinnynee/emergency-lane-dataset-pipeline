# Unified YOLO export and condition frame extraction

## Release `dataset-v0.1` / `PILOT_500`

Đây là release train thử, cố định đúng **500 ảnh** từ manifest đã có: 170
MIO-TCD, 165 AAU RainSnow, 165 UA-DETRAC; split là 421 train, 51 validation,
28 cross-test. Cross-test này là dữ liệu external, **không** phải main K230
test đã khóa.

Đặt ba archive nguồn theo default path đã ghi trong `active_dataset_sources.yaml`
hoặc truyền từng `--*-path`, rồi chạy từ repo root:

```powershell
.\.venv\Scripts\python.exe data_collection/scripts/export_pilot_500.py --allow-proposal-selection
```

`--allow-proposal-selection` là bắt buộc ở v0.1 vì manifest hiện vẫn ghi
`selected=FALSE`; cờ này được lưu vào provenance. Export chạy trong staging,
chỉ publish `data_collection/dataset_output/dataset-v0.1/PILOT_500` sau khi
generic YOLO QC và PILOT_500 QC cùng `PASS`. Output gồm `images/`, `labels/`,
`data.yaml`, source/annotation metadata, manifest snapshot, và
`metadata/pilot_validation_report.json`.

Có thể kiểm tra lại độc lập:

```powershell
.\.venv\Scripts\python.exe data_collection/scripts/validate_pilot_500.py `
  --dataset data_collection/dataset_output/dataset-v0.1/PILOT_500
```

Smoke train một epoch để kiểm tra môi trường/model chạy được:

```powershell
.\.venv\Scripts\python.exe data_collection/scripts/train_yolov8_baseline.py --epochs 1
```

Sau smoke run, bỏ `--epochs 1` để chạy baseline 50 epoch đã cấu hình.

## Export external data

Place the approved raw sources outside Git or under the documented placeholder paths, then run from the repository root:

```powershell
py data_collection/scripts/export_unified_yolo.py `
  --mio-path <path-to-MIO-TCD-Localization.tar> `
  --aau-path <path-to-aau-rainsnow.zip-or-directory> `
  --ua-path <path-to-ua-detrac-orig.zip> `
  --output <path-to-dataset_v1>

py data_collection/scripts/validate_yolo_dataset.py --dataset <path-to-dataset_v1>
```

The exporter creates `images/train`, `images/val`, `images/cross_test`, the matching `labels` tree, `metadata/`, and `data.yaml`. It uses class ID `0` only, clips boundary-crossing boxes, preserves `original_class`, and writes all exclusions to `metadata/rejected_annotations.csv`. MIO-TCD is always train-only; AAU and UA-DETRAC sequences are read from `split_proposal.csv` and cannot cross splits.

Do not train unless the validator reports `PASS`. Its report reconciles input annotations with retained and rejected boxes, checks pairs, class IDs, normalized coordinates, non-empty boxes, and sequence leakage.

## Extract frames by verified condition

Copy `templates/condition_frame_manifest.csv`, replace the sample rows with reviewed videos, and preserve the session IDs. `BACKLIT` is accepted only with an explicit manual-review status; a bright daytime video is not a replacement.

```powershell
# Preview the videos and condition folders without writing files.
py data_collection/scripts/extract_condition_frames.py `
  --manifest <condition_frame_manifest.csv> `
  --input-root <video-root> `
  --output <new-frame-output-dir>

# Extract at 1 FPS after the preview is correct.
py data_collection/scripts/extract_condition_frames.py `
  --manifest <condition_frame_manifest.csv> `
  --input-root <video-root> `
  --output <new-frame-output-dir> `
  --fps 1 --execute
```

The output contains `day/`, `night/`, `rain/`, and `backlit/` folders when those reviewed conditions exist, plus `frame_manifest.csv` with video, session, timestamp, dimensions, and review basis. A NIGHT+RAIN video is represented in both folders without changing its source identity.

## Train the baseline

After QC succeeds, install the training environment and run the fixed YOLO11n baseline (`imgsz=320`, seed `230`, 50 epochs, batch `16`):

```powershell
py -m pip install -r data_collection/requirements-training.txt
py data_collection/scripts/train_yolov8_baseline.py --dataset <path-to-dataset_v1>
```

Each run saves `run_parameters.json`, the exact baseline config, and a copy of `data.yaml` with the Ultralytics artifacts.
