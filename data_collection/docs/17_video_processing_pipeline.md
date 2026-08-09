# 17. Pipeline xử lý video online

## Mục lục
- [Luồng](#luồng)
- [Lưu trữ](#lưu-trữ)
- [Công cụ](#công-cụ)

## Luồng
Validate source/license → queue approved → tải an toàn → inspect → normalize không phá gốc → checksum → nhóm trùng → tách frame → near-duplicate → contact sheet → metadata/EDA/daily report. Mặc định pipeline là `--dry-run`; cần `--execute` mới xử lý/tải thật.

### Sequence diagram
```mermaid
sequenceDiagram
    autonumber
    actor Operator as Operator
    participant Policy as Source Policy
    participant Queue as Approved Queue
    participant Downloader as Safe Downloader
    participant Inspector as Inspector
    participant Storage as Storage
    participant Normalizer as Normalizer
    participant Duplicate as Duplicate Detector
    participant Extractor as Frame Extractor
    participant EDA as Metadata/EDA
    participant Report as Daily Report

    Operator->>Policy: Review source/license
    Policy-->>Queue: Approve candidate

    alt Dry run
        Queue-->>Operator: Validate only, no download
    else Execute mode
        Operator->>Downloader: Run with --execute
        Downloader->>Storage: Download raw source and checksum manifest
        Operator->>Inspector: Inspect media quality and metadata
        Inspector->>Storage: Mark approved / rejected / quarantine
        Storage->>Normalizer: Normalize without altering original
        Normalizer->>Storage: Save processed copy
        Storage->>Duplicate: Group near-duplicates
        Duplicate-->>Storage: Duplicate groups
        Storage->>Extractor: Extract frames and contact sheets
        Extractor->>Storage: Save frames and previews
        Extractor->>EDA: Emit metadata and EDA inputs
        EDA->>Report: Generate summary and daily report
        Report-->>Operator: Deliver review artifacts
    end
```

## Lưu trữ
Gốc: `storage_placeholders/online_data/raw/`; bản chuẩn: `processed/`; lỗi: `quarantine/`; rejected, frames và contact sheets tách riêng. Các thư mục chỉ giữ `.gitkeep` trong Git. Tên gốc `src_<source_id>_<item_id>_<original_name>.<ext>`; processed: `YYYYMMDD_<source_type>_<camera_type>_<condition>_<source_id>_<sequence>.mp4`, kèm JSON cùng stem.

## Công cụ
`ffprobe`/`ffmpeg` là tùy chọn. Nếu thiếu, script báo hướng dẫn cài rõ ràng; validate/report vẫn chạy. Không tự tải công cụ không rõ nguồn.
