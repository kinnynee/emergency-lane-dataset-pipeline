# Script hỗ trợ dữ liệu

## Mục lục
- [Cài đặt](#cài-đặt)
- [Cách chạy](#cách-chạy)
- [Lưu ý](#lưu-ý)

## Cài đặt
Từ thư mục `data_collection`:

```text
python -m pip install -r requirements-data.txt
```

## Cách chạy
```text
python scripts/create_dataset_folders.py --dry-run
python scripts/create_dataset_folders.py
python scripts/validate_metadata.py
python scripts/check_image_label_pairs.py
python scripts/generate_dataset_summary.py
python scripts/run_online_data_pipeline.py --dry-run
```

`check_image_label_pairs.py` mặc định đọc `dataset_output/images` và `dataset_output/labels`, ghi `reports/image_label_pair_report.csv`. Có thể dùng `--images`, `--labels`, `--output`.

## Lưu ý
- Script không xóa/ghi đè media.
- Đường dẫn được hiểu tương đối với `data_collection`.
- Exit code khác 0 của validator/pair checker cho biết cần sửa dữ liệu.

## Pipeline dữ liệu Internet

```text
python scripts/search_open_sources.py
python scripts/validate_source_license.py
python scripts/run_online_data_pipeline.py --dry-run
python scripts/generate_online_data_report.py
```

Chỉ dùng `--execute` khi reviewer đã đưa source vào trạng thái `APPROVED_FOR_DOWNLOAD`, hàng queue là `APPROVED`, có URL trực tiếp chính thức, `license_verified=TRUE` và permission `APPROVED`/`NOT_REQUIRED`. Không dùng script này để tải YouTube, bypass đăng nhập/paywall hoặc dùng cookie cá nhân.
