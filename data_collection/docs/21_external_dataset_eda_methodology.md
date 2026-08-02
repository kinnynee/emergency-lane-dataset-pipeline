# Phương pháp EDA dataset bên ngoài

## Phạm vi

Chỉ phân tích:

1. MIO-TCD Localization.
2. AAU RainSnow.
3. UA-DETRAC Original.

RADIATE là `EXCLUDED_VIEWPOINT_MISMATCH`. MIO-TCD Classification không được đưa vào discovery, parser hoặc báo cáo.

## Nguyên tắc

- Dữ liệu gốc chỉ đọc; không xóa, di chuyển, sửa hoặc tự áp dụng split.
- Ảnh được giải mã từng file, không giữ batch ảnh lớn trong RAM.
- Inventory và annotation được quét đầy đủ khi parser hỗ trợ; chất lượng pixel dùng `--sample-size` hoặc `--full-scan`.
- Brightness, blur và viewpoint tự động chỉ là chỉ số chất lượng, ghi `AUTOMATIC_ESTIMATE`; không dùng brightness để gán trục ánh sáng.
- `DAY/NIGHT/TWILIGHT` của AAU lấy từ review trực quan ba khung RGB trên đủ 22 sequence.
- BBox được kiểm tra biên, kích thước, NaN/Infinity và ước lượng letterbox 320×320.
- `STATIONARY_CANDIDATE` của UA-DETRAC chỉ là heuristic từ Track ID và độ dịch chuyển tâm box; không phải ground truth.
- Duplicate scan không xóa file. Leakage được kiểm tra theo sequence trước frame.

## Cách chạy

PowerShell chạy nhanh, một dòng:

```powershell
python data_collection/scripts/run_external_dataset_eda.py --sample-size 5000
```

PowerShell chạy toàn bộ pixel nếu tài nguyên cho phép:

```powershell
python data_collection/scripts/run_external_dataset_eda.py --full-scan --resume --workers 4
```

Chỉ tạo lại executive summary từ CSV:

```powershell
python data_collection/scripts/run_external_dataset_eda.py --report-only
```

Có thể truyền `--mio-path`, `--aau-path`, `--uadetrac-path`. Nếu thiếu một bộ, pipeline ghi `NOT_FOUND` và tiếp tục.

## Hạn chế

Ngưỡng ảnh và bbox là ngưỡng nghiên cứu ban đầu. Điều kiện mưa/tuyết của AAU không được tự suy ra chính xác theo sequence nếu metadata không tách rõ. Test chính phải dùng K230 tự quay.
