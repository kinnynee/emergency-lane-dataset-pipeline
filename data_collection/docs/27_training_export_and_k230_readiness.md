# 27. Xuất dữ liệu train và kiểm tra K230

## AAU RainSnow

Data Lead đã xem lại 66 khung RGB, gồm các mốc 10%, 50% và 90% của đủ 22 sequence. Kết quả được ký xác nhận: `DAY=10`, `NIGHT=11`, `TWILIGHT=1`. Độ sáng trung bình chỉ còn là chỉ số chất lượng, không được dùng để quyết định nhãn ánh sáng.

Bằng chứng quyết định nằm tại:

- `configs/aau_sequence_lighting_review.yaml`
- `reports/external_eda/aau_lighting_data_lead_review.md`

## UA-DETRAC sang YOLO

Script `scripts/export_ua_detrac_yolo.py` tạo dataset YOLO một lớp `vehicle` theo split nguyên sequence. Mỗi box vượt biên được clip về kích thước ảnh trước khi chuẩn hóa; chỉ box malformed hoặc không còn diện tích sau clip mới bị loại. Metadata kèm theo giữ `original_class`, `track_id`, tọa độ trước/sau clip và quyết định loại.

Chạy toàn bộ archive vào một thư mục trống ngoài Git:

```powershell
python data_collection/scripts/export_ua_detrac_yolo.py --output D:\datasets\ua_detrac_yolo
```

Smoke test trên 200 ảnh thật đã xuất 605 box, trong đó 31 box được clip. Tất cả 200 cặp ảnh/label khớp stem và mọi tọa độ YOLO đều nằm trong `[0,1]`. Xem `reports/external_eda/ua_yolo_export_smoke_test.md`.

## K230 và bảng mAP

Điền từng buổi quay thật vào `planning/k230_evaluation_sessions.csv`, rồi chạy:

```powershell
python data_collection/scripts/validate_k230_evaluation_readiness.py --diagnostics data_collection/reports/external_eda/k230_readiness_diagnostics.csv
```

Một session chỉ được tính khi là `K230_SELF_RECORDED`, thuộc `MAIN_K230_TEST`, đã khóa, ground truth có trạng thái `APPROVED`, và ảnh/label ghép cặp đầy đủ. Internet hoặc dataset ngoài không được dùng thay K230. Script chỉ báo trạng thái sẵn sàng; nó không tự tạo mAP và không biến dữ liệu thiếu thành điểm 0.

Hiện cả `DAY`, `NIGHT`, `BACKLIT`, `RAIN` đều chưa có session K230 đạt điều kiện. Đây là phần phải quay và gán nhãn thực tế, không thể sửa hợp lệ bằng code.
