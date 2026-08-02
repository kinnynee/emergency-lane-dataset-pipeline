# 26. Theo dõi sửa góp ý giảng viên ngày 02/08/2026

| Góp ý | Thay đổi đã thực hiện | Bằng chứng |
|---|---|---|
| Thiếu dữ liệu ngược sáng | Thêm slice `K230_BACKLIT` và hàng mAP `BACKLIT`; trạng thái hiện là thiếu dữ liệu, không điền số giả | `configs/split_policy.yaml`, `evaluation_slice_readiness.csv` |
| Nhãn ánh sáng AAU sai do brightness | Bỏ brightness khỏi quyết định nhãn; review trực quan 3 khung RGB của đủ 22 sequence | `configs/aau_sequence_lighting_review.yaml` |
| Box UA-DETRAC vượt biên không phải lỗi | Clip về biên ảnh và giữ xe; chỉ loại box không hợp lệ sau clip | `inspect_ua_detrac.py`, `quality_audit_summary.csv` |
| Gộp một lớp vehicle | Giữ mọi xe cơ giới kể cả xe máy; bỏ người/xe đạp; giữ nhãn gốc; `others` giữ có điều kiện và review phân tầng 60 mẫu | `vehicle_class_mapping.yaml`, `ua_others_stratified_review_queue.csv` |

## Việc còn phải thu ngoài thực địa

- Quay K230 ngược sáng thành session độc lập, dùng góc cao đúng vị trí triển khai.
- Thu đủ `DAY`, `NIGHT`, `BACKLIT`, `RAIN` và khóa toàn bộ session trong `MAIN_K230_TEST`.
- Sau khi có model, tính mAP riêng từng slice; hiện không có số `BACKLIT` hợp lệ để báo cáo.
