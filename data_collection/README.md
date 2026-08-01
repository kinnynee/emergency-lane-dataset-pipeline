# Bộ khung quản lý dữ liệu camera K230

## Mục lục
- [Mục tiêu và phạm vi](#mục-tiêu-và-phạm-vi)
- [Quy trình](#quy-trình)
- [Cấu trúc và lưu trữ](#cấu-trúc-và-lưu-trữ)
- [Cách sử dụng](#cách-sử-dụng)
- [Definition of Done](#definition-of-done)
- [Mốc 12 tuần](#mốc-12-tuần)

## Mục tiêu và phạm vi
Bộ khung phục vụ thu thập, truy vết, gán nhãn và kiểm soát chất lượng dữ liệu cho hệ thống cảnh báo xe dừng trong ROI bằng K230 và YOLOv8. Lớp mặc định là `vehicle` (ID 0). Cần bao phủ xe có/không có trong ROI, chạy xuyên, dừng/rời ROI, nhiều khoảng cách, ngày, đêm, mưa/đường ướt, ngược sáng và mẫu âm.

Nguồn ưu tiên: (1) K230 tại vị trí demo; (2) dashcam do nhóm tự quay; (3) dữ liệu mở/Internet có quyền rõ ràng; (4) sa bàn và dữ liệu BKU đã xác minh. Internet chỉ bổ sung, không thay thế góc nhìn triển khai thật.

## Quy trình
1. Xin phép và khảo sát vị trí; giảng viên xác nhận vị trí cuối.
2. Lập lịch, kịch bản, mã phiên và kiểm tra an toàn.
3. Quay, kiểm tra file, ghi metadata và sao lưu ngoài Git.
4. Tách frame, lọc ảnh hỏng/trùng/gần trùng.
5. Gán nhãn YOLO, kiểm tra chéo 10-20%.
6. Chia theo `video_id` hoặc `session_id`; khóa test.
7. Chạy EDA/QC, lập dataset card và changelog.
8. Phát hành phiên bản kèm evidence; chưa có evidence thì giữ `TODO`/`CHƯA THỰC HIỆN`.

## Cấu trúc và lưu trữ
- `configs/`: cấu hình dataset, điều kiện thu và quy tắc nhãn.
- `docs/`: chính sách và hướng dẫn.
- `planning/`: kế hoạch, tiến độ, nguồn, lỗi và evidence.
- `metadata/`: chỉ mục phiên/video/ảnh/nhãn.
- `templates/`: biểu mẫu báo cáo và phát hành.
- `scripts/`: công cụ tạo thư mục, kiểm tra và thống kê.
- `storage_placeholders/`: vị trí minh họa; chỉ có `.gitkeep`.

Dữ liệu thật lưu trên Google Drive hoặc ổ cứng ngoài. Trong CSV chỉ ghi đường dẫn tương đối với `storage_root` hoặc liên kết Drive được phân quyền. Không commit media, dataset, model, token, mật khẩu hay thông tin cá nhân. Ảnh công khai phải làm mờ biển số và khuôn mặt.

Tên video: `YYYYMMDD_location_device_condition_session_sequence.mp4`; ảnh: `YYYYMMDD_location_device_condition_session_frame.jpg`; nhãn trùng stem ảnh. Tên chữ thường, không dấu, không khoảng trắng. Session: `SYYYYMMDD_XXX`.

## Cách sử dụng
1. Sửa `configs/dataset_config.yaml`, nhất là `storage_root` và `created_by`.
2. Điền bảng trong `planning/`; không đổi tên cột. Trạng thái chưa làm là `NOT_STARTED`.
3. Mỗi buổi quay thêm một dòng `metadata/sessions.csv`, rồi liên kết video, ảnh và nhãn bằng ID.
4. Cài: `python -m pip install -r requirements-data.txt`.
5. Chạy từ thư mục `data_collection`:
   - `python scripts/create_dataset_folders.py --dry-run`
   - `python scripts/create_dataset_folders.py`
   - `python scripts/validate_metadata.py`
   - `python scripts/check_image_label_pairs.py`
   - `python scripts/generate_dataset_summary.py`
6. Chạy EDA ba dataset ngoài từ thư mục repo:
   - `python data_collection/scripts/run_external_dataset_eda.py --sample-size 5000 --resume --skip-contact-sheets`
   - Xem `reports/external_eda/quality_review_queue.csv` và `split_validation_summary.csv` trước khi áp dụng subset/split.

## Definition of Done
### Một buổi thu dữ liệu
- [ ] Có quyền ghi hình, vị trí và thiết bị an toàn.
- [ ] Video mở được, đúng tên, đã sao lưu ngoài Git.
- [ ] Session/video metadata đầy đủ và liên kết đúng.
- [ ] Evidence có trong registry; dữ liệu công khai đã ẩn danh.
- [ ] Không tuyên bố `DONE` khi chưa có evidence.

### Một phiên bản dataset
- [ ] Ảnh/nhãn ghép cặp hợp lệ; nhãn đã review chéo.
- [ ] Split theo session/video, không rò rỉ frame gần nhau.
- [ ] Test đã khóa, không dùng train; nguồn và quyền truy vết được.
- [ ] EDA, dataset card, changelog và evidence hoàn thành.
- [ ] Version, checksum/đường dẫn lưu trữ được ghi nhận.

## Mốc 12 tuần
- Tuần 1 (27-31/07/2026): spec và quy trình.
- Tuần 3 (10-14/08/2026): dataset-v0 khoảng 500 ảnh.
- Tuần 5 (24-28/08/2026): dataset-v1 tối thiểu 1.500 ảnh; test tối thiểu 300 ảnh và khóa cứng.
- Tuần 6-10: hỗ trợ model, đo theo điều kiện và hai chiến dịch thực nghiệm.
- Tuần 11-12 (đến 18/10/2026): báo cáo, video minh chứng, bài báo và hồ sơ.
