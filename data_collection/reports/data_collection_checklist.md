# LDKC-14 — Checklist thu thập dữ liệu K230

Biểu mẫu hiện có ở [`LDKC-14/README.md`](../field_collection/LDKC-14/README.md), [`video_manifest.csv`](../field_collection/LDKC-14/video_manifest.csv), [`su_kien.csv`](../field_collection/LDKC-14/su_kien.csv), [`khung_bao.csv`](../field_collection/LDKC-14/khung_bao.csv). Ba CSV chỉ có header, chưa có video gốc, ROI chính thức hoặc sự kiện thực. [`k230_evaluation_sessions.csv`](../planning/k230_evaluation_sessions.csv) vẫn là các dòng `MISSING_DATA`; không ghi là đã thu thập. Dữ liệu này hướng tới bản sửa 20/12.

## Trước khi quay

- [ ] Chọn vị trí/camera được phép, cố định góc và bảo đảm an toàn, riêng tư. Lưu `camera_id`, `location_id`, quyền ghi hình và ảnh góc lắp; xem [khảo sát vị trí](../docs/03_camera_location_survey.md).
- [ ] Xác nhận **cách cấp nguồn USB-C của đúng board K230 đang dùng** bằng tài liệu và phép thử có kiểm soát; `MISSING` model board, yêu cầu điện áp/dòng, nguồn và log ổn định. Không ghi nguồn USB-C là đã được kiểm thử.
- [ ] Xác nhận khả năng ghi video thực tế, codec, kích thước, timestamp và FPS bằng video thử từ chính camera; `MISSING` video thử/FPS đo. Lưu FPS khai báo và FPS tính từ timestamp; kiểm tra frame drop/biến thiên FPS.
- [ ] Tạo ROI JSON cho làn khẩn cấp và vùng loại trừ trên **đúng khung hình camera**; lưu `camera_id`, kích thước ảnh, phiên bản, người duyệt và ngày duyệt. [`CALIBRATION_AND_ROI.md`](../../CALIBRATION_AND_ROI.md) yêu cầu ROI đã hiệu chuẩn; `roi_nguoive_v2.json` mẫu chưa có trong workspace nên `MISSING` schema chính thức.
- [ ] Tạo thư mục ngoài Git theo `location_id/camera_id/session_id/raw`, `roi`, `annotations`, `review`, `checksums`; lưu video gốc bất biến và SHA-256. Đây là cấu trúc đề xuất để triển khai, chưa phải thư mục đã tồn tại.

## Khi quay và ghi nhãn

- [ ] Đặt tên video theo [`06_naming_convention.md`](../docs/06_naming_convention.md): `YYYYMMDD_location_device_condition_session_sequence.mp4`. Mỗi video ghi thời gian bắt đầu/kết thúc kèm múi giờ, camera, địa điểm, thời tiết, điều kiện sáng, kích thước, FPS đo, duration media, hash và ROI version. [`video_manifest.csv`](../field_collection/LDKC-14/video_manifest.csv) hiện thiếu cột `camera_id`, `location_id`, `session_id`, `thoi_gian_ket_thuc`, `timezone`, `sha256`, `fps_measured`, `roi_version`; bổ sung vào manifest vận hành trước khi quay.
- [ ] Ghi `khung_bat_dau`, `khung_ket_thuc`, mốc thời gian thực bắt đầu/kết thúc sự kiện, `id_xe`, `trong_lan`, điều kiện và review status. [`su_kien.csv`](../field_collection/LDKC-14/su_kien.csv) hiện chưa có cột timestamp; chỉ số frame/FPS không đủ cho video FPS biến thiên.
- [ ] Mỗi sự kiện lưu ít nhất ba bbox đầu/giữa/cuối trong [`khung_bao.csv`](../field_collection/LDKC-14/khung_bao.csv); đối chiếu với ROI, video gốc và người duyệt.
- [ ] Thu **ít nhất 60 sự kiện dừng ≥10 giây**, có sự kiện 30–120 giây; trong đó ít nhất 20 ban ngày và 20 ban đêm. Thu **ít nhất 5 giờ** giao thông bình thường đã xem hết, không có xe dừng trong làn. Đây là mục tiêu, chưa có thành tích thực tế.
- [ ] Chỉ ghi tình huống tự nhiên trên làn khẩn cấp đang khai thác. Nếu cần dàn dựng, chỉ làm ở khu thử nghiệm được phép và kiểm soát giao thông; không cố ý bố trí xe dừng trong làn khẩn cấp đang hoạt động.

## Sau khi quay

- [ ] Xác minh checksum, timestamp, FPS, số frame, ROI, ranh giới sự kiện và nhãn ngày/đêm; tách test theo session/video/camera trước khi dùng ảnh để train.
- [ ] Chạy [`LDKC-14/validate.py`](../field_collection/LDKC-14/validate.py) sau khi có dữ liệu và xem thủ công video. Hiện validator tính duration sự kiện bằng `(end-start+1)/fps`, vì vậy cần bổ sung kiểm tra timestamp cho video FPS biến thiên trước khi dùng duration để xác nhận ngưỡng 10 giây.
- [ ] Khóa manifest test và lưu log chạy trên board nếu công bố FPS, latency hoặc hiệu năng K230. Video test phải nằm ngoài train.
