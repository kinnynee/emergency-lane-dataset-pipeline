# Research findings — EDA ba dataset bên ngoài

## 1. Mục tiêu EDA

Đánh giá dữ liệu bổ sung cho YOLOv8n 320×320 và camera K230 cố định trên cao, không thay thế test thực địa K230.

## 2. Mô tả dataset

- **MIO-TCD Localization**: 137,743 ảnh/modality image, 0 video, 0 sequence, 351,549 bbox reported.
- **AAU RainSnow**: 4,396 ảnh/modality image, 44 video, 22 sequence, 13,297 bbox reported.
- **UA-DETRAC Original**: 140,131 ảnh/modality image, 0 video, 100 sequence, 1,274,055 bbox reported.

## 3. Lý do không sử dụng RADIATE

`EXCLUDED_VIEWPOINT_MISMATCH`: camera phía trước phương tiện khác camera cố định trên cao. Dữ liệu không bị xóa.

## 4. Quy trình

Inventory archive, parse annotation gốc, validation bbox, image-quality sample streaming, letterbox 320, duplicate sample, leakage theo sequence, viewpoint estimate và subset proposal. Không copy/xóa dữ liệu gốc.

Tổng 1.301.866 là **analysis-scope bbox sum**, không phải tổng full raw dataset: MIO dùng sample 5.000 ảnh, còn AAU và UA-DETRAC dùng phạm vi annotation rộng hơn/toàn bộ.

## 5. MIO-TCD Localization

Chỉ đọc TAR Localization; Classification bị chặn. MIO cung cấp localization ảnh tĩnh, không có Track ID và không chứng minh trạng thái dừng.

## 6. AAU RainSnow

Nhánh `aaurainsnow/` lặp được bỏ khỏi thống kê. Dataset có video RGB/thermal và COCO instance annotation; mưa/tuyết cụ thể theo sequence cần review vì metadata hiện có không tách rõ.

## 7. UA-DETRAC

Đọc toàn bộ XML train/test, Track ID, weather, camera state, occlusion/truncation. Stationary candidate là heuristic và luôn `manual_review_status=PENDING`.

## 8–10. Góc camera, điều kiện và class

Viewpoint cao nhất theo rule hiện tại: **AAU RainSnow 4.31/5**. AAU bổ sung adverse weather; UA hỗ trợ tracking; MIO/AAU có lớp xe máy, còn UA-DETRAC không quan sát thấy xe máy trong class XML đã đọc.

Cross-dataset test proposal theo road type: **HIGHWAY=2, INTERSECTION=5, URBAN_ROAD=3**. Chưa có sequence được xác nhận là `EMERGENCY_LANE_LIKE`; metadata cảnh vẫn chờ Data Lead duyệt.

Metadata cảnh cross-test: **camera_view: ELEVATED_OBLIQUE=10; lighting: DAY=5, NIGHT=2, TWILIGHT=3; traffic_density: HIGH=4, LOW=2, MEDIUM=4; weather: CLEAR=3, CLOUDY=2, RAIN_OR_WET_ROAD=2, UNKNOWN=3**. `weather=UNKNOWN` được giữ lại khi XML chỉ ghi `night`, vì `night` là điều kiện ánh sáng chứ không phải thời tiết. Mật độ xe được tính bằng số bbox phương tiện trung bình trên mỗi ảnh có annotation.

Phân bố chi tiết theo class và kích thước bbox được xuất ở `class_distribution_by_scene.csv` và `bbox_distribution_by_scene.csv`; số đếm cấp sequence nằm trong `cross_test_sequence_statistics.csv`. Các tỷ lệ theo scene dùng đúng phạm vi bbox analysis sample và không được trình bày như toàn bộ dataset.

## 11–12. Bounding box và resize 320×320

Tỷ lệ box dưới 8 px theo dataset: MIO-TCD Localization=0.13741649, AAU RainSnow=0.37476683, UA-DETRAC Original=0.06013477. Đây là ngưỡng phân tích ban đầu, không phải ngưỡng ground truth.

## 13–15. Chất lượng ảnh, annotation, trùng và leakage

Đã kiểm tra 12,198 ảnh/frame mẫu; ghi 1,503 annotation lỗi duy nhất (1,503 issue). Duplicate scan chỉ áp dụng trên mẫu đã đọc ảnh. Phát hiện 0 leakage CRITICAL theo sequence metadata.

Quality gate hiện tại: **REVIEW_REQUIRED=3**. Pipeline đã sửa **0** bbox sample theo `vehicle_class_mapping.yaml`. Class policy đã chốt; chỉ `UA-DETRAC:others` tiếp tục được lấy mẫu review. Hàng đợi hành động nằm tại `quality_review_queue.csv`.

Với `UA-DETRAC:others`, 60 mẫu được chọn phân tầng từ 48 sequence. Pre-review ghi nhận 48 xe chắc chắn, 9 khả năng là xe, 2 non-vehicle và 1 chưa xác định. Các quyết định cấp mẫu nằm trong `ua_others_stratified_review_queue.csv` và vẫn cần Data Lead ký xác nhận; không dùng tỷ lệ mẫu để khẳng định toàn bộ 20.641 annotation đều là xe.

## 16–17. Vehicle detection và giới hạn xe dừng

Ba bộ hỗ trợ nhận diện phương tiện. Không bộ nào có ground-truth xe dừng trong ROI. Không được kết luận xe dừng từ một ảnh.

## 18. Khoảng trống so với K230

Thiếu ROI làn khẩn cấp, thời gian dừng 2–3 giây/>5 giây, xe rời ROI, ngược sáng/đèn pha đã xác minh và domain camera K230 tại trường.

## 19–20. Subset

PILOT_500 và DATASET_V1_1500 được chia gần cân bằng giữa ba nguồn, theo sequence và chỉ là proposal. Xem `balanced_subset_plan.csv` và `selected_data_manifest.csv`.

## 21. Đề xuất thu K230

Quay theo sequence độc lập: ngày/đêm/mưa/đường ướt/ngược sáng/đèn pha; có xe chạy, chậm, dừng, rời ROI và negative. Main test khóa theo session/video.

## 22. Kết luận

Dùng dữ liệu ngoài cho train, validation sequence-level và cross-domain test. Main project test không được chỉ dùng ba dataset ngoài.

Split validation: **PARTIAL=1, PASS=8, PENDING=2**. MIO-TCD không có sequence/session tin cậy nên chỉ được đề xuất vào `EXTERNAL_TRAIN`; test chính `MAIN_K230_TEST` vẫn là placeholder chờ thu và khóa manifest.

## 23. Nguồn

- MIO-TCD: https://tcd.miovision.com/
- AAU RainSnow: https://www.kaggle.com/datasets/aalborguniversity/aau-rainsnow
- UA-DETRAC original dataset name; Kaggle download mirror: https://www.kaggle.com/datasets/bratjay/ua-detrac-orig
