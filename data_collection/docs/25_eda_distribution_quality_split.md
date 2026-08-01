# Hoàn thiện EDA: phân bố, chất lượng và chia tập

Ngày chạy: 01/08/2026. Phạm vi gồm MIO-TCD Localization, AAU RainSnow và UA-DETRAC Original. RADIATE vẫn bị loại do sai góc camera.

## 1. Phân bố dữ liệu

Phân bố tổng theo dataset và class nằm trong `dataset_inventory.csv`, `class_distribution.csv` và `bbox_statistics.csv`. Phân tích theo metadata cảnh của 10 sequence cross-test được xuất riêng:

- `cross_test_sequence_statistics.csv`: số ảnh, số bbox, mật độ và tỷ lệ bbox khó của từng sequence.
- `class_distribution_by_scene.csv`: original class/mapped class theo road type, thời tiết, ánh sáng, góc camera và mật độ.
- `bbox_distribution_by_scene.csv`: kích thước bbox gốc và sau letterbox 320×320 theo từng lát cắt.

AAU dùng toàn bộ annotation RGB của sequence đã review. UA-DETRAC dùng số ảnh/bbox toàn sequence từ XML; thống kê class và kích thước dùng reservoir sample 100.000 bbox phủ đủ 100 sequence. Mọi tỷ lệ sample đều được ghi `BBOX_ANALYSIS_SAMPLE`.

## 2. Chất lượng dữ liệu

`quality_audit_summary.csv` tổng hợp ảnh hỏng, tối/sáng, blur, annotation lỗi, exact/near duplicate và bất nhất mapping. `quality_review_queue.csv` chuyển các phát hiện thành danh sách hành động có severity.

Kết quả chính:

- Không có ảnh corrupt/unreadable trong 12.198 ảnh kiểm tra chất lượng.
- MIO có 16 ảnh nghi thiếu sáng và 29 nhóm exact duplicate trong mẫu.
- AAU có 658 ảnh nghi blur, 1.503 annotation lỗi duy nhất và temporal near-duplicate cao.
- UA-DETRAC có 313 ảnh nghi blur và 130.181 annotation lỗi duy nhất; cần lọc box lỗi trước train.
- Pipeline đã sửa mapping trong bbox sample theo `vehicle_class_mapping.yaml`, đặc biệt không còn coi pedestrian/bicycle chưa duyệt là `vehicle`.

Các ngưỡng ảnh chỉ là automatic estimate; quyết định reject vẫn cần review ảnh thật. Duplicate basename không được xem là bằng chứng nội dung trùng.

## 3. Chia tập và chống leakage

Chính sách nằm trong `configs/split_policy.yaml`. Proposal hiện có:

- MIO-TCD: train-only vì không có sequence/session tin cậy.
- AAU: 18 sequence train, 2 validation, 2 cross-test.
- UA-DETRAC: 71 sequence train, 21 validation, 8 cross-test.
- 10 sequence cross-test được cố định sau khi review metadata cảnh.
- `MAIN_K230_TEST` được giữ riêng trong `k230_holdout_plan.csv`, chưa có dữ liệu và chưa khóa.

`split_validation_summary.csv` xác nhận không có sequence hoặc source file xuất hiện ở nhiều split và không chia random frame. Coverage road type hiện `PARTIAL` vì chưa có `EMERGENCY_LANE_LIKE`; K230 holdout và test lock vẫn `PENDING`.

Tất cả split vẫn là `PROPOSAL_ONLY`: pipeline chưa copy, di chuyển hay xóa dữ liệu gốc.
