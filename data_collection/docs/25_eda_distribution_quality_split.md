# Hoàn thiện EDA: phân bố, chất lượng và chia tập

Cập nhật ngày 02/08/2026 theo góp ý giảng viên. Phạm vi gồm MIO-TCD Localization, AAU RainSnow và UA-DETRAC Original; RADIATE bị loại do sai góc camera.

## 1. Phân bố dữ liệu

Các báo cáo chính gồm `dataset_inventory.csv`, `class_distribution.csv`, `bbox_statistics.csv`, `cross_test_sequence_statistics.csv`, `class_distribution_by_scene.csv` và `bbox_distribution_by_scene.csv`.

Class detection được chốt thành một lớp `vehicle`: giữ mọi xe cơ giới kể cả xe máy; bỏ người đi bộ và xe đạp. `preserve_original_class=true` để audit và có thể tính mAP riêng theo lớp gốc. `UA-DETRAC:others` chỉ được giữ có điều kiện. Hàng đợi 60 mẫu phân tầng từ 48 sequence nằm tại `ua_others_stratified_review_queue.csv`; pre-review phát hiện 48 xe chắc chắn, 9 khả năng là xe, 2 non-vehicle và 1 chưa xác định. Hai mẫu non-vehicle bị đánh dấu loại, mẫu chưa xác định chờ review lần hai; toàn bộ quyết định vẫn cần Data Lead ký xác nhận.

## 2. Ánh sáng AAU RainSnow

Không dùng độ sáng trung bình để gán `DAY/NIGHT`. Độ sáng chỉ còn là chỉ số chất lượng ảnh.

Đã xem thủ công khung RGB tại 10%, 50% và 90% của đủ 22 sequence AAU. Kết quả cấu hình tại `configs/aau_sequence_lighting_review.yaml`: `DAY=10`, `NIGHT=11`, `TWILIGHT=1`. Hai sequence cross-test giữ kết quả review: `Hjorringvej-4=NIGHT`, `Ringvej-3=TWILIGHT`.

## 3. Bounding box UA-DETRAC

Audit độc lập xác nhận 130.181 box vượt biên đều nằm ở cạnh phải và/hoặc cạnh dưới, mức vượt tối đa 1 pixel. Mẫu hình này phù hợp với giả thuyết khác quy ước tọa độ/off-by-one; chưa có đủ bằng chứng để kết luận nguyên nhân là xe đang đi vào hoặc ra khỏi khung hình. Đây là trường hợp cần chuẩn hóa tọa độ, không phải lý do để xóa annotation. Pipeline hiện:

1. Giữ tọa độ gốc để truy vết.
2. Clip box về `[0, width] × [0, height]`.
3. Giữ đối tượng và dùng box đã clip để tạo label train.
4. Chỉ loại box malformed hoặc không còn diện tích nhìn thấy sau clip.

Con số 1.301.866 phải được ghi là **tổng box trong phạm vi EDA**: MIO dùng sample 5.000 ảnh, còn AAU và UA-DETRAC dùng phạm vi annotation rộng hơn/toàn bộ. Không trình bày con số này như tổng box của toàn bộ raw dataset.

`boundary_clipped_bbox_count` được báo cáo riêng; không cộng các box này vào `invalid_annotations_unique`. Không được dùng hướng dẫn cũ “lọc box lỗi trước train”.

## 4. Chất lượng và chia tập

Ảnh mờ, tối/sáng, corrupt, duplicate và nhãn malformed vẫn đi qua quality audit. Việc clip box biên là thao tác chuẩn hóa có chủ đích, không tự làm quality gate thất bại.

Dữ liệu tiếp tục chia nguyên sequence: MIO train-only, AAU 18/2/2 và UA-DETRAC 71/21/8 theo train/validation/cross-test proposal. Tất cả vẫn là `PROPOSAL_ONLY`; pipeline không copy, di chuyển hoặc xóa dữ liệu gốc.

## 5. Slice mAP còn thiếu

Bảng bài báo cần `DAY`, `NIGHT`, `BACKLIT`, `RAIN`. `evaluation_slice_readiness.csv` ghi rõ `BACKLIT=BLOCKED_MISSING_DATA`, không điền số giả. `configs/split_policy.yaml` đã thêm `K230_BACKLIT` làm slice riêng của `MAIN_K230_TEST`.
