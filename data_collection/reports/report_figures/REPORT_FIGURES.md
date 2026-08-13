# Hình báo cáo — trạng thái sau khi khoá car-only

Các biểu đồ hiệu năng cũ không còn được dùng để kết luận về model của nhóm
hoặc K230 vì chúng sử dụng model COCO hoặc confidence 0,25.

- Không gộp train/validation/cross-test, không gộp AAU với UA-DETRAC.
- Pipeline mới tạo hai biểu đồ: `A5_recall_by_dataset_size_confidence_000.png`
  và `A5_recall_by_dataset_size_confidence_050.png`. Mỗi biểu đồ tách riêng
  AAU RainSnow và UA-DETRAC Original.
- MIO-TCD chỉ là dữ liệu train; không được đưa vào chỉ số cross-test.
- AAU hiện có rất ít bbox trong ROI proxy. Khi rerun, báo cáo phải ghi rõ
  cỡ mẫu và không suy rộng kết quả từ UA-DETRAC sang AAU/K230.
- DAY/NIGHT/RAIN/BACKLIT của K230 vẫn là `NOT_MEASURED` cho đến khi có bộ
  session quay thực tế, nhãn đã duyệt và test split đã khoá.

Chạy `data_collection/scripts/generate_report_figures.py` sau một final run
đã được xác thực để sinh lại toàn bộ hình. Chính sách lớp và cổng phát hành
nằm ở `data_collection/docs/28_car_only_k230_release_policy.md`.
