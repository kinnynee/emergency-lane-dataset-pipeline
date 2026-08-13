# Hình báo cáo — trạng thái sau khóa car-only

Các biểu đồ hiệu năng cũ đã bị rút khỏi báo cáo phát hành vì chúng dùng model
COCO hoặc confidence 0,25. Không được dùng chúng để kết luận về model nhóm hay
K230.

- A4 đã loại bỏ hoàn toàn: không gộp train/val/cross-test và không báo cáo mAP
  gộp AAU + UA-DETRAC.
- A2 là hình tổng quan duy nhất; heatmap overview trùng lặp không đưa vào báo
  cáo chính.
- A5 mới sẽ có tên `A5_recall_by_dataset_size_confidence_050.png` sau khi train
  model nhóm, export car-only và rerun cross-test ở confidence 0,50. Báo cáo
  AAU và UA-DETRAC tách riêng; phải nêu rõ AAU small-bbox recall xấp xỉ 16,7%
  nếu phép đo mới xác nhận giá trị đó.
- AAU chỉ có 2 bbox trong ROI proxy. Kết luận hợp lệ hiện tại là **chưa đủ dữ
  liệu để đo đáng tin cậy**; không suy rộng từ MIO hoặc UA-DETRAC.
- K230 DAY/NIGHT/RAIN/BACKLIT là `NOT_MEASURED` cho đến khi có session tự quay,
  nhãn approved và split test đã khoá.

Pipeline tạo hình mới nằm ở `data_collection/scripts/generate_report_figures.py`.
Chính sách lớp và cổng phát hành nằm ở
`data_collection/docs/28_car_only_k230_release_policy.md`.
