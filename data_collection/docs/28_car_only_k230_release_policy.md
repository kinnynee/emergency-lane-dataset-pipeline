# Chính sách phát hành model K230 — car-only

## Phạm vi nhãn

- Detector một lớp (`class_id=0`, tên nội bộ `vehicle`) chỉ giữ ô tô và các
  phương tiện bốn bánh đã được mapping phê duyệt.
- `motorcycle`, `motorbike` và `bicycle` không được xuất thành nhãn dương.
  Bbox hợp lệ của chúng được ghi vào `metadata/ignored_annotations.csv` với
  `handling=IGNORE_REGION`.
- Khi export train, vùng ignore được che đen trên **bản sao xuất ra**; ảnh và
  annotation nguồn không bị sửa. Khi đánh giá, prediction có tâm trong vùng
  ignore phải bị loại trước matching. Vì thế các đối tượng này không bị học
  ngầm là background.

## Quy tắc model và ngưỡng

- Khởi tạo train từ `yolo11n.yaml`, `pretrained: false`. Không dùng checkpoint
  COCO (`yolo11n.pt`, `yolov8*.pt`) như model fallback hoặc model bàn giao.
- Chạy ban đầu dùng 500 ảnh train có phân bổ cố định 166 MIO, 167 AAU và 167
  UA-DETRAC; mở rộng dữ liệu chỉ sau khi artifact trên K230 ổn định.
- Runtime và mọi replay vận hành khoá `class_id=0`, `confidence=0.50`.
  Kết quả 0,25 không được đưa vào báo cáo vận hành.

## Cổng phát hành K230

Một `.kmodel` chỉ được phát hành khi có cùng release:

1. checkpoint do pipeline team-model tạo ra và manifest provenance;
2. ONNX/K230 compile xuất từ checkpoint đó, không phải artifact COCO cũ;
3. log load model và inference thành công trên board/camera K230;
4. các session DAY, NIGHT, RAIN, BACKLIT do K230 tự quay, gán nhãn, review và
   khoá hoàn toàn ngoài train/validation;
5. metric tách riêng AAU, UA-DETRAC và K230 theo confidence 0,50.

Thiếu bất kỳ mục nào thì trạng thái là `NOT_MEASURED` hoặc
`BLOCKED_BOARD_RUN_REQUIRED`, không được suy rộng mAP public cross-test thành
hiệu năng camera thực địa.
