# Trạng thái phát hành model và replay

Ngày kiểm tra: 2026-08-14.

## Final training

`yolo11n_320_final.yaml` đã được kiểm tra với tham số bắt buộc: YOLO11n
pretrained, 320 px, 100 epoch, patience 20, seed 230. Final runner nhận
external path qua `--source-dataset` và `--ua-annotation-root`. Preflight đã
PASS với `D:\UMT_EVIDENCE`: dataset root, `images/`, `labels/`,
`metadata/export_summary.json` và hai thư mục UA XML đều tồn tại.

Training vẫn dừng an toàn **trước Ultralytics** vì QC của chính export ngoài
repo phát hiện hai lỗi:

```text
SUMMARY_ANNOTATION_RECONCILIATION_FAILED
IMAGE_COUNT_MISMATCH:train
```

Chi tiết đã kiểm tra: summary ghi 211.483 train image nhưng filesystem có
211.549 cặp image/label (dư 66 cặp); summary annotation có chênh lệch 3.715
box (`input_annotations - exported_boxes - rejected_annotations = -3715`).
Không có metadata giả, không tự sửa `export_summary.json` và không xóa dữ liệu
ngoài Git. Cần tái xuất dataset bằng canonical export pipeline hoặc có quyết
định được duyệt về 66 cặp train dư và số liệu annotation, rồi chạy final
training lại từ đầu.

Vì vậy hiện vẫn chưa có checkpoint fine-tune, manifest, ONNX, `.kmodel`, mAP
hay K230 metric nào được phát hành.

## UA-DETRAC stopped-vehicle replay

Logic replay đã đổi sang median ground-plane speed và hysteresis. Mọi kết quả
replay trước thay đổi này bị **INVALIDATED**. Raw UA-DETRAC video/XML đầy đủ
không có trong clone hiện tại nên full replay chưa thể chạy; không có tỷ lệ
alert hoặc mAP replay nào được dùng trong báo cáo.

Khi raw data đã được cung cấp, chạy lại `replay_ua_detrac_alerts.py` trên toàn
bộ split đã khoá, lưu version logic và ground-truth event manifest cùng run.
Chỉ kết quả mới đó mới đủ điều kiện xuất hiện trong báo cáo phát hành.

## K230

Deployment contract và release validator đã có trong repo nhưng board run chưa
có. DAY/NIGHT/RAIN/BACKLIT vẫn là `NOT_MEASURED` cho đến khi có video tự quay,
nhãn approved, test split locked và board log hash khớp với contract.
