# Trạng thái phát hành model và replay

Ngày kiểm tra: 2026-08-14.

## Final training

`yolo11n_320_final.yaml` đã được kiểm tra với tham số bắt buộc: YOLO11n
pretrained, 320 px, 100 epoch, patience 20, seed 230. Lệnh final training đã
được gọi nhưng dừng trước khi tạo run directory vì clone không có validated
export tại `data_collection/dataset_output/dataset-v1-full`:

```text
Full export failed QC: ['MISSING_EXPORT_SUMMARY']
```

Vì thế hiện không có checkpoint fine-tune, manifest, ONNX, `.kmodel`, mAP hay
K230 metric nào được phát hành. Cần đặt validated export và hai UA XML roots ở
các đường dẫn tương đối trong config (hoặc truyền override CLI), sau đó chạy
`run_yolo11n_final.py` lại từ đầu.

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
