# Chính sách phát hành model car-only cho K230

## Nhãn và đánh giá

- Detector chỉ có một lớp: `class_id=0`, tên `vehicle`, cho ô tô và phương tiện
  bốn bánh đã được phê duyệt. Motorcycle, motorbike và bicycle không là positive.
- AAU RainSnow và UA-DETRAC Original luôn có mAP/recall riêng tại
  `confidence=0.00` và `confidence=0.50`; không công bố headline gộp hai domain.
- MIO-TCD chỉ dùng để train. Replay cũ dùng logic trước median/hysteresis không
  được dùng cho báo cáo phát hành.

## Transfer learning và bằng chứng fine-tune

Được phép khởi tạo từ COCO YOLO11n cho smoke và final training. Tuy vậy, trọng số
COCO nguyên bản không phải là model nhóm và không được biên dịch/phát hành cho K230.
`team_model_manifest.json` phải lưu `base_weights`, `checkpoint`, `finetuned_on` và
`training_run`. Cổng phát hành bắt buộc từ chối khi:

```text
SHA256(final_checkpoint) == SHA256(base_weights)
```

Mã lỗi là `FINAL_CHECKPOINT_EQUALS_BASE_WEIGHTS`.

| Cấp độ | Dữ liệu | Tham số |
| --- | --- | --- |
| Smoke pipeline | đúng 500 train image | 320 px, 25 epoch, seed 230 |
| Final release | toàn bộ validated export | 320 px, 100 epoch, patience 20, seed 230 |

Config dùng đường dẫn tương đối; khi dataset/weights ở ngoài Git, truyền đường dẫn
thật bằng CLI. Với layout external chuẩn, chạy final training bằng:

```powershell
python data_collection/scripts/run_yolo11n_final.py `
  --run-dir runs/final-320-100ep-seed230 `
  --source-dataset "D:\UMT_EVIDENCE\dataset-v1-full" `
  --ua-annotation-root "D:\UMT_EVIDENCE"
```

Preflight phải xác nhận `images/`, `labels/`, `metadata/export_summary.json`
và cả hai XML directory trước QC; thiếu bất kỳ mục nào thì dừng với danh sách
đường dẫn chính xác. Final training phải chạy lại report ở hai confidence sau
khi hoàn tất.

## Hợp đồng Host → K230

- Host dùng Ultralytics, OpenCV và ByteTrack; đó là dependency phía PC, không phải
  dependency CanMV.
- K230 chỉ được load `.kmodel` được đóng gói bằng deployment contract có hash của
  checkpoint-manifest và `.kmodel`. Sai/thiếu contract phải fail-closed với
  `MODEL_LOAD_REJECTED`; không có COCO fallback.
- Lớp, input và ngưỡng phải cố định: `vehicle` / class `0`, RGB NCHW 320×320
  uint8, confidence `0.50`, NMS IoU `0.50`.
- Host là nơi chạy state machine cảnh báo: median ground-plane speed, vào dừng ở
  `<=2.0 km/h` trong 3 giây, chỉ đóng event sau `>=3.0 km/h` liên tục 1 giây.
  K230 gửi detection/telemetry theo contract, không chạy một logic cảnh báo khác.
- Board log phải có `MODEL_LOAD_OK`, `INFERENCE_OK`, hash manifest và hash kmodel
  đúng với contract. Thiếu log/session K230 đã khoá thì trạng thái là
  `NOT_MEASURED` / `BLOCKED_BOARD_RUN_REQUIRED`.

Chi tiết file contract và đoạn tích hợp CanMV ở [`k230/README.md`](../../k230/README.md).
