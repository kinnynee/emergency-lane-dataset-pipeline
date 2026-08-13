# Chạy replay phát hiện xe dừng

`host_yolo_loop.py` là vòng lặp PC cho YOLO, ByteTrack, calibration và ROI.
Nó dùng tốc độ ground-plane (km/h), không dùng độ dịch pixel. CanMV/K230 không
chạy các dependency này; K230 chỉ gửi detection đã được xác thực theo contract.

## Chuẩn bị

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-replay.txt
```

Tạo calibration/ROI riêng ngoài Git từ các file template. `frame_size_px` phải
trùng video; thay đổi camera, crop hoặc resolution thì phải calibration lại.

## Chạy model nhóm

Đường dẫn dưới đây chỉ là ví dụ tương đối, không phụ thuộc `D:/UMT_EVIDENCE`:

```powershell
python host_yolo_loop.py `
  --video ".\evidence\k230\lane_a.mp4" `
  --model ".\runs\final\weights\best.pt" `
  --speed-calibration ".\evidence\k230\calibration\lane_a.json" `
  --roi-config ".\evidence\k230\calibration\lane_a_roi.json" `
  --classes 0 `
  --confidence 0.50 `
  --stop-speed-kmh 2 --stop-seconds 3 `
  --resume-speed-kmh 3 --resume-seconds 1 `
  --output ".\evidence\replays\lane_a_annotated.mp4"
```

State machine vào trạng thái dừng sau 3 giây ở `<=2 km/h`; nó chỉ đóng event
sau 1 giây liên tục ở `>=3 km/h`. Tốc độ là median trong cửa sổ 1,5 giây để
giảm ảnh hưởng jitter tracker. Replay chỉ là kiểm chứng vận hành, không thay
thế K230 test set đã khoá và có ground truth.
