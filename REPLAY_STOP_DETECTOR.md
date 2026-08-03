# Chạy replay phát hiện xe dừng

`replay_stop_detector.py` đọc video gốc, YOLO phát hiện xe, ByteTrack gán `track_id`, và chỉ cảnh báo khi xe đứng yên trong ROI đủ thời gian. Video overlay và log CSV là **ứng viên cần duyệt thủ công**, không phải ground truth.

## Chuẩn bị một lần

Máy cần Python 3.11 hoặc 3.12. Từ thư mục này, tạo môi trường riêng và cài dependency:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-replay.txt
```

Để thử với YOLO COCO (chỉ để test), lấy một model local:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

Khi dùng model một lớp `vehicle` của dự án, thay đường dẫn model bằng `best.pt` và truyền `--classes 0`.

## Test khuyến nghị

Dùng A40 vì camera cố định, dễ đánh giá dừng trong ROI. Chạy lệnh sau, rồi kéo ROI quanh làn khẩn cấp/đoạn đường cần giám sát và nhấn Enter:

```powershell
python replay_stop_detector.py `
  --video "D:\UMT_Evidence\online_data\raw\youtube_cc_verified\YTSave_YouTube_Motorway-A40-From-Bridge_Media_A7HvgY2s2f8_001_720p.mp4" `
  --model ".\yolo11n.pt" `
  --select-roi `
  --classes 2,3,5,7 `
  --stop-seconds 3 `
  --output "D:\UMT_Evidence\online_data\replays\a40_stop_test.mp4" `
  --display
```

`2,3,5,7` là các lớp car, motorcycle, bus, truck của YOLO COCO. Khi test ngắn, thêm `--max-frames 300`; với video 30 FPS đó là khoảng 10 giây.

## Kết quả và hiệu chỉnh

- Video có khung ROI màu vàng; chuyển đỏ khi có `STOPPED`.
- File `.events.csv` cùng tên video liệt kê track ID, thời điểm alert và lý do kết thúc. Mọi dòng có `NEEDS_MANUAL_REVIEW`.
- Chỉ dùng camera cố định/elevated cho cảnh báo dừng. Dashcam di chuyển cùng xe nên chỉ phù hợp demo phát hiện xe, không phải xác minh dừng.
- Nếu cảnh báo nhạy quá, giảm `--motion-threshold-px` xuống `8` hoặc tăng `--stop-seconds` lên `4`.
- Chưa thấy cảnh báo không có nghĩa là video không có xe dừng; video hiện tại chưa có positive event đã xác nhận.

Không công bố video overlay khi chưa xử lý quyền riêng tư (biển số/khuôn mặt) và kiểm tra license.
