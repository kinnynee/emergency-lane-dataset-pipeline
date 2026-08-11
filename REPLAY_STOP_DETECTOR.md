# Chạy replay phát hiện xe dừng

`host_yolo_loop.py` là vòng lặp dùng chung cho YOLO + ByteTrack + ROI. Cảnh
báo dừng dựa trên **tốc độ mặt đường (km/h)**, không dùng chênh lệch pixel.
`replay_stop_detector.py` được giữ lại như lệnh tương thích và gọi đúng vòng
lặp này.

Kết quả vẫn là công cụ hỗ trợ vận hành; mọi cảnh báo phải được người xem lại.
Chỉ dùng cho camera cố định/góc cao. Dashcam hoặc video bị đổi độ phân giải
không hợp lệ cho phép đo tốc độ đã hiệu chuẩn.

## Chuẩn bị

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-replay.txt
```

## Hiệu chuẩn bắt buộc

Sao chép `speed_calibration.template.json` thành một tệp riêng ngoài Git, rồi
điền bốn điểm cùng thứ tự:

- `image_points_px`: bốn điểm trên mặt đường trong ảnh/video.
- `world_points_m`: đúng bốn điểm đó trong hệ toạ độ mét đã đo tại hiện trường.
- `frame_size_px`: độ phân giải gốc của stream được hiệu chuẩn.

Ví dụ, một hình chữ nhật làn đường rộng 3,5 m và dài 20 m có thể đặt hệ toạ
độ mặt đường là `(0,0)`, `(3.5,0)`, `(3.5,20)`, `(0,20)`. Đây chỉ là quy ước
toạ độ; các vị trí pixel phải được lấy từ camera K230 thực tế. Không dùng các
giá trị `0` trong tệp template để chạy đo tốc độ.

Vòng lặp dừng ngay khi độ phân giải video khác `frame_size_px`, để tránh dùng
nhầm hiệu chuẩn.

## Chạy

Lệnh dưới đây mở giao diện chọn ROI trên frame đầu tiên. Kéo ROI quanh làn
khẩn cấp rồi nhấn Enter. Dùng `--roi x1,y1,x2,y2` nếu muốn nhập ROI chuẩn hoá
thay cho giao diện.

```powershell
python host_yolo_loop.py `
  --video "D:\UMT_Evidence\online_data\raw\youtube_cc_verified\example.mp4" `
  --model ".\yolo11n.pt" `
  --speed-calibration "D:\UMT_Evidence\k230\calibration\lane_a.json" `
  --select-roi `
  --classes 2,3,5,7 `
  --stop-speed-kmh 2 `
  --stop-seconds 3 `
  --output "D:\UMT_Evidence\online_data\replays\a40_stop_test.mp4" `
  --display
```

Với model dự án một lớp `vehicle`, thay model bằng `best.pt` và truyền
`--classes 0`. Có thể thử nhanh bằng `--max-frames 300`; khi review cuối cùng
nên giữ `--frame-stride 1`.

## Đầu ra và giới hạn

- Video overlay hiển thị `km/h` theo hiệu chuẩn, trạng thái ROI và cảnh báo.
- File `.events.csv` lưu hiệu chuẩn, ngưỡng tốc độ, tốc độ tại lúc alert và
  thời lượng dừng để truy vết.
- `--stop-speed-kmh` là ngưỡng vận hành; nó cần được xác nhận bằng video K230
  có ground truth, không tự xem là tiêu chí đánh giá model.
- Không công bố video overlay trước khi xử lý quyền riêng tư (biển số/khuôn
  mặt) và kiểm tra giấy phép nguồn video, đặc biệt với YouTube.
