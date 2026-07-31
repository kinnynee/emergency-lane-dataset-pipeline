# Chính sách sử dụng dataset bên ngoài

- Ba nguồn ngoài chỉ phục vụ train, external validation và cross-dataset test.
- Main project test phải ưu tiên dữ liệu K230 tự quay, khóa theo session/video.
- Không chia frame cùng sequence/video vào nhiều split.
- Không dùng bbox ảnh tĩnh để kết luận xe dừng.
- Không dùng `STATIONARY_CANDIDATE` làm nhãn train trước khi người thật duyệt.
- Luôn giữ `original_class`; class mapping về `vehicle` đang `PENDING_DATA_LEAD_APPROVAL`.
- Xe máy, xe đạp, người đi bộ và lớp mơ hồ phải được quyết định riêng.
- Không commit archive, ảnh/video hoặc contact sheet chưa bảo đảm riêng tư.
- Không tự suy đoán license. UA-DETRAC phải tách tên dataset gốc khỏi Kaggle mirror tải xuống.
- RADIATE được giữ nguyên trên ổ đĩa nhưng trạng thái là `EXCLUDED_VIEWPOINT_MISMATCH`.
