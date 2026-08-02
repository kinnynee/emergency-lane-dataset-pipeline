# Chính sách sử dụng dataset bên ngoài

- Ba nguồn ngoài chỉ phục vụ train, external validation và cross-dataset test.
- Main project test phải ưu tiên dữ liệu K230 tự quay, khóa theo session/video.
- Không chia frame cùng sequence/video vào nhiều split.
- Không dùng bbox ảnh tĩnh để kết luận xe dừng.
- Không dùng `STATIONARY_CANDIDATE` làm nhãn train trước khi người thật duyệt.
- Luôn giữ `original_class`; mapping về `vehicle` đã chốt theo góp ý ngày 02/08/2026. Mọi xe cơ giới kể cả xe máy được giữ; người đi bộ và xe đạp bị loại. `UA-DETRAC:others` được giữ sau review đủ 74 track, ngoại trừ `MVI_40172 / track 79` bị loại toàn bộ 201 box vì không phải xe.
- Xe máy, xe đạp, người đi bộ và lớp mơ hồ phải được quyết định riêng.
- Không commit archive, ảnh/video hoặc contact sheet chưa bảo đảm riêng tư.
- Không tự suy đoán license. UA-DETRAC phải tách tên dataset gốc khỏi Kaggle mirror tải xuống.
- RADIATE được giữ nguyên trên ổ đĩa nhưng trạng thái là `EXCLUDED_VIEWPOINT_MISMATCH`.
