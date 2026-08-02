# 07. Hướng dẫn gán nhãn YOLO

## Lớp mục tiêu

Mô hình detection dùng duy nhất lớp `vehicle`, class ID `0`. Bao gồm toàn bộ phương tiện cơ giới như ô tô, xe tải, xe buýt, xe van và xe máy. Không gán người đi bộ hoặc xe đạp.

Luôn giữ `original_class` trong metadata theo `configs/vehicle_class_mapping.yaml` để có thể audit nguồn và tính lại mAP theo lớp gốc, đặc biệt cho xe máy. Class `others` của UA-DETRAC vẫn được ánh xạ sang `vehicle`, nhưng phải tiếp tục lấy mẫu kiểm tra hình ảnh.

## Quy tắc bounding box

Mỗi dòng YOLO có dạng `0 x_center y_center width height`, tọa độ chuẩn hóa trong `[0,1]`.

- Gán box sát phần phương tiện nhìn thấy.
- Phương tiện đang đi vào hoặc ra khỏi ảnh vẫn là mẫu hợp lệ.
- Nếu box nguồn vượt biên, clip tọa độ về biên ảnh rồi giữ đối tượng.
- Chỉ loại box malformed, không hữu hạn, hoặc không còn diện tích nhìn thấy sau khi clip.
- Không gán biển báo, bóng cây, người đi bộ, xe đạp hoặc vật thể gây nhiễu.
- Ảnh không có phương tiện được giữ làm negative với label rỗng.

## Review

Kiểm tra chéo có phân tầng 10–20%, gồm thiếu/thừa box, box sau clip, đối tượng ở mép ảnh, class gốc và các mẫu `UA-DETRAC:others`. Không xóa hàng loạt box vượt biên.

## Checklist

- [ ] Ảnh và label có cùng stem.
- [ ] Tọa độ xuất train nằm trong `[0,1]`.
- [ ] Box ở biên đã được clip, không bị loại chỉ vì vượt biên nguồn.
- [ ] `mapped_class=vehicle` và `original_class` vẫn được lưu.
- [ ] Batch có annotator, reviewer và trạng thái.
