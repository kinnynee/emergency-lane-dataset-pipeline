# 07. Hướng dẫn gán nhãn YOLO

## Mục lục
- [Lớp](#lớp)
- [Quy tắc box](#quy-tắc-box)
- [Review](#review)
- [Checklist](#checklist)

## Lớp
Mặc định duy nhất: `vehicle`, class ID `0`. Mỗi dòng YOLO: `0 x_center y_center width height`, tọa độ chuẩn hóa 0-1.

## Quy tắc box
Box sát phương tiện. Xe bị cắt vẫn gán nếu nhận diện được; quá nhỏ/không chắc thì đánh dấu `REVIEW`. Gán đủ mọi xe trong ảnh. Không gán biển báo, bóng cây, người hoặc vật thể nhiễu. Ảnh không xe được giữ làm negative với label rỗng khi công cụ yêu cầu.

## Review
Kiểm tra chéo ngẫu nhiên có phân tầng 10-20%. Kiểm tra thiếu/thừa box, class, vượt biên và box quá nhỏ. Mọi thay đổi class phải được thống nhất trước khi gán hàng loạt.

## Checklist
- [ ] Ảnh và label cùng stem.
- [ ] Tất cả tọa độ trong [0,1].
- [ ] Batch có annotator, reviewer và trạng thái.

