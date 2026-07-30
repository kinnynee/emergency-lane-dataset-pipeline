# 08. Chính sách chia train, validation và test

## Mục lục
- [Chỉ tiêu](#chỉ-tiêu)
- [Chống rò rỉ](#chống-rò-rỉ)
- [Checklist](#checklist)

## Chỉ tiêu
Gợi ý ban đầu: train 1.000, validation 200, test tối thiểu 300 ảnh. Test chính ưu tiên K230 và bao phủ ngày, đêm, mưa, ngược sáng, xe gần/xa và không xe.

## Chống rò rỉ
Chia theo toàn bộ `session_id` hoặc `video_id`, tuyệt đối không chia ngẫu nhiên frame gần nhau sang nhiều split. Khóa danh sách test sau khi duyệt; không dùng test để train, chọn siêu tham số hay augmentation. Mọi thay đổi test cần version mới, lý do và duyệt.

## Checklist
- [ ] Không trùng session/video giữa split.
- [ ] Test locked và chỉ đọc.
- [ ] Phân bố điều kiện được EDA xác nhận.

