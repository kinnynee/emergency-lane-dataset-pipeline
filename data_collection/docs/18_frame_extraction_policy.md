# 18. Chính sách tách frame

## Mục lục
- [Tần suất](#tần-suất)
- [Truy vết và split](#truy-vết-và-split)
- [Checklist](#checklist)

## Tần suất
Mặc định 1 frame/giây; cảnh chậm có thể 0,5 FPS, thay đổi nhiều có thể 2 FPS. Không lấy toàn bộ frame. Tên: `<processed_video_name>_f000001_t000001000.jpg`, timestamp tính millisecond.

## Truy vết và split
Chỉ tách video có `final_status=APPROVED`. Mỗi ảnh có `image_id`, `video_id`, `source_id`, timestamp, condition, source/camera type, chất lượng, duplicate status và storage path. Chia theo video/source/session; không trộn frame cùng video hoặc video gần trùng vào các split khác nhau. Test ưu tiên K230 và khóa cứng.

## Checklist
- [ ] Không ghi đè frame tồn tại.
- [ ] Near-duplicate chỉ đánh dấu, không xóa tự động.
- [ ] Nhãn là `UNASSIGNED`/cần review cho đến khi người thật duyệt.
