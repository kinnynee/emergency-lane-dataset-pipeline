# 15. Tiêu chuẩn chấp nhận video

## Mục lục
- [Pháp lý](#pháp-lý)
- [Kỹ thuật](#kỹ-thuật)
- [Nội dung và góc quay](#nội-dung-và-góc-quay)
- [Từ chối](#từ-chối)

## Pháp lý
Mỗi video cần `source_id`, URL/đường dẫn gốc, nhà cung cấp, ngày tải, người thu, mục đích và license hoặc trạng thái xin phép. Rủi ro pháp lý chưa xử lý là không đạt. File gốc chỉ đọc; mọi bản công khai phải làm mờ mặt và biển số riêng.

## Kỹ thuật
Ưu tiên MP4/H.264, 1920×1080, 25/30 FPS; tối thiểu 1280×720 và 15 FPS. Phải đọc được frame đầu/giữa/cuối, đúng chiều, không đen kéo dài/hỏng/giật nặng. Không upscale hoặc tăng FPS giả để vượt kiểm tra.

## Nội dung và góc quay
Cần ít nhất một nội dung nghiệp vụ: xe, chạy xuyên, chậm/dừng/rời, nhiều xe, che khuất, gần/xa, negative hoặc điều kiện khó. Phân loại `FIXED_SURVEILLANCE`, `DASHCAM`, `K230_FIXED`, `AERIAL`, `HANDHELD`, `UNKNOWN`; ưu tiên fixed/K230, dashcam chỉ bổ sung.

## Từ chối
Từ chối nguồn/license không rõ, file hỏng/chất lượng thấp, ảnh tĩnh kéo dài, trùng hoàn toàn, góc không liên quan, dữ liệu riêng tư không thể xử lý, watermark/điều khoản cấm xử lý, nội dung nguy hiểm/không phù hợp hoặc không truy được nguồn/thời điểm.

## Checklist
- [ ] SHA-256 cho gốc và processed.
- [ ] Không thay đổi gốc.
- [ ] Nội dung “xe dừng” chỉ `NEEDS_MANUAL_REVIEW` đến khi có xác minh đáng tin.
