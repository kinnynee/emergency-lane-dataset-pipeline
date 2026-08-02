# 08. Chính sách chia train, validation và test

## Chỉ tiêu

Proposal dữ liệu ngoài dùng tỷ lệ sequence `75% EXTERNAL_TRAIN`, `15% EXTERNAL_VALIDATION`, `10% CROSS_DATASET_TEST`. Không chia random frame.

- MIO-TCD không có sequence/session đáng tin cậy nên chỉ dùng cho `EXTERNAL_TRAIN`.
- AAU RainSnow và UA-DETRAC được chia nguyên sequence.
- Test chính là `MAIN_K230_TEST`, tách khỏi dữ liệu ngoài và khóa sau khi Data Lead duyệt.

Bảng kết quả bài báo phải báo cáo mAP riêng cho bốn slice `DAY`, `NIGHT`, `BACKLIT` và `RAIN`. Vì chưa có dữ liệu ngược sáng, K230 phải quay một sequence `K230_BACKLIT` riêng; không được suy diễn cột ngược sáng từ ảnh ban ngày có độ sáng cao.

## Chống rò rỉ

Toàn bộ `session_id`, `video_id` hoặc `sequence_id` chỉ thuộc một split. Không dùng test để train, chọn siêu tham số hoặc augmentation. Mỗi thay đổi test phải tạo version, ghi lý do và được duyệt.

## Checklist

- [x] Proposal không trùng sequence hoặc source file giữa split.
- [x] Không chia random frame.
- [x] Cross-test hiện có ngày, đêm, chạng vạng và ba loại đường.
- [ ] Bổ sung `EMERGENCY_LANE_LIKE`.
- [ ] Quay riêng K230 `BACKLIT` và ba slice `DAY`, `NIGHT`, `RAIN`.
- [ ] Tạo manifest rồi khóa `MAIN_K230_TEST`.
