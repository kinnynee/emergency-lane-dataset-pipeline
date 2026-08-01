# 08. Chính sách chia train, validation và test

## Mục lục
- [Chỉ tiêu](#chỉ-tiêu)
- [Chống rò rỉ](#chống-rò-rỉ)
- [Checklist](#checklist)

## Chỉ tiêu
Gợi ý ban đầu: train 1.000, validation 200, test tối thiểu 300 ảnh. Test chính ưu tiên K230 và bao phủ ngày, đêm, mưa, ngược sáng, xe gần/xa và không xe.

Proposal dữ liệu ngoài dùng tỷ lệ sequence `75% EXTERNAL_TRAIN`, `15% EXTERNAL_VALIDATION`, `10% CROSS_DATASET_TEST`. Mười sequence cross-test đã review được cố định trong `configs/split_policy.yaml`, không phụ thuộc random frame.

MIO-TCD Localization không có sequence/session đáng tin cậy nên chỉ được dùng cho `EXTERNAL_TRAIN`; không dùng MIO trong validation hoặc test. Test chính của dự án là `MAIN_K230_TEST`, hiện mới là kế hoạch chờ thu dữ liệu và chưa khóa.

## Chống rò rỉ
Chia theo toàn bộ `session_id` hoặc `video_id`, tuyệt đối không chia ngẫu nhiên frame gần nhau sang nhiều split. Khóa danh sách test sau khi duyệt; không dùng test để train, chọn siêu tham số hay augmentation. Mọi thay đổi test cần version mới, lý do và duyệt.

## Checklist
- [x] Proposal không trùng sequence hoặc source file giữa split (`split_validation_summary.csv`).
- [x] Không chia random frame; đơn vị là sequence hoặc nhóm train-only không có sequence.
- [x] Cross-test có ngày, đêm, chạng vạng và ba loại đường hiện có.
- [ ] Bổ sung `EMERGENCY_LANE_LIKE` cho cross-test.
- [ ] Thu dữ liệu K230 và tạo manifest main test.
- [ ] Khóa main test chỉ đọc sau khi Data Lead duyệt.
