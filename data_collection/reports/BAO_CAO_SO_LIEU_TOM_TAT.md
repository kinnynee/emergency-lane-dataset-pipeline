# Báo cáo tóm tắt số liệu bộ dữ liệu và mô hình phát hiện phương tiện

**Ngày lập:** 10/10/2026
**Mô hình:** YOLOv8n, một lớp `vehicle`, đầu vào 320 × 320
**Nguồn số liệu:** [LDKC-23 cập nhật 09/10](evaluation_summary.md) và [nguồn JSON trên `tridung-train`](references/SO_LIEU_09_10_2026.md). [Báo cáo V8 ngày 07/10](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) chỉ giữ để đối chiếu lịch sử.

## 1. Quy mô dữ liệu

| Tập dữ liệu | Số ảnh | Ghi chú |
|---|---:|---|
| Huấn luyện | 211.025 | UA-DETRAC: 99.227; MIO-TCD: 110.000; AAU: 1.798 |
| Validation | 26.903 | Chọn checkpoint tại epoch 36 trong 50 epoch |
| Cross-test | 12.522 | 12.322 ảnh UA-DETRAC từ 8 sequence; 200 ảnh AAU từ 2 sequence |

Tập cross-test được tách theo sequence khỏi train và validation. Tổng **237.928 ảnh** là train cộng validation, không phải số ảnh huấn luyện.

## 2. Kết quả đánh giá chính

Các số recall dưới đây là **recall gộp**: cộng true positive và số nhãn của các sequence trong nhóm rồi mới chia. Recall tại ngưỡng vận hành dùng confidence > 0,5, NMS 0,5 và guard 25 px trên khung hiển thị 640 × 480. mAP@50 được tính từ confidence ≥ 0,01 và là trung bình theo sequence.

| Nhóm cross-test | Số ảnh | Recall H1 (%) | Recall H3 (%) | mAP@50 trung bình sequence H1 / H3 (%) |
|---|---:|---:|---:|---:|
| UA-DETRAC, 8 sequence | 12.322 | 74,43 | **82,11** | **94,24 / 95,42** |
| AAU, 2 sequence | 200 | 37,60 | **37,60** | 71,32 / 71,32 |
| Cả 10 sequence | 12.522 | — | **81,71** | **89,66 / 90,60** |

**H1** ép toàn bộ ảnh về 320 × 320. **H3** cắt giữa ảnh về tỉ lệ 4:3 rồi ép về 320 × 320, tương ứng hình học đầu vào camera của board. Số 81,71% của toàn bộ 10 sequence là kết quả ONNX với H3 và guard 25 px. Trên UA-DETRAC, H3 có 107.522 nhãn được chấm, so với 120.432 ở H1; vì tập nhãn thay đổi, không nên diễn giải chênh lệch recall H1–H3 như phép so sánh trên cùng một tập xe.

Hai sequence AAU gồm cảnh đêm mưa và chạng vạng mưa. Kết quả của chúng chưa đại diện cho mọi điều kiện của AAU. Xe nhỏ dưới 25 px là nhóm khó phát hiện nhất trong các nhóm được báo cáo.

mAP@50 đã tính lại từ toàn bộ box/score confidence ≥ 0,01, **sau khi trừ vùng ignore UA**, không dùng guard 25 px. [CSV theo sequence và phép tính](cross_test_raw_20261010/map50_after_ua_regions.csv) cho MVI_39311 H1/H3 **87,62% / 89,23%**; mAP@50 trung bình 8 sequence UA của squash/letterbox H1 là **88,97% / 76,55%**. Các số mAP cũ của bảng 07/10 không dùng cho bản báo cáo hiện tại.

## 3. Đối chiếu ONNX và kmodel

Phép đối chiếu dùng H3 trên toàn bộ **12.522 ảnh cross-test**, với **108.498 nhãn**. Kmodel int16 được chạy bằng **trình mô phỏng K230**, chưa đo trên board thật.

| Guard | Recall ONNX (%) | Recall kmodel (%) | FP ONNX sau ignore UA | FP kmodel sau ignore UA | Box trùng IoU ≥ 0,5 (%) |
|---:|---:|---:|---:|---:|---:|
| 25 px | **81,71** | **81,82** | **2.633** | **2.535** | 99,00 |
| 15 px | 91,49 | 91,40 | 5.095 | 4.876 | 98,55 |

Ở guard 25 px, recall kmodel mô phỏng cao hơn ONNX **0,11 điểm phần trăm**; McNemar theo cụm 100 khung cho **p = 0,057**, nên chưa thể kết luận tương đương. Hạ guard xuống 15 px làm recall tăng, đồng thời số false positive sau ignore cũng tăng. Precision sau ignore trên cả 10 sequence là **97,12% ONNX / 97,22% kmodel**; riêng 8 sequence UA, full H1/H3 ONNX đạt **96,58% / 97,11%**. Squash/letterbox H1 trên UA đạt **96,29% / 96,21%**. [Bảng 4b McNemar](evaluation_summary.md) còn ghi V5 lượng tử hoá giảm **1,84 điểm** trên 300 ảnh (p < 0,001) và squash hơn letterbox **18,65 điểm** (p < 0,001).

## 4. Kết luận và giới hạn

- Mô hình đạt recall gộp **82,11%** trên 8 sequence UA-DETRAC khi dùng H3 và guard 25 px; trên 2 sequence AAU là **37,60%**.
- Kết quả kmodel trong trình mô phỏng K230 gần với ONNX trên cùng tập thử. Báo cáo hiện **chưa có FPS, độ trễ hoặc kết quả đo trên board K230 thật**.
- Precision và false positive chính thức trên UA-DETRAC đã trừ 29 vùng ignore gốc theo IoA ≥ 0,5; các số trước ignore trong bảng 07/10 chỉ còn dùng để truy vết.
- Các kết quả suy luận đều là **offline**. Cần kiểm tra trên board và rà soát vùng ignore trước khi kết luận về hiệu năng triển khai hoặc precision thực tế.

**Lưu ý về nguồn:** [JSON kết quả toàn bộ và McNemar gốc](references/SO_LIEU_09_10_2026.md) đã có trên nhánh `tridung-train`. Phép kiểm ZIP dự đoán thô đã đối chiếu TP/FP ONNX sau ignore; p McNemar được đọc từ JSON gốc, chưa tính lại độc lập.
