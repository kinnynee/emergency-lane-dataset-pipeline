# LDKC-23 — Tóm tắt đánh giá và so sánh mô hình

## Mức chứng cứ

`<DATA_ROOT>\dataset-v1-full-reexport\metadata\images.csv`, `export_summary.json`, `qc_report.json` xác nhận số ảnh và split. ZIP dự đoán thô do người dùng cung cấp ngày 10/10 có box/score/GT cho 12.522 ảnh; [báo cáo kiểm tra độc lập](cross_test_raw_20261010/cross_test_raw_detection_audit.md) và [CSV theo sequence](cross_test_raw_20261010/raw_detection_recall.csv) tính lại **recall, TP, FN, FP và precision ở ngưỡng vận hành**, khớp các số V8 được đối chiếu. [Báo cáo V8 ngày 07/10/2026](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) vẫn là nguồn của mAP@50, kmodel mô phỏng và độ phức tạp; JSON/script đánh giá gốc chưa có. Không chạy lại ONNX hoặc đo real-time trên board.

## Mô hình đầy đủ: kết quả theo sequence được báo cáo

YOLOv8n một lớp `vehicle`, input 320 × 320; H1 ép ảnh thành vuông, H3 cắt giữa về 4:3 rồi ép thành vuông. Bảng sau là ONNX **offline**, recall với confidence > 0,5, NMS 0,5, guard 25 px; giá trị `H1 / H3`. Số ảnh và recall đã được tính lại từ pickle; mAP@50 vẫn chỉ là số báo cáo.

| Sequence | Nguồn | Ảnh đã đối chiếu | Nhãn H1 / H3 báo cáo | Recall H1 / H3 (%) báo cáo | mAP@50 H1 / H3 (%) báo cáo |
|---|---|---:|---:|---:|---:|
| Hjorringvej-4 | AAU | 100 | 592 / 592 | 33,28 / 33,28 | 57,19 / 57,19 |
| Ringvej-3 | AAU | 100 | 384 / 384 | 44,27 / 44,27 | 85,46 / 85,46 |
| MVI_39311 | UA | 1.505 | 22.457 / 22.457 | 63,01 / 73,06 | 63,23 / 72,80 |
| MVI_39401 | UA | 1.385 | 14.097 / 12.374 | 95,53 / 96,49 | 90,43 / 92,07 |
| MVI_39781 | UA | 1.861 | 7.647 / 6.531 | 84,41 / 89,85 | 93,17 / 95,53 |
| MVI_39851 | UA | 1.286 | 5.126 / 4.957 | 73,41 / 81,70 | 85,23 / 91,38 |
| MVI_40191 | UA | 2.495 | 38.401 / 34.088 | 64,60 / 77,60 | 97,07 / 96,18 |
| MVI_40751 | UA | 1.145 | 7.386 / 5.345 | 88,19 / 85,84 | 89,70 / 88,37 |
| MVI_40851 | UA | 1.140 | 15.270 / 14.292 | 78,59 / 84,60 | 88,82 / 89,37 |
| MVI_41063 | UA | 1.505 | 10.048 / 7.478 | 84,36 / 92,20 | 94,98 / 99,05 |

Nguồn mAP@50: [mục 3.1 báo cáo V8](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md); nguồn recall/TP/FN tính lại: [CSV](cross_test_raw_20261010/raw_detection_recall.csv). `export_summary.json` ghi 121.408 box cross-test, đúng số nhãn H1 trong pickle; H3 cắt bớt nhãn UA nên không được so hai recall như cùng tập object. `cross_test_sequence_statistics.csv` thuộc EDA raw có số bbox AAU/UA khác sau lọc/export; không dùng thay mẫu số đánh giá.

| Kết quả tổng hợp được báo cáo | Giá trị | Nguồn |
|---|---:|---|
| Recall gộp UA H1 / H3, guard 25 | 74,43% / 82,11% | [V8 mục 3.1](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) |
| Recall gộp AAU H1 / H3, guard 25 | 37,60% / 37,60% | [V8 mục 3.1](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) |
| Recall H3 cả 10 sequence: ONNX / kmodel mô phỏng | 81,71% / 81,82% | [V8 mục 5](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) |
| FP H3 guard 25: ONNX / kmodel mô phỏng | 19.908 / 19.782 | [V8 mục 5](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md); chưa có danh sách FP |
| 43/48 FP được xem là xe thật thiếu nhãn/vùng ignore | Chỉ là kết luận review được báo cáo | [V8 mục 3.3](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md); `MISSING` 48 ảnh, reviewer và quyết định từng ảnh |

Precision UA trước khi áp dụng 29 vùng ignore gốc có thể thấp giả: [phép tính lại](cross_test_raw_20261010/cross_test_raw_detection_audit.md) cho full/H3 **TP 88.656, FN 19.842, FP 19.908**, khớp FP báo cáo. CSV có TP/FN/FP theo sequence; kết quả sau áp dụng 29 vùng UA chỉ là thăm dò, chưa dùng làm precision chính thức. McNemar có thể xét từ kết quả ghép cặp từng object trong pickle, nhưng cần chốt đơn vị độc lập hoặc cách xử lý frame cùng sequence; chưa tính p-value.

## Squash so với Letterbox trên 500 ảnh

[V8 mục 4](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) nói hai model YOLOv8n cùng 500 ảnh train. Từ pickle, H1/8 sequence UA/guard 25 tính lại recall squash **66,39%**, letterbox **47,74%** và precision tương ứng **86,23% / 90,68%** khi chưa áp dụng 29 vùng ignore UA gốc; xem [TP/FP/FN](cross_test_raw_20261010/cross_test_raw_detection_audit.md). Chưa có hai manifest 500 ảnh, checkpoint và cấu hình train để xác minh **cùng 500 ảnh train**. Chênh recall đã tính lại là **18,65 điểm phần trăm**; không dùng chênh +12,33 điểm từ hai quy mô train khác nhau. V8 còn báo cáo mAP@50 trung bình UA H1 squash 83,99% và letterbox 70,95%; mAP chưa tính lại và khác precision.

## Số liệu SV3 và giới hạn

| Mục | Trạng thái | Bằng chứng cần/đã có |
|---|---|---|
| Kmodel int16 3.483.896 byte; 3.011.043 tham số; 2,05 GFLOPs ở 320 × 320 | **Chỉ được báo cáo**, chưa đối chiếu artifact | [V8 mục 6](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md); `MISSING` kmodel, checkpoint, ONNX, log profiler của đúng model V8 |
| Confidence 0,5; NMS IoU 0,5 | **Cấu hình dự kiến kiểm được**, chưa xác nhận firmware đã chạy | [K230 contract loader](../../k230/team_model_config.py) yêu cầu 0,50/0,50; [V8 mục 2.2](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) báo ngưỡng đánh giá |
| max_boxes 20; guard 25 | **Được báo cáo**, chưa có run config gốc | [V8 mục 2.2](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) |
| NMS mặc định firmware 0,2 khi thiếu config | `MISSING` firmware/build và runtime log | Không suy ra NMS thực chạy từ loader hoặc giá trị mặc định được nêu trong yêu cầu |

Các khẳng định “V5 giảm 1,84 điểm trên 300 ảnh mới”, “model đầy đủ không giảm trên 12.522 ảnh”, “V5 đêm thấp hơn ngày 6,9 điểm sau chuẩn hóa” hiện `MISSING` cặp run và phép tính gốc; không thay bằng số của V8. Không dùng 97,9% và 95,4% để kết luận chính về suy giảm. Nếu không lấy được artifact/model log trước khi chốt bản nộp, bỏ bảng kích thước/tham số/GFLOPs hoặc ghi rõ chỉ là số chưa xác minh. [Báo cáo trạng thái model trong repo](model_and_replay_release_status.md) mô tả một pipeline YOLO11n khác, không phải bằng chứng cho checkpoint YOLOv8n V8.
