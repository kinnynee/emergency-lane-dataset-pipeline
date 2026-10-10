# LDKC-23 — Tóm tắt đánh giá và so sánh mô hình

## Mức chứng cứ

`<DATA_ROOT>\dataset-v1-full-reexport\metadata\images.csv`, `export_summary.json`, `qc_report.json` xác nhận số ảnh và split. ZIP dự đoán thô ngày 10/10 có box/score/GT cho 12.522 ảnh; [CSV ở ngưỡng vận hành](cross_test_raw_20261010/raw_detection_recall.csv) tính **recall, TP, FN, FP và precision sau 29 vùng ignore UA**. Người thực hiện thí nghiệm xác nhận phép kiểm khớp 140/140 ô GT/TP và 100/100 ô FP H1/H3 sau ignore. [mAP@50 theo sequence](cross_test_raw_20261010/map50_after_ua_regions.csv) cũng đã được tính lại từ box/score ở confidence ≥ 0,01 và khớp các mốc bản 09/10. [Nguồn cập nhật 09/10](references/SO_LIEU_09_10_2026.md) dẫn đến JSON gốc trên nhánh `tridung-train` cho FP kmodel và McNemar. [Báo cáo V8 ngày 07/10](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) chỉ còn là nguồn lịch sử cho số liệu mô hình chưa có artifact để kiểm riêng. Không chạy lại ONNX, McNemar hoặc đo trên board.

## Mô hình đầy đủ: kết quả theo sequence được báo cáo

YOLOv8n một lớp `vehicle`, input 320 × 320; H1 ép ảnh thành vuông, H3 cắt giữa về 4:3 rồi ép thành vuông. Bảng sau là ONNX **offline**, giá trị `H1 / H3`. Recall dùng confidence > 0,5 và guard 25 px; **mAP@50 dùng confidence ≥ 0,01, không dùng guard**, sau khi loại box chưa ghép trong vùng ignore UA. Cả hai đã được tính lại từ pickle.

| Sequence | Nguồn | Ảnh đã đối chiếu | Nhãn H1 / H3 báo cáo | Recall H1 / H3 (%) báo cáo | mAP@50 H1 / H3 (%) báo cáo |
|---|---|---:|---:|---:|---:|
| Hjorringvej-4 | AAU | 100 | 592 / 592 | 33,28 / 33,28 | 57,19 / 57,19 |
| Ringvej-3 | AAU | 100 | 384 / 384 | 44,27 / 44,27 | 85,46 / 85,46 |
| MVI_39311 | UA | 1.505 | 22.457 / 22.457 | 63,01 / 73,06 | **87,62 / 89,23** |
| MVI_39401 | UA | 1.385 | 14.097 / 12.374 | 95,53 / 96,49 | **96,80 / 97,26** |
| MVI_39781 | UA | 1.861 | 7.647 / 6.531 | 84,41 / 89,85 | **96,04 / 97,22** |
| MVI_39851 | UA | 1.286 | 5.126 / 4.957 | 73,41 / 81,70 | **87,78 / 94,58** |
| MVI_40191 | UA | 2.495 | 38.401 / 34.088 | 64,60 / 77,60 | **99,48 / 99,53** |
| MVI_40751 | UA | 1.145 | 7.386 / 5.345 | 88,19 / 85,84 | **90,80 / 88,37** |
| MVI_40851 | UA | 1.140 | 15.270 / 14.292 | 78,59 / 84,60 | **97,57 / 98,09** |
| MVI_41063 | UA | 1.505 | 10.048 / 7.478 | 84,36 / 92,20 | **97,85 / 99,10** |

Nguồn mAP@50: [CSV tính lại](cross_test_raw_20261010/map50_after_ua_regions.csv), [script](../scripts/audit_cross_test_map50.py) và [tổng hợp](cross_test_raw_20261010/map50_after_ua_regions_summary.json); nguồn recall/TP/FN: [CSV ngưỡng vận hành](cross_test_raw_20261010/raw_detection_recall.csv). `export_summary.json` ghi 121.408 box cross-test, đúng số nhãn H1 trong pickle; H3 cắt bớt nhãn UA nên không được so hai recall như cùng tập object. `cross_test_sequence_statistics.csv` thuộc EDA raw có số bbox AAU/UA khác sau lọc/export; không dùng thay mẫu số đánh giá.

| Kết quả tổng hợp được báo cáo | Giá trị | Nguồn |
|---|---:|---|
| Recall gộp UA H1 / H3, guard 25 | 74,43% / 82,11% | [V8 mục 3.1](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) |
| Recall gộp AAU H1 / H3, guard 25 | 37,60% / 37,60% | [V8 mục 3.1](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) |
| Recall H3 cả 10 sequence: ONNX / kmodel mô phỏng | 81,71% / 81,82% | [V8 mục 5](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) |
| mAP@50 TB 8 sequence UA, H1 / H3, sau ignore UA | **94,24% / 95,42%** | [CSV và tổng hợp 09/10](cross_test_raw_20261010/map50_after_ua_regions_summary.json) |
| mAP@50 TB 10 sequence, H1 / H3, sau ignore UA | **89,66% / 90,60%** | [Cùng nguồn](cross_test_raw_20261010/map50_after_ua_regions_summary.json) |
| FP H3 guard 25 sau ignore UA: ONNX / kmodel mô phỏng | **2.633 / 2.535** | [JSON gốc 09/10, `gop.g25`](references/SO_LIEU_09_10_2026.md) |
| Precision H3 guard 25 sau ignore UA: ONNX / kmodel mô phỏng | **97,12% / 97,22%** | [Cùng nguồn](references/SO_LIEU_09_10_2026.md) |
| Precision ONNX trên 8 sequence UA, H1 / H3 sau ignore UA | **96,58% / 97,11%** | [CSV kiểm lại](cross_test_raw_20261010/raw_detection_recall.csv) |
| 43/48 FP được xem là xe thật thiếu nhãn/vùng ignore | Chỉ là kết luận review được báo cáo | [V8 mục 3.3](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md); `MISSING` 48 ảnh, reviewer và quyết định từng ảnh |

FP và precision **sau ignore UA** là số chính thức cho bài báo. [Phép kiểm từ ZIP](cross_test_raw_20261010/cross_test_raw_detection_audit.md) cho full/H3 trên 10 sequence: GT 108.498, TP 88.656, FN 19.842, **FP 2.633**. Riêng 8 sequence UA: H1 TP 89.633, FP 3.175, precision **96,58%**; H3 TP 88.289, FP 2.625, precision **97,11%**. Cột FP trước ignore trong CSV/JSON được giữ để truy vết, không dùng làm kết quả công bố.

## Squash so với Letterbox trên 500 ảnh

[Báo cáo V8 mục 4](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) mô tả hai model YOLOv8n cùng 500 ảnh train. Từ pickle, H1/8 sequence UA/guard 25 tính lại recall squash **66,39%**, letterbox **47,74%** và precision **sau ignore UA 96,29% / 96,21%**; xem [TP/FP/FN](cross_test_raw_20261010/cross_test_raw_detection_audit.md). **mAP@50 TB UA H1 sau ignore UA là 88,97% / 76,55%**, lần lượt cho squash/letterbox, theo [CSV mAP tính lại](cross_test_raw_20261010/map50_after_ua_regions.csv). Chưa có hai manifest 500 ảnh, checkpoint và cấu hình train để xác minh độc lập **cùng 500 ảnh train**. Chênh recall **18,65 điểm phần trăm**, phù hợp [McNemar bắt cặp trong JSON gốc](references/SO_LIEU_09_10_2026.md); không dùng chênh +12,33 điểm từ hai quy mô train khác nhau. mAP@50 dùng dải confidence từ 0,01 và khác precision tại ngưỡng vận hành.

## Bảng 4b — McNemar theo cụm 100 khung

[JSON gốc trên `tridung-train`](references/SO_LIEU_09_10_2026.md) dùng từng nhãn xe bắt cặp, tính p theo các cụm 100 khung liên tiếp trong cùng sequence. Các p dưới đây được đọc từ JSON, chưa tính lại độc lập.

| So sánh tại guard 25 | Chênh recall | p theo cụm | Diễn giải |
|---|---:|---:|---|
| V5 kmodel so với ONNX, H3, 300 ảnh | **−1,84 điểm** | **< 0,001** | Recall giảm sau lượng tử hoá trên mẫu 300 ảnh |
| Full kmodel so với ONNX, H3, 12.522 ảnh | **+0,11 điểm** | **0,057** | Không quan sát thấy giảm; p này không chứng minh tương đương |
| Squash V5 so với letterbox, H1, 8 sequence UA | **+18,65 điểm** | **< 0,001** | Squash có recall cao hơn trên tập so sánh cùng 500 ảnh train được báo cáo |

## Số liệu SV3 và giới hạn

| Mục | Trạng thái | Bằng chứng cần/đã có |
|---|---|---|
| Kmodel int16 3.483.896 byte; 3.011.043 tham số; 2,05 GFLOPs ở 320 × 320 | **Chỉ được báo cáo**, chưa đối chiếu artifact | [V8 mục 6](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md); `MISSING` kmodel, checkpoint, ONNX, log profiler của đúng model V8 |
| Confidence 0,5; NMS IoU 0,5 | **Cấu hình dự kiến kiểm được**, chưa xác nhận firmware đã chạy | [K230 contract loader](../../k230/team_model_config.py) yêu cầu 0,50/0,50; [V8 mục 2.2](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) báo ngưỡng đánh giá |
| max_boxes 20; guard 25 | **Được báo cáo**, chưa có run config gốc | [V8 mục 2.2](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) |
| NMS mặc định firmware 0,2 khi thiếu config | `MISSING` firmware/build và runtime log | Không suy ra NMS thực chạy từ loader hoặc giá trị mặc định được nêu trong yêu cầu |

Hai kết quả V5 lượng tử hoá và full trên 12.522 ảnh đã có [JSON McNemar gốc](references/SO_LIEU_09_10_2026.md); nguồn này không phải phép đo trên board. Khẳng định “V5 đêm thấp hơn ngày 6,9 điểm sau chuẩn hóa” vẫn `MISSING` bảng theo điều kiện và phép chuẩn hóa. Không dùng 97,9% và 95,4% để kết luận chính về suy giảm. Nếu không lấy được artifact/model log trước khi chốt bản nộp, bỏ bảng kích thước/tham số/GFLOPs hoặc ghi rõ chỉ là số chưa xác minh. [Báo cáo trạng thái model trong repo](model_and_replay_release_status.md) mô tả một pipeline YOLO11n khác, không phải bằng chứng cho checkpoint YOLOv8n V8.
