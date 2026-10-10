# Kiểm tra dự đoán thô cross-test ngày 10/10/2026

## Nguồn và cách kiểm

Nguồn người dùng cung cấp: `PHAT_HIEN_THO_CROSS_TEST_20261010.zip (ngu?n ng??i d?ng cung c?p, l?u ngo?i Git)`. ZIP có `README.md`, `detections_ba_model.pkl` và `detections.pkl`. Hai hash SHA-256 **của nội dung pickle sau giải nén** khớp README:

| File | SHA-256 |
|---|---|
| `detections_ba_model.pkl` | `0822f98bdb44a235900a4fa7602e07c23a9b86712395890207b311c3be234a6a` |
| `detections.pkl` | `04603f35b036aaa4b81f0d1ae68f059fb86ce56478645d561f486ac42221028b` |

Phân tích opcode pickle cho thấy chỉ có hai global `numpy._core.numeric._frombuffer` và `numpy.dtype`; [script kiểm tra](../../scripts/audit_cross_test_raw_detections.py) dùng unpickler giới hạn đúng hai global này. Cả hai file có 12.522 row với 12.522 `stem` duy nhất, khớp toàn bộ ID của manifest cross-test `dataset-v1-full-reexport/metadata/images.csv` (l?u ngo?i Git) tách từ `<DATA_ROOT>\dataset-v1-full-reexport\metadata\images.csv`. Mỗi file có đúng 10 sequence với số ảnh như [dataset audit](../dataset_audit.md). `detections_ba_model.pkl` có H1/H3 cho `full`, `squash`, `letterbox`; `detections.pkl` có H1–H4 cho `squash`, `letterbox`.

Script tính **lại từ box và score**, không chạy ONNX: score > 0,5, lấy tối đa 20 box, quy box 320 × 320 sang khung hiển thị 640 × 480, giữ box có cả hai cạnh ≥25 px, ghép tham lam theo score với GT tại IoU ≥0,5. Recall độc lập khớp số trong [V8](../references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) đến hai chữ số thập phân. FP loại box chưa ghép trùng `ign`/`bo` với IoU >0,3. Cột **không áp dụng 29 vùng UA gốc** tái tạo đúng FP/precision được báo cáo; phép tính sau khi áp dụng [29 vùng UA](ua_ignore_regions.csv) theo IoA ≥0,5 được giữ riêng và chỉ dùng thăm dò vì script chấm gốc không có trong ZIP.

## Kết quả tính lại: model đầy đủ, H3, guard 25

| Sequence | Ảnh | GT | TP | FN | FP chưa áp dụng vùng UA gốc | Recall (%) |
|---|---:|---:|---:|---:|---:|---:|
| Hjorringvej-4 | 100 | 592 | 197 | 395 | 7 | 33,28 |
| Ringvej-3 | 100 | 384 | 170 | 214 | 1 | 44,27 |
| MVI_39311 | 1.505 | 22.457 | 16.406 | 6.051 | 7.084 | 73,06 |
| MVI_39401 | 1.385 | 12.374 | 11.940 | 434 | 8.252 | 96,49 |
| MVI_39781 | 1.861 | 6.531 | 5.868 | 663 | 1.818 | 89,85 |
| MVI_39851 | 1.286 | 4.957 | 4.050 | 907 | 308 | 81,70 |
| MVI_40191 | 2.495 | 34.088 | 26.451 | 7.637 | 229 | 77,60 |
| MVI_40751 | 1.145 | 5.345 | 4.588 | 757 | 31 | 85,84 |
| MVI_40851 | 1.140 | 14.292 | 12.091 | 2.201 | 2.058 | 84,60 |
| MVI_41063 | 1.505 | 7.478 | 6.895 | 583 | 120 | 92,20 |
| **Tổng** | **12.522** | **108.498** | **88.656** | **19.842** | **19.908** | **81,71** |

Tất cả hàng, gồm các cách H1–H4 và ba model, nằm trong [CSV tính lại](raw_detection_recall.csv). `FN` và `FP` ở đây là kết quả agent tính lại từ dữ liệu thô với luật trên. FP 19.908 khớp [V8 mục 5](../references/BAO_CAO_DANH_GIA_MO_HINH_V8.md). Áp dụng 29 vùng UA gốc làm FP H3 của `full` trên UA từ 19.900 xuống 2.625 (cộng AAU 8 thành 2.633); đây **chưa phải precision chính thức** vì cần script chấm gốc, review vùng ignore và xác minh từng ảnh FP.

## Hai model 500 ảnh, H1, 8 sequence UA

| Model | GT | TP | FN | FP chưa áp dụng vùng UA gốc | Recall (%) | Precision (%) |
|---|---:|---:|---:|---:|---:|---:|
| Squash V5 | 120.432 | 79.952 | 40.480 | 12.765 | 66,39 | 86,23 |
| Letterbox | 120.432 | 57.490 | 62.942 | 5.911 | 47,74 | 90,68 |

Các số precision trên **đã tính lại** từ pickle và khớp số tham chiếu trong yêu cầu. Hai file dự đoán cùng 12.522 ảnh test; điều đó **chưa chứng minh** hai checkpoint được train trên cùng 500 ảnh vì không có train manifest/checkpoint trong ZIP. Chênh recall H1 trên UA là 18,65 điểm phần trăm. H3 của hai model cũng nằm trong CSV, không trộn với model đầy đủ.

## Giới hạn còn lại

- Đây là **suy luận ONNX offline đã lưu**, không có kết quả kmodel hoặc số đo board/FPS. README chỉ ghi hash ONNX rút gọn; thiếu artifact và train/run manifest để kiểm chứng nguồn gốc model.
- mAP@50 không thể tái tạo đầy đủ từ phép tính ngưỡng vận hành này; cần script evaluator/định nghĩa AP gốc. Nhận định 43/48 FP là xe thật cũng cần 48 ảnh và biên bản review.
- Các box/GT có thể ghép theo từng ảnh và từng object để kiểm tra bất đồng squash–letterbox; tuy nhiên frame cùng sequence tương quan mạnh. Chưa tính kiểm định McNemar hay p-value từ tỷ lệ tổng hợp; cần chốt đơn vị phân tích và xử lý phụ thuộc theo video/sequence.
- ZIP không chứa 300 ảnh “chưa thấy”, log V5 so với model đầy đủ, 81 video replay, hoặc danh sách 10 ảnh test cũ. Các claim đó vẫn `MISSING` bằng chứng riêng.
