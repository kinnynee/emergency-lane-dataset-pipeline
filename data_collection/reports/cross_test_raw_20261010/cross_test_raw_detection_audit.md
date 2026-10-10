# Kiểm tra dự đoán thô cross-test ngày 10/10/2026

## Nguồn và cách kiểm

Nguồn người dùng cung cấp: `PHAT_HIEN_THO_CROSS_TEST_20261010.zip` (lưu ngoài Git). ZIP có `README.md`, `detections_ba_model.pkl` và `detections.pkl`. Hai hash SHA-256 **của nội dung pickle sau giải nén** khớp README:

| File | SHA-256 |
|---|---|
| `detections_ba_model.pkl` | `0822f98bdb44a235900a4fa7602e07c23a9b86712395890207b311c3be234a6a` |
| `detections.pkl` | `04603f35b036aaa4b81f0d1ae68f059fb86ce56478645d561f486ac42221028b` |

Phân tích opcode pickle cho thấy chỉ có hai global `numpy._core.numeric._frombuffer` và `numpy.dtype`; [script kiểm tra](../../scripts/audit_cross_test_raw_detections.py) dùng unpickler giới hạn đúng hai global này. Cả hai file có 12.522 row với 12.522 `stem` duy nhất, khớp toàn bộ ID của manifest cross-test `dataset-v1-full-reexport/metadata/images.csv` (lưu ngoài Git) tách từ `<DATA_ROOT>\dataset-v1-full-reexport\metadata\images.csv`. Mỗi file có đúng 10 sequence với số ảnh như [dataset audit](../dataset_audit.md). `detections_ba_model.pkl` có H1/H3 cho `full`, `squash`, `letterbox`; `detections.pkl` có H1–H4 cho `squash`, `letterbox`.

Script tính **lại từ box và score**, không chạy ONNX: score > 0,5, lấy tối đa 20 box, quy box 320 × 320 sang khung hiển thị 640 × 480, giữ box có cả hai cạnh ≥25 px, ghép tham lam theo score với GT tại IoU ≥0,5. FP loại box chưa ghép trùng `ign`/`bo` với IoU >0,3, rồi loại box có IoA ≥0,5 với [29 vùng ignore UA gốc](ua_ignore_regions.csv). Người thực hiện thí nghiệm xác nhận phép kiểm khớp **140/140 ô GT/TP và 100/100 ô FP H1/H3 sau ignore** với đường chấm chính thức ngày 09/10. Vì vậy, **FP và precision sau ignore là số chính thức**; cột trước ignore trong CSV chỉ để truy vết phép tính cũ. [Nguồn 09/10 và JSON gốc](../references/SO_LIEU_09_10_2026.md).

## Kết quả tính lại: model đầy đủ, H3, guard 25

| Sequence | Ảnh | GT | TP | FN | FP sau ignore UA | Recall (%) |
|---|---:|---:|---:|---:|---:|---:|
| Hjorringvej-4 | 100 | 592 | 197 | 395 | 7 | 33,28 |
| Ringvej-3 | 100 | 384 | 170 | 214 | 1 | 44,27 |
| MVI_39311 | 1.505 | 22.457 | 16.406 | 6.051 | 737 | 73,06 |
| MVI_39401 | 1.385 | 12.374 | 11.940 | 434 | 1.330 | 96,49 |
| MVI_39781 | 1.861 | 6.531 | 5.868 | 663 | 231 | 89,85 |
| MVI_39851 | 1.286 | 4.957 | 4.050 | 907 | 107 | 81,70 |
| MVI_40191 | 2.495 | 34.088 | 26.451 | 7.637 | 91 | 77,60 |
| MVI_40751 | 1.145 | 5.345 | 4.588 | 757 | 31 | 85,84 |
| MVI_40851 | 1.140 | 14.292 | 12.091 | 2.201 | 35 | 84,60 |
| MVI_41063 | 1.505 | 7.478 | 6.895 | 583 | 63 | 92,20 |
| **Tổng** | **12.522** | **108.498** | **88.656** | **19.842** | **2.633** | **81,71** |

Tất cả hàng, gồm các cách H1–H4 và ba model, nằm trong [CSV tính lại](raw_detection_recall.csv); hai file pickle có hàng lặp cho squash/letterbox H1/H3, nên **chỉ lấy một file** khi cộng. Tổng H3/10 sequence có precision sau ignore **97,12%**. Trên 8 sequence UA, full/H1 có TP 89.633, FP 3.175, precision **96,58%**; full/H3 có TP 88.289, FP 2.625, precision **97,11%**. [JSON gốc](../references/SO_LIEU_09_10_2026.md) xác nhận FP H3/10 sequence ONNX **2.633** và kmodel mô phỏng **2.535**.

## Hai model 500 ảnh, H1, 8 sequence UA

| Model | GT | TP | FN | FP sau ignore UA | Recall (%) | Precision sau ignore (%) |
|---|---:|---:|---:|---:|---:|---:|
| Squash V5 | 120.432 | 79.952 | 40.480 | 3.081 | 66,39 | **96,29** |
| Letterbox | 120.432 | 57.490 | 62.942 | 2.264 | 47,74 | **96,21** |

Precision trên tính từ `TP / (TP + FP sau ignore UA)`. Hai file dự đoán cùng 12.522 ảnh test; điều đó **chưa chứng minh độc lập** hai checkpoint được train trên cùng 500 ảnh vì không có train manifest/checkpoint trong ZIP. Chênh recall H1 trên UA là **18,65 điểm phần trăm**; [McNemar theo cụm 100 khung](../references/SO_LIEU_09_10_2026.md) cho p < 0,001. H3 của hai model cũng nằm trong CSV, không trộn với model đầy đủ.

## Giới hạn còn lại

- Đây là **suy luận ONNX offline đã lưu**, không có kết quả kmodel hoặc số đo board/FPS. README chỉ ghi hash ONNX rút gọn; thiếu artifact và train/run manifest để kiểm chứng nguồn gốc model.
- mAP@50 không thể tái tạo đầy đủ từ phép tính ngưỡng vận hành này; cần script evaluator/định nghĩa AP gốc. Nhận định 43/48 FP là xe thật cũng cần 48 ảnh và biên bản review.
- [JSON McNemar gốc](../references/SO_LIEU_09_10_2026.md) có các cặp bất đồng từng nhãn và p theo cụm 100 khung; báo cáo này đọc số từ JSON, chưa chạy lại phép kiểm định. Với full/H3/12.522 ảnh, chênh kmodel − ONNX +0,11 điểm, p = 0,0568: không phát hiện giảm recall trong phép đo này, cũng chưa chứng minh tương đương.
- ZIP không chứa 300 ảnh V5 cho phép tự chạy lại phép so lượng tử hoá, 81 video replay hoặc danh sách 10 ảnh test cũ. Claim V5 −1,84 điểm, p < 0,001 đã có JSON gốc riêng; các claim khác vẫn cần bằng chứng riêng.
