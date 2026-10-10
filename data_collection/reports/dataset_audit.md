# LDKC-12 — Kiểm kê dataset cho bản nghiên cứu 10/10/2026

## Phạm vi và nguồn

Bảng này kiểm kê **bản export `dataset-v1-full-reexport`**, không cộng số ảnh của các thử nghiệm lặp lại. Số ảnh lấy từ `<DATA_ROOT>\dataset-v1-full-reexport\metadata\images.csv`, đối chiếu với `export_summary.json` và `qc_report.json` trong cùng thư mục. Cả ba ghi 250.450 ảnh; QC báo `status=PASS`, không lệch tổng ảnh/nhãn/box. Dataset gốc được chỉ ra trong khóa `sources` của `export_summary.json`: `<DATA_ROOT>\MIO-TCD-Localization.tar`, `<DATA_ROOT>\aau-rainsnow.zip`, và cây `<DATA_ROOT>` chứa UA-DETRAC. Không sửa dữ liệu gốc.

Thông tin nguồn/tải được ghi riêng trong [MIO](mio_tcd_radiate_download_evidence.md), [AAU](aau_rainsnow_download_evidence.md) và [UA-DETRAC](ua_detrac_download_evidence.md). **Số ảnh gốc trước lọc/export: `MISSING` kiểm đếm độc lập cho từng archive**; cột tổng bên dưới là số ảnh trong bản export đã xác minh, không được gọi là tổng ảnh nguyên bản của nhà cung cấp.

| Dataset | Tổng ảnh export | Train | Val | Cross-test | Ảnh đêm theo metadata sequence | Nguồn kiểm tra |
|---|---:|---:|---:|---:|---|---|
| MIO-TCD Localization | 110.000 | 110.000 | 0 | 0 | `MISSING` nhãn đêm ở mức từng ảnh; metadata gán `DAY` cho cả nhóm | `images.csv`; `<DATA_ROOT>\dataset-v1-full\metadata\sequence_scene_metadata.csv` |
| AAU RainSnow | 2.198 | 1.798 | 200 | 200 | 1.098: train 998, val 0, test 100 | Hai file trên; [review AAU](external_eda/aau_lighting_data_lead_review.md) |
| UA-DETRAC Original | 138.252 | 99.227 | 26.703 | 12.322 | 41.298: train 21.418, val 15.588, test 4.292 | Hai file trên; nhãn UA trong `sequence_scene_metadata.csv` |
| **Tổng** | **250.450** | **211.025** | **26.903** | **12.522** | `MISSING` tổng đêm chính xác vì MIO không có nhãn từng ảnh | `export_summary.json`, `qc_report.json` |

Số đêm AAU/UA được đếm bằng cách nối `(dataset, sequence_id)` từ `images.csv` sang `sequence_scene_metadata.csv` của export trước; đây là **nhãn ở mức sequence**, không phải nhãn xác nhận từng frame. Riêng MIO, dòng `MIO_NO_SEQUENCE_TRAIN_ONLY` ghi `DAY` dựa trên đặc tính tổng thể của dataset, nên không thể suy ra 110.000 ảnh đều là ảnh ban ngày hoặc số ảnh đêm bằng 0. Không gộp `TWILIGHT` vào đêm. Cross-test còn 100 ảnh AAU `TWILIGHT` từ Ringvej-3. [Review AAU](external_eda/aau_lighting_data_lead_review.md) ghi 10 sequence DAY, 11 NIGHT, 1 TWILIGHT; nhãn UA cần xác nhận theo từng frame nếu dùng để so sánh điều kiện sáng.

## Cách chia và đối chiếu tổng

- `<DATA_ROOT>\dataset-v1-full-reexport\metadata\sequence_splits.csv` và `images.csv` ghi split thực của export. [Split proposal](external_eda/split_proposal.csv) và [policy](../configs/split_policy.yaml) là đầu vào thiết kế, không thay thế manifest của export đã tạo. MIO chỉ train vì thiếu sequence/session tin cậy; AAU và UA chia nguyên sequence.
- Khi quét toàn bộ `images.csv`, không thấy `(dataset, sequence_id)` hoặc `(dataset, source_image)` xuất hiện ở nhiều split. Đây là kiểm tra **ID/đường dẫn nguồn**, chưa phải chứng minh không có ảnh cùng nội dung hoặc frame gần giống ở các đường dẫn khác nhau. Xem [báo cáo leakage](leakage_report.md).
- 211.025 + 26.903 + 12.522 = 250.450. Con số 237.928 trong [báo cáo V8](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) là train + val, không bao gồm cross-test.
- `dataset-v1-full` là bản export khác: `<DATA_ROOT>\dataset-v1-full\metadata\export_summary.json` ghi 251.074 ảnh và 12.422 cross-test. Không cộng hoặc thay thế số của bản `reexport` bằng bản này. Các số thống kê EDA như 1.301.866 **box** trong [audit cũ](audit/FULL_EDA_AUDIT_REPORT.md) cũng không phải số ảnh export.

## Cross-test của bản reexport

| Nguồn | Sequence | Ảnh export | Nhãn sáng trong metadata |
|---|---|---:|---|
| AAU | Hjorringvej-4 | 100 | NIGHT |
| AAU | Ringvej-3 | 100 | TWILIGHT |
| UA | MVI_39311 | 1.505 | DAY |
| UA | MVI_39401 | 1.385 | DAY |
| UA | MVI_39781 | 1.861 | NIGHT |
| UA | MVI_39851 | 1.286 | NIGHT |
| UA | MVI_40191 | 2.495 | DAY |
| UA | MVI_40751 | 1.145 | NIGHT |
| UA | MVI_40851 | 1.140 | DAY |
| UA | MVI_41063 | 1.505 | DAY |

Số ảnh trong bảng được đếm từ `images.csv`; chúng khớp bảng 10 sequence trong [báo cáo V8](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md). Việc cùng số lượng không xác nhận checkpoint nào đã sử dụng đúng export này: `MISSING` hash manifest train/val/test gắn với checkpoint và lệnh chạy đánh giá.
