# BÁO CÁO ĐÁNH GIÁ MÔ HÌNH PHÁT HIỆN PHƯƠNG TIỆN YOLOv8n

> **Lưu trữ bản 07/10/2026.** mAP@50, FP và precision chưa trừ 29 vùng ignore UA trong tài liệu này đã được thay thế bởi [số chính thức ngày 09/10](SO_LIEU_09_10_2026.md). Khi viết bài báo, dùng [LDKC-23 đã cập nhật](../evaluation_summary.md) và các bảng kiểm lại được dẫn tại đó.

**Cơ sở lập báo cáo:** tệp *BANG_SO_LIEU_BAI_BAO.md* do người dùng cung cấp. Các kết quả dưới đây được tổng hợp từ bảng số liệu đó; báo cáo này không chạy lại mã đánh giá hoặc kiểm chứng các tệp JSON gốc.
**Phạm vi:** đánh giá mô hình đầy đủ trên tập cross_test, phép thử hình học với hai mô hình 500 ảnh, và đối chiếu ONNX với kmodel bằng trình mô phỏng K230.
**Ngày lập:** 07/10/2026.

## 1. Tóm tắt kết quả

Mô hình chính là YOLOv8n một lớp *vehicle*, đầu vào 320 × 320. Mô hình được huấn luyện trên 211.025 ảnh trong 50 epoch và chọn checkpoint ở epoch 36. Tập đánh giá cross_test gồm 12.522 ảnh thuộc 10 sequence, tách theo sequence khỏi tập train và validation.

Trên 8 sequence UA-DETRAC, recall gộp tại ngưỡng vận hành guard 25 px đạt **74,43% với H1** và **82,11% với H3**. Trên 2 sequence AAU, recall gộp là **37,60%** ở cả hai cách xử lý ảnh. H3 cắt giữa khung hình về tỉ lệ 4:3 nên có ít nhãn UA hơn H1; hai tỉ lệ recall này không được hiểu là phép so sánh trên cùng một tập xe.

Với đầu vào H3 và guard 25 px, ONNX đạt recall **81,71%** trên toàn bộ 10 sequence; kmodel int16 chạy trong **trình mô phỏng K230** đạt **81,82%**. Đây là kết quả offline, chưa phải phép đo trên board thật. Chưa có số FPS trên board.

## 2. Mô hình, dữ liệu và quy trình đánh giá

### 2.1. Mô hình và tập dữ liệu

| Thành phần | Mô tả |
|---|---|
| Kiến trúc | YOLOv8n, một lớp *vehicle*, đầu vào 1 × 3 × 320 × 320 |
| Huấn luyện | 211.025 ảnh: UA-DETRAC 99.227, MIO-TCD 110.000, AAU 1.798 |
| Validation | 26.903 ảnh; checkpoint chọn tại epoch 36 trong 50 epoch |
| Cross_test | 12.522 ảnh: UA-DETRAC 12.322 ảnh từ 8 sequence; AAU 200 ảnh từ 2 sequence |
| Cách chia | Theo sequence; không có sequence cross_test trùng với train/validation |

Tổng 237.928 là **train cộng validation** sau khi đưa ảnh về 320 × 320, không phải số ảnh huấn luyện.

### 2.2. Hình học đầu vào và luật chấm

- **H1:** ép thẳng toàn bộ ảnh về 320 × 320, giống cách xử lý khi huấn luyện mô hình đầy đủ.
- **H3:** cắt giữa ảnh về tỉ lệ 4:3 rồi ép về 320 × 320, tương ứng hình học đầu vào camera của board.
- Điểm vận hành: confidence > 0,5; NMS 0,5; tối đa 20 box; guard 25 px hoặc 15 px trên khung hiển thị 640 × 480.
- Ghép nhãn tại IoU ≥ 0,5; mAP được tính từ confidence ≥ 0,01.
- Detection trùng vùng ignore với IoU > 0,3 không được tính là false positive (FP). Với H3, xe bị cắt còn dưới 50% diện tích cũng được đưa vào vùng ignore.

Hai sequence AAU vốn có tỉ lệ 4:3 nên H1 và H3 cho cùng số liệu. Với UA, H3 loại bớt phần hai bên ảnh 16:9. Vì vậy, khi diễn giải chênh lệch H1–H3 cần xét cả thay đổi về số nhãn được chấm.

## 3. Kết quả của mô hình đầy đủ

### 3.1. Theo từng sequence

Kết quả ONNX trên cross_test. mAP@50 và recall được ghi theo thứ tự **H1 / H3**; recall dùng guard 25 px.

| Sequence | Nguồn | Ánh sáng | Ảnh | Nhãn H1 / H3 | mAP@50 H1 / H3 (%) | Recall H1 / H3 (%) |
|---|---|---|---:|---:|---:|---:|
| Hjorringvej-4 | AAU | Đêm | 100 | 592 / 592 | 57,19 / 57,19 | 33,28 / 33,28 |
| Ringvej-3 | AAU | Chạng vạng | 100 | 384 / 384 | 85,46 / 85,46 | 44,27 / 44,27 |
| MVI_39311 | UA | Ngày | 1.505 | 22.457 / 22.457 | 63,23 / 72,80 | 63,01 / 73,06 |
| MVI_39401 | UA | Ngày | 1.385 | 14.097 / 12.374 | 90,43 / 92,07 | 95,53 / 96,49 |
| MVI_39781 | UA | Đêm | 1.861 | 7.647 / 6.531 | 93,17 / 95,53 | 84,41 / 89,85 |
| MVI_39851 | UA | Đêm | 1.286 | 5.126 / 4.957 | 85,23 / 91,38 | 73,41 / 81,70 |
| MVI_40191 | UA | Ngày | 2.495 | 38.401 / 34.088 | 97,07 / 96,18 | 64,60 / 77,60 |
| MVI_40751 | UA | Đêm | 1.145 | 7.386 / 5.345 | 89,70 / 88,37 | 88,19 / 85,84 |
| MVI_40851 | UA | Ngày | 1.140 | 15.270 / 14.292 | 88,82 / 89,37 | 78,59 / 84,60 |
| MVI_41063 | UA | Ngày | 1.505 | 10.048 / 7.478 | 94,98 / 99,05 | 84,36 / 92,20 |

| Cách tổng hợp | H1 (%) | H3 (%) |
|---|---:|---:|
| Recall gộp 8 sequence UA, guard 25 | 74,43 | 82,11 |
| Recall gộp 2 sequence AAU, guard 25 | 37,60 | 37,60 |
| Recall gộp 8 sequence UA, guard 15 | 89,64 | 91,96 |
| Recall gộp 2 sequence AAU, guard 15 | 40,78 | 40,78 |
| mAP@50 trung bình 8 sequence UA | 87,83 | 90,59 |
| mAP@50 trung bình 2 sequence AAU | 71,32 | 71,32 |
| mAP@50 trung bình 10 sequence | 84,53 | 86,74 |

**Gộp** nghĩa là cộng số true positive và số nhãn của các sequence rồi mới chia; **trung bình sequence** cho mỗi sequence cùng trọng số. Đây là hai cách tổng hợp khác nhau, không nên dùng thay thế cho nhau.

Trên UA, H3 có 107.522 nhãn so với 120.432 nhãn ở H1, giảm 12.910 nhãn do cắt khung. Recall của phần lớn sequence UA cao hơn ở H3, nhưng MVI_40751 giảm từ 88,19% xuống 85,84%. Kết quả AAU thấp, đặc biệt cảnh đêm Hjorringvej-4 đạt 33,28% recall.

### 3.2. Theo nguồn và điều kiện ánh sáng

Các số sau dùng ONNX, H1 và guard 25 px. Nhãn ánh sáng được gán theo sequence, không theo từng ảnh.

| Nhóm | Sequence | Ảnh | Nhãn | mAP@50 (%) | Recall (%) | Recall xe < 25 px (%) | Recall xe 25–49 px (%) | Recall xe ≥ 50 px (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UA ngày | 5 | 8.030 | 100.273 | 87,71 | 72,70 | 11,8 | 86,3 | 97,6 |
| UA đêm | 3 | 4.292 | 20.159 | 89,17 | 83,00 | 18,6 | 84,1 | 97,3 |
| AAU đêm | 1 | 100 | 592 | 57,19 | 33,28 | 8,4 | 60,6 | 84,1 |
| AAU chạng vạng | 1 | 100 | 384 | 85,46 | 44,27 | 15,6 | 54,8 | 100,0 |

Recall thô của UA đêm cao hơn UA ngày, nhưng cảnh đêm có ít xe và xe lớn hơn. Khi chuẩn hóa theo phân bố kích thước xe của nhóm UA ngày, recall UA đêm là **72,93%**, gần với **72,70%** của UA ngày (chênh khoảng 0,2 điểm phần trăm). Vì thế, số thô 83,00% không đủ để kết luận mô hình hoạt động tốt hơn vào ban đêm. Xe dưới 25 px là nhóm khó nhất ở cả bốn nhóm dữ liệu.

Hai sequence AAU là cảnh **đêm mưa** và **chạng vạng mưa**. Không thể xem chúng là đại diện cho toàn bộ AAU hoặc mọi điều kiện ánh sáng.

### 3.3. Giới hạn của precision

Precision trên UA có xu hướng bị tính thấp: bản export validation/cross_test thiếu tọa độ vùng ignore gốc của UA-DETRAC, gồm xe đỗ và xe ở xa không được gán nhãn. Xe thật được phát hiện trong những vùng này có thể bị tính là FP. Theo tệp nguồn, trong 48 FP được kiểm tra ngẫu nhiên có 43 trường hợp là xe thật. Lỗi vùng ignore này không ảnh hưởng đến recall. Vì vậy, báo cáo lấy **recall** làm chỉ số vận hành chính và chỉ trình bày precision kèm cảnh báo khi cần.

## 4. Phép thử hình học với hai mô hình 500 ảnh

Đây là **hai mô hình thử nghiệm riêng**, cùng kiến trúc YOLOv8n và cùng 500 ảnh train, chỉ khác cách xử lý ảnh lúc huấn luyện: ép thẳng (*squash*) hoặc giữ tỉ lệ bằng đệm (*letterbox*). Chúng không phải mô hình đầy đủ ở mục 3. H2 là đệm về 4:3 rồi ép 320 × 320; H4 là letterbox giữ tỉ lệ ảnh gốc.

| Hình học test | Cách train | Recall gộp UA (%) | mAP@50 TB UA (%) | Recall gộp AAU (%) | mAP@50 TB AAU (%) |
|---|---|---:|---:|---:|---:|
| H1 | Letterbox | 47,74 | 70,95 | 10,86 | 24,64 |
| H1 | Squash | 66,39 | 83,99 | 15,68 | 33,31 |
| H2 | Letterbox | 48,35 | 78,96 | 10,86 | 24,64 |
| H2 | Squash | 56,14 | 82,56 | 15,68 | 33,31 |
| H3 | Letterbox | 63,56 | 82,50 | 10,86 | 24,64 |
| H3 | Squash | 74,63 | 88,42 | 15,68 | 33,31 |
| H4 | Letterbox | 38,82 | 78,76 | 11,99 | 30,38 |
| H4 | Squash | 41,93 | 76,20 | 14,86 | 30,23 |

Ở hình học H3 của board, mô hình train squash cao hơn mô hình train letterbox **11,07 điểm phần trăm recall trên UA** và **4,82 điểm trên AAU**. Kết quả này hỗ trợ lựa chọn squash cho mô hình đầy đủ, nhưng chỉ là bằng chứng từ thí nghiệm 500 ảnh; không được gán trực tiếp các mức chênh đó cho mô hình huấn luyện trên 211.025 ảnh.

## 5. Đối chiếu ONNX và kmodel

Kmodel int16 được chạy qua **trình mô phỏng K230** bằng nncase 2.9.0, so với ONNX trên PC. Phép đối chiếu dùng H3, toàn bộ 12.522 ảnh cross_test và 108.498 nhãn.

| Guard (px) | Recall ONNX (%) | Recall kmodel (%) | FP ONNX | FP kmodel | Box trùng IoU ≥ 0,5 (%) | IoU trung bình |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 81,71 | 81,82 | 19.908 | 19.782 | 99,00 | 0,9857 |
| 15 | 91,49 | 91,40 | 45.861 | 45.289 | 98,55 | 0,9830 |

Ở guard 25, chênh recall kmodel so với ONNX là **+0,11 điểm phần trăm**; ở guard 15 là **−0,09 điểm**. Theo tệp nguồn, sai khác recall từng sequence ở guard 25 nằm từ −0,76 đến +0,43 điểm. Các số này cho thấy kết quả mô phỏng của kmodel gần ONNX trong tập thử đã dùng.

Giảm guard từ 25 xuống 15 px tăng recall ONNX từ 81,71% lên 91,49%, đồng thời số FP ONNX tăng từ 19.908 lên 45.861. Việc chọn guard cần dựa trên mục tiêu vận hành và kiểm tra thực tế. Phép mô phỏng chưa xác nhận tốc độ, độ trễ hay độ ổn định trên board K230.

## 6. Độ phức tạp và chuyển đổi mô hình

| Đại lượng | Giá trị |
|---|---:|
| Số tham số | 3.011.043; sau gộp Conv + BN: 3.005.843 |
| GFLOPs ở 320 × 320 | 2,05; sau gộp Conv + BN: 2,02 |
| Kích thước kmodel int16 | 3.483.896 byte |
| Kích thước ONNX FP32 | 12.139.054 byte |
| Kích thước checkpoint PyTorch FP16 | 6.212.906 byte |
| Đầu ra mô hình | 1 × 5 × 2100; 2100 = 40² + 20² + 10² |

Chuỗi chuyển đổi được tệp nguồn ghi là PyTorch → ONNX opset 13 (simplify, batch 1, không NMS) → nncase 2.9.0 và nncase-kpu 2.9.0, target K230. Đầu vào uint8 NCHW được chia 255 trong bước tiền xử lý; PTQ dùng 100 ảnh calibration ép 320 × 320, NoClip và lượng tử hóa int16. Con số **8,1 GFLOPs** do Ultralytics in ra ứng với ảnh **640 × 640**; ở kích thước 320 × 320 của mô hình này phải ghi **2,05 GFLOPs**.

## 7. Giới hạn và kết luận

Các phát hiện chính là: mô hình đầy đủ đạt recall gộp 82,11% trên UA với H3 và guard 25; kết quả AAU thấp hơn nhiều trên hai cảnh mưa được thử; nhóm xe nhỏ dưới 25 px còn khó phát hiện; kmodel mô phỏng cho kết quả gần với ONNX trên cùng tập ảnh. Phép thử 500 ảnh ủng hộ cách train squash tại hình học H3.

Khi sử dụng các số này trong bài báo cần ghi rõ: **(1)** tất cả kết quả suy luận là offline; **(2)** Bảng đối chiếu kmodel dùng trình mô phỏng, chưa đo trên board; **(3)** precision UA bị ảnh hưởng bởi vùng ignore thiếu trong bản export; **(4)** H1 và H3 có tập nhãn khác nhau trên UA; **(5)** thí nghiệm 500 ảnh không phải kết quả của mô hình đầy đủ; **(6)** chưa có số FPS trên board. Phần đánh giá tiếp theo nên đo trực tiếp trên K230 và rà soát vùng ignore gốc trước khi đưa ra kết luận về precision thực tế.

---

**Nguồn duy nhất:** *BANG_SO_LIEU_BAI_BAO.md* (tệp do người dùng cung cấp). Tệp nguồn cho biết bảng được tạo từ các JSON đã lưu bằng script *scripts/bao_cao/bang_bai_bao_v8.py*; các JSON và script đó không có trong workspace hiện tại để đối chiếu độc lập.
