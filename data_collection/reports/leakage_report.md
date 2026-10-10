# LDKC-12 — Rà soát data leakage và giới hạn đánh giá

## Kết quả kiểm tra có thể tái đối chiếu

| Kiểm tra | Kết quả | Bằng chứng và giới hạn |
|---|---|---|
| Trùng sequence giữa train/val/cross-test trong bản reexport | 0 nhóm trùng ID | Quét `<DATA_ROOT>\dataset-v1-full-reexport\metadata\images.csv` theo `(dataset, sequence_id)`; [sequence manifest](references/BAO_CAO_DANH_GIA_MO_HINH_V8.md) chỉ là nguồn tường thuật cho thí nghiệm V8. |
| Trùng đường dẫn `source_image` giữa split trong bản reexport | 0 đường dẫn trùng | Quét `images.csv` theo `(dataset, source_image)`. Không phát hiện file khác đường dẫn nhưng cùng nội dung. |
| Audit nội dung exact AAU/UA của proposal trước | PASS trong phạm vi audit cũ | [sequence_leakage_audit.csv](audit/sequence_leakage_audit.csv), [duplicate_cross_split_audit.csv](audit/duplicate_cross_split_audit.csv). Audit này không tự động chứng minh toàn bộ pixel export reexport hoặc model V8 sạch leakage. |
| Near duplicate giữa split | `MISSING` | [audit](audit/FULL_EDA_AUDIT_REPORT.md) nêu pHash hiện chỉ quét mẫu và so sánh các ảnh kề nhau trong cùng sequence; chưa có quét đầy đủ giữa split. |
| Video/session/camera độc lập | `MISSING` xác nhận toàn bộ | MIO không có sequence tin cậy; AAU Hjorringvej-4 cùng họ camera/góc với Hjorringvej-1/2/3 trong train theo mô tả nghiên cứu. UA chưa có manifest camera/địa điểm đầy đủ gắn với mọi sequence. |

`<DATA_ROOT>\dataset-v1-full-reexport\metadata\sequence_splits.csv` xác nhận Hjorringvej-1/2/3 ở train và Hjorringvej-4 ở cross-test. Vì vậy, “khác sequence” không đồng nghĩa “unseen camera”. Không dùng nhãn **unseen camera** cho AAU Hjorringvej-4 hoặc UA nếu thiếu camera ID và địa điểm kiểm được. Kết quả cross-test chỉ hỗ trợ đánh giá trên sequence tách khỏi train/val trong manifest hiện có.

## Hai cảnh báo từ yêu cầu nghiên cứu cần đối chiếu danh sách gốc

- **10 ảnh test cũ nằm trong train:** hiện là thông tin do người dùng cung cấp, `MISSING` danh sách 10 đường dẫn/hash và phiên bản split. Không đưa 10 ảnh này vào tập test độc lập hoặc tính metric công bố cho đến khi đối chiếu từng ảnh.
- **58 trong 81 video replay vòng 7 thuộc train của model đầy đủ:** hiện là thông tin do người dùng cung cấp, `MISSING` manifest 81 video, 58 cặp trùng, checkpoint và phiên bản train. Nếu xác nhận, replay này là kiểm tra nội bộ trên dữ liệu đã thấy, không phải external/held-out evaluation. [Trạng thái replay của repo](model_and_replay_release_status.md) còn ghi các kết quả replay trước thay đổi logic đã bị vô hiệu hóa.

Không có dữ liệu đủ để liệt kê tên từng mẫu trùng của hai cảnh báo trên. Không tự coi các số do người dùng đưa là kết quả quét mới.

## Ảnh hưởng và phương án

1. Khóa **phiên bản dataset, manifest ảnh/sequence/camera/video và checkpoint** cho từng kết quả, lưu SHA-256 của file đầu vào. Bỏ mọi ảnh/video có quan hệ train khỏi phép đánh giá độc lập; giữ báo cáo lịch sử với nhãn `contaminated` nếu cần truy vết.
2. Quét hash exact theo file/pixel của cả ba split; lập danh sách ứng viên near duplicate bằng pHash/embedding trên các split, xác nhận thủ công và lưu cặp đường dẫn, hash, độ tương đồng. Kiểm tra frame từ cùng video/session kể cả tên file khác.
3. Lập `camera_id`, `location_id`, `video_id`, `sequence_id` cho UA/AAU/K230. Nếu mục tiêu là camera chưa từng thấy, chia theo **camera hoặc địa điểm**, rồi khóa test trước khi chọn mô hình. Với AAU Hjorringvej-4 cần loại cả nhóm Hjorringvej khỏi train cho phép thử đó.
4. Nếu chia lại làm thay đổi train hoặc test, tạo **phiên bản thí nghiệm mới** và chạy lại đánh giá; không sửa metric đã công bố thành metric của split mới. Chưa chia lại dataset hoặc train lại model trong đợt rà soát này.

Mức ảnh hưởng: hai cảnh báo 10 ảnh và 58/81 video có nguy cơ **cao** nếu xác nhận; cùng camera làm yếu kết luận về khả năng tổng quát hóa sang camera mới; near duplicate còn **chưa xác định**. Kết luận này phù hợp với giới hạn của [audit EDA](audit/FULL_EDA_AUDIT_REPORT.md) và không mở rộng thành khẳng định “không leakage”.
