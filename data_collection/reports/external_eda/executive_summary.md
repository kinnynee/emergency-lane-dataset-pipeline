# Executive summary — External Dataset EDA

- Ngày chạy: 2026-08-01 15:21:01
- Dataset: MIO-TCD Localization, AAU RainSnow, UA-DETRAC Original.
- RADIATE: `EXCLUDED_VIEWPOINT_MISMATCH`, không chạy EDA.
- Ảnh/frame kiểm tra chất lượng thật: **12,198**.
- Annotation rows đọc: **1,638,901**.
- Bounding box kiểm tra/phân tích: **1,171,685**.
- Annotation lỗi duy nhất: **131,684**; tổng issue: **141,490**.
- Nhóm trùng/nghi gần trùng trên mẫu: **3,333**.
- Leakage mức CRITICAL: **0**.
- Road type trong cross test proposal: **HIGHWAY=2, INTERSECTION=5, URBAN_ROAD=3**; `EMERGENCY_LANE_LIKE=0` nếu không xuất hiện.
- Điểm viewpoint trung bình cao nhất: **AAU RainSnow (4.20/5, AUTOMATIC_ESTIMATE)**.

Không dataset nào có ground-truth “xe dừng trong làn khẩn cấp”. UA-DETRAC chỉ tạo `STATIONARY_CANDIDATE` từ track để con người review; không dùng làm nhãn train. Main test bắt buộc ưu tiên K230 tự quay.
