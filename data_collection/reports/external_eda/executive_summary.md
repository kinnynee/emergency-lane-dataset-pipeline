# Executive summary — External Dataset EDA

- Ngày chạy: 2026-08-01 15:43:42
- Dataset: MIO-TCD Localization, AAU RainSnow, UA-DETRAC Original.
- RADIATE: `EXCLUDED_VIEWPOINT_MISMATCH`, không chạy EDA.
- Ảnh/frame kiểm tra chất lượng thật: **12,198**.
- Annotation rows đọc: **1,638,901**.
- Bounding box kiểm tra/phân tích: **1,171,685**.
- Annotation lỗi duy nhất: **131,684**; tổng issue: **141,490**.
- Nhóm trùng/nghi gần trùng trên mẫu: **3,333**.
- Leakage mức CRITICAL: **0**.
- Road type trong cross test proposal: **HIGHWAY=2, INTERSECTION=5, URBAN_ROAD=3**; `EMERGENCY_LANE_LIKE=0` nếu không xuất hiện.
- Điều kiện cross test đã review: **camera_view: ELEVATED_OBLIQUE=10; lighting: DAY=5, NIGHT=2, TWILIGHT=3; traffic_density: HIGH=4, LOW=2, MEDIUM=4; weather: CLEAR=3, CLOUDY=2, RAIN_OR_WET_ROAD=2, UNKNOWN=3**.
- Quality gate: **REVIEW_REQUIRED=3**; bbox sample được sửa theo class mapping: **1,313**.
- Kiểm tra split: **PARTIAL=1, PASS=8, PENDING=2**; MIO không có sequence được giữ train-only.
- Điểm viewpoint trung bình cao nhất: **AAU RainSnow (4.20/5, AUTOMATIC_ESTIMATE)**.

Không dataset nào có ground-truth “xe dừng trong làn khẩn cấp”. UA-DETRAC chỉ tạo `STATIONARY_CANDIDATE` từ track để con người review; không dùng làm nhãn train. Main test bắt buộc ưu tiên K230 tự quay.
