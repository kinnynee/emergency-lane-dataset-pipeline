[SV1 – 01/08/2026]

1. Hôm nay làm:
Thực hiện EDA cho MIO-TCD Localization, AAU RainSnow và UA-DETRAC nhằm đánh giá mức độ phù hợp với hệ thống camera K230 đặt cố định trên cao.

2. Kết quả/bằng chứng:
- Dataset đã tìm thấy: MIO-TCD Localization, AAU RainSnow, UA-DETRAC Original
- Dataset chưa tìm thấy: KHÔNG CÓ
- Số ảnh đã kiểm tra: 12,198
- Số video/sequence đã kiểm tra: 166
- Số annotation đã đọc: 1,638,901
- Số bounding box đã phân tích: 1,171,685
- Số annotation lỗi duy nhất: 131,684
- Tổng số issue annotation: 141,490
- Số nhóm ảnh nghi ngờ trùng: 3,333
- Tỷ lệ box dưới 8 px sau resize 320×320: {'MIO-TCD Localization': 0.13741649, 'AAU RainSnow': 0.37476683, 'UA-DETRAC Original': 0.06631237}
- Dataset phù hợp nhất về góc camera: AAU RainSnow (4.20/5, cần review)
- Road type trong cross test proposal: HIGHWAY=2, INTERSECTION=5, URBAN_ROAD=3; chưa có `EMERGENCY_LANE_LIKE`.
- Điều kiện cross test đã review: camera_view: ELEVATED_OBLIQUE=10; lighting: DAY=5, NIGHT=2, TWILIGHT=3; traffic_density: HIGH=4, LOW=2, MEDIUM=4; weather: CLEAR=3, CLOUDY=2, RAIN_OR_WET_ROAD=2, UNKNOWN=3.
- Điều kiện dữ liệu được bổ sung: mưa/tuyết, camera cố định, tracking sequence.
- Link báo cáo: reports/external_eda/research_findings.md
- Link biểu đồ: reports/external_eda/figures/
- Link commit/PR: commit hiện tại 976e0f182744ccededb87910ff6cbee5e06cecff

3. Vướng mắc/cần hỗ trợ:
- Class mapping chưa được xác nhận: CÓ, trạng thái PENDING_DATA_LEAD_APPROVAL.
- Dữ liệu chưa có nhãn detection: KHÔNG; AAU có COCO instance annotation, nhưng điều kiện theo sequence cần review.
- Dataset quá lớn: CÓ; image quality chạy theo sample/streaming.
- Thiếu dung lượng: KHÔNG XÁC NHẬN LÀ VƯỚNG MẮC.
- Thiếu dữ liệu K230 thực tế: CÓ.
- Cần giảng viên xác nhận: class xe máy/xe đạp, subset, vị trí K230 và protocol main test.

4. Ngày mai:
- Review các ảnh lỗi.
- Chốt class mapping.
- Chốt subset cân bằng.
- Chuẩn bị dữ liệu gán nhãn còn thiếu.
- Tiếp tục khảo sát dữ liệu K230 thực tế.
