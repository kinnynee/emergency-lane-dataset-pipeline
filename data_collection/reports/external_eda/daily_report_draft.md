[SV1 – 02/08/2026]

1. Hôm nay làm:
Thực hiện EDA cho MIO-TCD Localization, AAU RainSnow và UA-DETRAC nhằm đánh giá mức độ phù hợp với hệ thống camera K230 đặt cố định trên cao.

2. Kết quả/bằng chứng:
- Dataset đã tìm thấy: MIO-TCD Localization, AAU RainSnow, UA-DETRAC Original
- Dataset chưa tìm thấy: KHÔNG CÓ
- Số ảnh đã kiểm tra: 12,198
- Số video/sequence đã kiểm tra: 166
- Số annotation đã đọc: 1,638,901
- Số bounding box đã phân tích: 1,301,866
- Số annotation lỗi duy nhất: 1,503
- Tổng số issue annotation: 1,503
- Số nhóm ảnh nghi ngờ trùng: 3,333
- Tỷ lệ box dưới 8 px sau resize 320×320: {'MIO-TCD Localization': 0.13741649, 'AAU RainSnow': 0.37476683, 'UA-DETRAC Original': 0.06013477}
- Dataset phù hợp nhất về góc camera: AAU RainSnow (4.31/5, cần review)
- Road type trong cross test proposal: HIGHWAY=2, INTERSECTION=5, URBAN_ROAD=3; chưa có `EMERGENCY_LANE_LIKE`.
- Điều kiện cross test đã review: camera_view: ELEVATED_OBLIQUE=10; lighting: DAY=5, NIGHT=2, TWILIGHT=3; traffic_density: HIGH=4, LOW=2, MEDIUM=4; weather: CLEAR=3, CLOUDY=2, RAIN_OR_WET_ROAD=2, UNKNOWN=3.
- Quality gate: REVIEW_REQUIRED=3; class mapping corrections trong bbox sample: 0.
- Split validation: PARTIAL=1, PASS=8, PENDING=2; MIO train-only, K230 main test đang chờ thu.
- Điều kiện dữ liệu được bổ sung: mưa/tuyết, camera cố định, tracking sequence.
- Link báo cáo: reports/external_eda/research_findings.md
- Link biểu đồ: reports/external_eda/figures/
- Link commit/PR: commit hiện tại f5c943a517139f43dbd3db2aa6f6db8eef9ab8d2

3. Vướng mắc/cần hỗ trợ:
- Class mapping đã chốt: CÓ; chỉ `UA-DETRAC:others` tiếp tục sample review và luôn giữ original class.
- Dữ liệu chưa có nhãn detection: KHÔNG; AAU có COCO instance annotation, nhưng điều kiện theo sequence cần review.
- Dataset quá lớn: CÓ; image quality chạy theo sample/streaming.
- Thiếu dung lượng: KHÔNG XÁC NHẬN LÀ VƯỚNG MẮC.
- Thiếu dữ liệu K230 thực tế: CÓ.
- Cần giảng viên xác nhận: class xe máy/xe đạp, subset, vị trí K230 và protocol main test.

4. Ngày mai:
- Review các ảnh lỗi.
- Tiếp tục sample review class `UA-DETRAC:others`.
- Chốt subset cân bằng.
- Chuẩn bị dữ liệu gán nhãn còn thiếu.
- Tiếp tục khảo sát dữ liệu K230 thực tế.
