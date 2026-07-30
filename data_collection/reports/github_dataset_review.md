# Rà soát dataset trên GitHub cho cảnh báo xe dừng trong làn khẩn cấp

Ngày rà soát: 2026-07-29  
Người rà soát: Codex

## Kết luận

Không tìm thấy repository GitHub công khai nào cung cấp sẵn nhãn trực tiếp `xe dừng trong làn khẩn cấp`. Các nguồn gần nhất chỉ đáp ứng một phần:

- Nhận diện phương tiện từ camera giao thông cố định.
- Cảnh cao tốc, ban đêm, mưa, đường ướt hoặc xe ở xa.
- Video bất thường giao thông để tìm thủ công đoạn xe dừng hoặc tai nạn.

Dataset Internet chỉ nên dùng để tăng độ đa dạng cho lớp `vehicle`. Trạng thái **dừng trong ROI** vẫn phải được xác định bằng chuỗi thời gian/tracking và dữ liệu K230 tự quay; không nên biến `stopped_vehicle` thành class YOLO mới khi guideline hiện tại chỉ có class `vehicle`.

## Xếp hạng mức phù hợp

| Hạng | Nguồn | Phần phù hợp | Khoảng trống | Trạng thái |
|---:|---|---|---|---|
| 1 | UA-DETRAC | Camera cố định, 100 chuỗi, xe có box/track, có đêm và mưa | Không có nhãn làn khẩn cấp hoặc trạng thái dừng; license/kênh tải chính thức chưa xác minh | `LICENSE_UNVERIFIED` |
| 2 | AAU RainSnow | Camera cố định, 22 video tại 7 giao lộ, ngày–chạng vạng–đêm, mưa/tuyết, RGB và thermal | Không phải cao tốc; archive có hai cây dữ liệu lặp cần lọc trước khi xử lý | `DOWNLOADED` |
| 3 | Vehicle-Rear | Hơn 3 giờ video camera cố định, gần 3.000 xe, nhiều loại xe và ánh sáng khó | Thiên về re-identification, chứa biển số; license repo chưa chứng minh áp dụng cho media | `LICENSE_UNVERIFIED` |
| 4 | RADIATE | Motorway, đêm, mưa, sương, tuyết, xe xa và nhiều loại xe | Camera ego 672×376/15 FPS; nhãn chính trên radar; toàn bộ dataset phải đăng ký | Sample `DOWNLOADED`; full `NEEDS_PERMISSION` |
| 5 | AI City Challenge 2020 Track 4 | Camera giao lộ/cao tốc và bài toán bất thường giao thông | Repo kết quả không chứa dataset; dữ liệu phải xin ban tổ chức | `LICENSE_UNVERIFIED` (đã có `SRC_ONL003`) |
| 6 | RWVC-BDD100K | 13.3K ảnh có nhãn đường, thời tiết, tầm nhìn; hữu ích lọc mưa/ướt/đêm | Chỉ là annotation mở rộng; ảnh gốc và quyền dùng thuộc BDD100K | `NEEDS_PERMISSION` |
| 7 | L-RadSet | 280 cảnh, 11.2K keyframe, cao tốc, xe tới 220 m, đêm/mưa/sương | Camera ego đa cảm biến; phải ký agreement và nhận link qua email | `NEEDS_PERMISSION` |
| 8 | DoTA | Video anomaly và mốc thời gian, có thể chứa xe dừng/tai nạn | Dashcam; không có nhãn lane/ROI; điều khoản media chưa rõ; không được chạy luồng YouTube/cookie | `NEEDS_PERMISSION` |

## Kiểm tra pháp lý quan trọng

- License hiển thị ở đầu repository thường chỉ chắc chắn áp dụng cho **mã nguồn**. Không suy diễn rằng Apache-2.0/MIT/GPL của repo cũng cấp quyền với ảnh và video được liên kết.
- AAU RainSnow có license dữ liệu `CC BY 4.0` được xác nhận qua metadata API và Kaggle CLI chính thức. Kaggle CLI 2.2.4 cho phép tải dataset công khai ở chế độ ẩn danh; archive đã được tải và kiểm tra checksum/CRC.
- RADIATE công bố `CC BY-NC-SA 4.0` trên website chính thức. Vì tải qua đăng ký và chỉ cho mục đích phi thương mại, nguồn vẫn cần review/đăng ký trước khi tải.
- DoTA có tuyên bố tác giả đã liên hệ chủ video để được phép chia sẻ, nhưng repo không trình bày đủ điều khoản cấp phép media cho mục đích hiện tại. Luồng tải YouTube bằng cookie không được dùng trong dự án này.
- L-RadSet yêu cầu ký agreement và gửi email; không tự động hóa bước xin quyền.

## Nguồn bị loại khỏi vai trò dataset

- `gustavovelascoh/traffic-surveillance-dataset`: chỉ là danh mục liên kết, không chứa media.
- `achen353/Taiwanese-Traffic-Object-Detection`: README nói rõ dữ liệu cuộc thi được giữ bí mật.
- `hustvl/YOLOP`: repo mã mô hình, dữ liệu thực là BDD100K đã được quản lý bằng nguồn riêng.
- Các repo “emergency vehicle detection” về xe cứu thương/cứu hỏa không liên quan đến xe dừng trong làn khẩn cấp.
- Dataset tổng hợp/simulator không được dùng vì kế hoạch yêu cầu dữ liệu thật và không tạo dữ liệu giả.

## Hướng đề xuất

1. Ưu tiên xin/kiểm tra UA-DETRAC hoặc dùng AAU RainSnow để bổ sung camera cố định, mưa và đêm.
2. Dùng RADIATE hoặc BDD100K chỉ để bổ sung cảnh cao tốc/điều kiện khó, không dùng thay dữ liệu K230.
3. Gắn nhãn YOLO duy nhất `vehicle`; lưu `in_roi`, `stopped`, `moving`, `entered_roi`, `left_roi` ở metadata sự kiện hoặc kết quả tracker.
4. Muốn có đúng cảnh làn khẩn cấp, cần quay K230 tại khu vực an toàn/được phép hoặc xin dữ liệu từ đơn vị vận hành cao tốc; GitHub hiện không cung cấp bộ khớp hoàn toàn.

## Bằng chứng kỹ thuật đã kiểm tra

- GitHub API được dùng để đối chiếu license của repository và xác định repo có thực sự chứa data/release hay chỉ chứa code.
- README chính thức được kiểm tra để lấy quy mô, góc camera, thời tiết, loại annotation và cách cấp quyền.
- AAU RainSnow đã được tải bằng Kaggle CLI chính thức vào `storage_placeholders/online_data/raw/aau_rainsnow/aau-rainsnow.zip`.
- Hai archive MIO-TCD Classification/Localization và sample RADIATE `tiny_foggy.zip` cũng đã được tải từ trang chính thức; xem báo cáo bằng chứng tải riêng.
