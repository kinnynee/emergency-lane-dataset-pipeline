# Bằng chứng tải MIO-TCD và RADIATE sample

Ngày tải/kiểm tra: `2026-07-29`

## MIO-TCD

- Trang chính thức: `https://tcd.miovision.com/challenge/dataset.html`
- Provider: Université de Sherbrooke và Miovision
- License công bố trên trang tải: `CC BY-NC-SA 4.0`
- Camera: hàng nghìn camera giao thông tại Canada và Hoa Kỳ

### Classification

- URL: `https://tcd.miovision.com/static/dataset/MIO-TCD-Classification.tar`
- File: `storage_placeholders/online_data/raw/mio_tcd/MIO-TCD-Classification.tar`
- Kích thước: `3,117,240,320` byte (`2,972.83 MiB`)
- SHA-256: `0CF70C660F399AEF05D069D9C993E2E3F89E887C823C781D3064FEE3B2E6979C`
- Số member TAR: `648,977`
- JPG: `648,959`
- Đường dẫn tuyệt đối hoặc chứa `..`: `0`
- Đọc cấu trúc TAR đến EOF: `PASS`

### Localization

- URL: `https://tcd.miovision.com/static/dataset/MIO-TCD-Localization.tar`
- File: `storage_placeholders/online_data/raw/mio_tcd/MIO-TCD-Localization.tar`
- Kích thước: `3,739,303,424` byte (`3,566.08 MiB`)
- SHA-256: `0801D0EE34E7FB211945CD5EF836551EFADAC3A1EC876B126608EC5FBE7FAB64`
- Số member TAR: `137,750`
- JPG: `137,743`
- Có `3` CSV và `1` README
- Đường dẫn tuyệt đối hoặc chứa `..`: `0`
- Đọc cấu trúc TAR đến EOF: `PASS`

Hai TAR chưa được giải nén. Chỉ Localization có bounding box phù hợp trực tiếp hơn với bài toán YOLO; Classification chủ yếu là crop/nhãn lớp và background.

## RADIATE sample

- Trang chính thức: `https://pro.hw.ac.uk/radiate/downloads/`
- License công bố: `CC BY-NC-SA 4.0`, dùng học thuật phi thương mại
- Sample: short sequence on a foggy day
- File: `storage_placeholders/online_data/raw/radiate_sample/tiny_foggy.zip`
- Kích thước: `34,550,966` byte (`32.95 MiB`)
- SHA-256: `395126112A8776BCC8907BC3F754F006351A22D22591B43A751AF71D2BBDC486`
- Số entry ZIP: `461`
- Kích thước sau giải nén: `59,401,670` byte
- Đường dẫn tuyệt đối hoặc chứa `..`: `0`
- Kiểm tra CRC: `PASS`

Chỉ sample công khai đã được tải. Toàn bộ RADIATE vẫn phải đăng ký và chờ nhà cung cấp cấp quyền Dropbox cho email tổ chức.

## Trạng thái dữ liệu

Các archive ở trạng thái `DOWNLOADED`, chưa chuẩn hóa, chưa gán nhãn lại và chưa được đưa vào train/val/test. File gốc được giữ nguyên để bảo toàn provenance.
