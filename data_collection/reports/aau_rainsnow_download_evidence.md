# Bằng chứng tải AAU RainSnow

- Thời điểm tải/kiểm tra: `2026-07-29T12:26:11+07:00`
- Source ID: `SRC_ONL008`
- Download ID: `DL011`
- Trang nguồn: `https://www.kaggle.com/datasets/aalborguniversity/aau-rainsnow`
- Nhà cung cấp: Aalborg University
- Công cụ: Kaggle CLI `2.2.4`, truy cập dataset công khai không cần token
- Lệnh: `kaggle datasets download -d aalborguniversity/aau-rainsnow`
- License do Kaggle CLI trả về: `Attribution 4.0 International (CC BY 4.0)`
- File gốc: `storage_placeholders/online_data/raw/aau_rainsnow/aau-rainsnow.zip`
- Kích thước: `3,391,982,600` byte (`3,234.85 MiB`)
- SHA-256: `C8C6F114A229762E41EE66C7B8757E438025C93997179874F46C829DF925F0A6`
- Kiểm tra CRC toàn bộ ZIP: `PASS`

## Cấu trúc archive quan sát được

- Tổng entry/file: `26,624`
- PNG: `26,472`
- MKV: `88`
- YAML: `44`
- JSON: `4`
- Notebook: `2`
- Python: `2`
- Windows batch: `4`
- Thumbs.db: `8`
- Tổng kích thước sau giải nén theo central directory: `3,627,862,110` byte

Archive chứa các địa điểm `Egensevej`, `Hadsundvej`, `Hasserisvej`, `Hjorringvej`, `Hobrovej`, `Ostre`, `Ringvej`; đồng thời có cây `aaurainsnow/` lặp lại nội dung cấp gốc. Không giải nén hàng loạt cả hai cây. Khi xử lý cần chọn một cây chuẩn, đối chiếu checksum nội bộ và giữ nguyên ZIP gốc.

## Trạng thái

`DOWNLOADED` — chưa chuyển thành dataset YOLO, chưa đánh dấu nội dung là ground truth của dự án và chưa thực hiện lọc ảnh theo ROI/làn khẩn cấp.
