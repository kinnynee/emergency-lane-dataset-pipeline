# LDKC-14 — bộ thu video kiểm thử thực địa

## Thành phần cần giao

Mỗi video quay bằng chính camera K230, cố định ở vị trí lắp thật, có:

1. Tệp video gốc và một dòng trong `video_manifest.csv` với FPS thực tế lúc quay và thời lượng theo timestamp media.
2. Tệp ROI JSON cho làn khẩn cấp và vùng loại trừ. Dùng đúng cấu trúc của `roi_nguoive_v2.json` sau khi có tệp mẫu; tọa độ phải lấy trên khung hình của chính video đó.
3. Các dòng sự kiện trong `su_kien.csv`.
4. Tối thiểu ba dòng khung bao trong `khung_bao.csv` cho mỗi lần dừng: khung đầu, giữa và cuối.

## Chỉ tiêu kiểm tra

- Ít nhất 60 lần xe dừng trong làn khẩn cấp, mỗi lần từ 10 giây trở lên; có các lần 30–120 giây.
- Trong các lần dừng hợp lệ: ít nhất 20 ban ngày và 20 ban đêm.
- Ít nhất 5 giờ giao thông bình thường được kiểm tra là không có xe dừng trong làn.
- Các video này chỉ thuộc tập **kiểm thử**; không đưa vào tập train.

## Quy ước ghi nhãn

- `khung_bat_dau` và `khung_ket_thuc` là chỉ số khung hình của tệp video giao nộp, bắt đầu từ 0, tính cả hai đầu. Nếu quy ước của dự án khác, cần chuyển đồng bộ trước khi giao.
- `id_xe` giữ ổn định cho cùng chiếc xe trong một video. Mỗi lần dừng là một dòng sự kiện riêng, kể cả khi cùng xe dừng nhiều lần.
- `loai_video` là `su_kien` hoặc `binh_thuong_khong_dung`. Chỉ ghi `xac_nhan_khong_dung=co` sau khi đã xem hết video bình thường. `tap_du_lieu` luôn là `TEST_ONLY`.
- `trong_lan` ghi `co` hoặc `khong` theo ROI đã chốt.
- `dieu_kien` dùng các giá trị `ngay`, `dem`, `mua`, `nguoc_sang`; nếu đồng thời có nhiều điều kiện, ngăn cách bằng `|` (ví dụ `ngay|mua`).
- Khung bao là tọa độ pixel `(x_min, y_min, x_max, y_max)` trên khung hình gốc; `vi_tri` là `dau`, `giua` hoặc `cuoi`.
- `dau` ứng với `khung_bat_dau`, `giua` ứng với phần nguyên của trung điểm, `cuoi` ứng với `khung_ket_thuc`.
- Thời lượng sự kiện được tính từ mốc thời gian thực của video. Nếu FPS biến thiên, không suy thời lượng chỉ bằng số khung chia FPS danh nghĩa.
- Giữ bản gốc và mọi nhãn ở ngoài đường dẫn/tập train.

## Điều kiện thu thập

Chỉ tổ chức dừng xe có chủ đích tại khu vực thử nghiệm được phép và đã kiểm soát giao thông. Trên làn khẩn cấp đang khai thác, ghi hình các tình huống xảy ra tự nhiên; không tạo thêm nguy cơ cho người đi đường để lấy dữ liệu.

## Trạng thái

Biểu mẫu đã sẵn sàng nhưng chưa có video, ROI mẫu hoặc nhãn sự kiện thực. Cần đối chiếu `roi_nguoive_v2.json` trước khi tạo tệp ROI chính thức.

Khi có dữ liệu, chạy `python LDKC-14/validate.py` để kiểm tra các chỉ tiêu định lượng và tính đủ của nhãn. Kiểm tra tự động không thay thế việc xem video để xác nhận làn đường, điều kiện thời tiết và không có xe dừng.
