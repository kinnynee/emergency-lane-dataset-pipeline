# 14. Kế hoạch tìm dữ liệu Internet

## Mục lục
- [Mục đích](#mục-đích)
- [Thứ tự ưu tiên](#thứ-tự-ưu-tiên)
- [Quy trình](#quy-trình)
- [Checklist](#checklist)

## Mục đích
Dữ liệu Internet chỉ bổ sung khoảng trống mà K230 không tự thu an toàn: đêm, mưa, ngược sáng, xe tải/khách, xe xa, nhiều xe và góc camera cố định. Không thay thế nguồn K230.

## Thứ tự ưu tiên
Website đại học/phòng thí nghiệm, trang dataset/bài báo chính thức, kho nghiên cứu, release chính thức tác giả, nền tảng có license rõ, stock có license rõ, rồi mới đến nguồn cần xin phép. Không dùng blog sao chép làm nguồn gốc.

## Quy trình
Điền truy vấn → ghi nguồn vào `planning/online_source_candidates.csv` → kiểm tra trang license và điều khoản tải → reviewer quyết định → chỉ đưa hàng đủ điều kiện vào queue → tải/kiểm tra/chuẩn hóa/EDA. YouTube và website chia sẻ video chỉ được ghi URL với `NEEDS_PERMISSION` hoặc `LICENSE_UNVERIFIED` trừ khi có phép rõ bằng văn bản.

## Checklist
- [ ] Không bypass đăng nhập, paywall, chống bot hoặc cookie cá nhân.
- [ ] Không tự suy đoán license từ tên website.
- [ ] Có evidence trang chính thức, ngày truy cập và reviewer.
