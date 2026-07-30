# 06. Quy tắc đặt tên

## Mục lục
- [Mẫu tên](#mẫu-tên)
- [Quy tắc](#quy-tắc)
- [Checklist](#checklist)

## Mẫu tên
- Video: `YYYYMMDD_location_device_condition_session_sequence.mp4`
- Ví dụ: `20260730_gate_k230_day_s001_001.mp4`
- Ảnh: `YYYYMMDD_location_device_condition_session_frame.jpg`
- Ví dụ: `20260730_gate_k230_day_s001_f000120.jpg`
- Nhãn: `20260730_gate_k230_day_s001_f000120.txt`
- Session: `SYYYYMMDD_XXX`, ví dụ `S20260730_001`

## Quy tắc
Tên file chữ thường (trừ session ID trong metadata), ASCII không dấu, không khoảng trắng; dùng `_`. Stem nhãn phải giống ảnh. ID không tái sử dụng.

## Checklist
- [ ] Ngày và session khớp metadata.
- [ ] Condition thuộc cấu hình.
- [ ] Không chứa tên người hay thông tin cá nhân.

