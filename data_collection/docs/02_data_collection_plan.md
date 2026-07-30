# 02. Kế hoạch thu thập dữ liệu 12 tuần

## Mục lục
- [Lộ trình](#lộ-trình)
- [Vai trò](#vai-trò)
- [Checklist](#checklist)

## Lộ trình
| Tuần | Thời gian | Đầu ra dữ liệu |
|---|---|---|
| 1 | 27-31/07 | Khảo sát, xin phép, spec v0, quy trình nhãn |
| 2 | 03-07/08 | Bắt đầu quay K230, baseline và replay v0 |
| 3 | 10-14/08 | dataset-v0 khoảng 500 ảnh đã gán nhãn |
| 4 | 17-21/08 | Bổ sung kịch bản dừng/chạy và dữ liệu khó |
| 5 | 24-28/08 | dataset-v1 >=1.500 ảnh; test >=300 ảnh khóa |
| 6 | 31/08-04/09 | Hỗ trợ model-v1, so sánh cùng video test |
| 7 | 07-11/09 | Thu lỗi thực địa, log sự kiện |
| 8 | 14-18/09 | Dataset cho demo tích hợp và đo latency |
| 9 | 21-25/09 | Chiến dịch đo #1, bổ sung đêm/ngược sáng nếu thiếu |
| 10 | 28/09-02/10 | Chiến dịch đo #2, chốt bảng số liệu |
| 11 | 05-09/10 | Dataset card, báo cáo và evidence cuối |
| 12 | 12-18/10 | Đóng gói nghiệm thu, bài báo v1, buffer |

## Vai trò
SV1 chịu trách nhiệm dữ liệu; SV2 hỗ trợ yêu cầu model; SV3 xác định kịch bản dừng; SV4 hỗ trợ evidence cloud; SV5 quản lý protocol và số liệu. Mỗi task chỉ `DONE` khi có evidence.

## Checklist
- [ ] Kế hoạch tuần có owner, dependency, đầu ra.
- [ ] Mỗi tuần rà thiếu điều kiện và rủi ro.
- [ ] Mốc trượt hơn một tuần được báo GVHD.

