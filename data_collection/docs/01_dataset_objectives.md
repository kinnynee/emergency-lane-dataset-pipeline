# 01. Mục tiêu dataset

## Mục lục
- [Mục tiêu](#mục-tiêu)
- [Phạm vi](#phạm-vi)
- [Checklist](#checklist)

## Mục tiêu
Tạo dataset truy vết được để huấn luyện và đánh giá YOLOv8 nhận diện `vehicle` từ góc camera K230, làm đầu vào cho logic xác định xe dừng trong ROI.

## Phạm vi
Ưu tiên dữ liệu thật tại demo; bao phủ ngày, đêm, mưa/đường ướt, ngược sáng, xe gần/xa, che khuất và negative. Dataset không tự chứng minh trạng thái dừng từ một ảnh; trạng thái sự kiện phải dựa trên chuỗi video và protocol riêng.

## Checklist
- [ ] Nguồn, quyền và điều kiện được ghi.
- [ ] Test tối thiểu 300 ảnh, khóa cứng.
- [ ] Không có dữ liệu thật trong Git.

