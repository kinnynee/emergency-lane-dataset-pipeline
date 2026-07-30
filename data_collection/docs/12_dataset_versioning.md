# 12. Version hóa dataset

## Mục lục
- [Các phiên bản](#các-phiên-bản)
- [Quy trình phát hành](#quy-trình-phát-hành)
- [Checklist](#checklist)

## Các phiên bản
- `dataset-v0.1`: pipeline thử nghiệm, ít ảnh, kiểm tra quy trình.
- `dataset-v0.2`: sửa lỗi nhãn, bổ sung điều kiện thiếu.
- `dataset-v1.0`: tối thiểu 1.500 ảnh đã kiểm tra; test khóa; có EDA, dataset card, changelog; sẵn sàng train model-v1.

## Quy trình phát hành
Đóng băng manifest/split, chạy QC/EDA, lập card và changelog, ghi checksum/đường dẫn kho ngoài Git, reviewer duyệt. Không sửa âm thầm bản đã phát hành; thay đổi dữ liệu tạo phiên bản mới.

## Checklist
- [ ] Phiên bản và ngày nhất quán.
- [ ] Test manifest khóa.
- [ ] Nguồn/quyền/evidence truy vết được.

