# Lựa chọn nguồn dữ liệu — 31/07/2026

## Đang sử dụng

1. **MIO-TCD Localization** — dữ liệu localization có bounding box, camera giao thông cố định.
2. **AAU RainSnow** — bổ sung điều kiện mưa, tuyết, ban đêm và camera cố định.
3. **UA-DETRAC Original** — toàn bộ bản Kaggle `bratjay/ua-detrac-orig`, phù hợp góc camera trên cao/cố định.

## Không sử dụng

- **MIO-TCD Classification** — loại khỏi pipeline theo quyết định chỉ dùng bản Localization.
- **RADIATE** — loại vì dữ liệu dùng camera phía trước xe, không phù hợp góc camera trên cao.

Các archive đã tải trước đây vẫn được giữ trong `raw` để tránh mất dữ liệu, nhưng pipeline phải đọc danh sách trong `configs/active_dataset_sources.yaml` và không xử lý các mục thuộc `excluded_sources`.
