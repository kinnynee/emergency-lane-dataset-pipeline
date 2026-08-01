# So sánh dataset phục vụ nghiên cứu

| Dataset | Vai trò phù hợp | Điểm mạnh | Hạn chế chính |
|---|---|---|---|
| MIO-TCD Localization | Train detection | Nhiều ảnh và class phương tiện; có motorcycle | Ảnh tĩnh, không Track ID, không nhãn xe dừng/điều kiện chi tiết |
| AAU RainSnow | Train/validation adverse weather | Camera cố định, RGB/thermal, video và COCO instance annotation | Weather chính xác theo sequence cần review; frame liên tiếp dễ trùng |
| UA-DETRAC Original | Train, tracking analysis, cross-domain test | 100 sequence, XML Track ID, weather/camera state | Không có nhãn xe dừng trong ROI; không thấy motorcycle trong class XML |

Kết luận định lượng phải lấy từ `reports/external_eda/*.csv`, không lấy từ bảng mô tả này.

## Road type cho cross-dataset test

`road_type` được gán ở cấp sequence, không gán ngẫu nhiên theo frame. Các giá trị được phép là `HIGHWAY`, `URBAN_ROAD`, `INTERSECTION`, `EMERGENCY_LANE_LIKE` và `UNKNOWN`. Mapping đã review ảnh đại diện nằm trong `configs/sequence_road_types.yaml`; mọi đánh giá vẫn cần Data Lead xác nhận trước khi áp dụng split.

Cùng mapping này lưu thêm `weather`, `lighting`, `camera_view` và `traffic_density`. Nhãn thời tiết không dùng giá trị `night`; khi XML UA-DETRAC ghi `night`, giá trị đó được chuyển sang trường ánh sáng và thời tiết được giữ `UNKNOWN`. Chi tiết phương pháp và kết quả review nằm trong `docs/24_cross_test_scene_metadata_review.md`.
