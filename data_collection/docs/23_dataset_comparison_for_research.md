# So sánh dataset phục vụ nghiên cứu

| Dataset | Vai trò phù hợp | Điểm mạnh | Hạn chế chính |
|---|---|---|---|
| MIO-TCD Localization | Train detection | Nhiều ảnh và class phương tiện; có motorcycle | Ảnh tĩnh, không Track ID, không nhãn xe dừng/điều kiện chi tiết |
| AAU RainSnow | Train/validation adverse weather | Camera cố định, RGB/thermal, video và COCO instance annotation | Weather chính xác theo sequence cần review; frame liên tiếp dễ trùng |
| UA-DETRAC Original | Train, tracking analysis, cross-domain test | 100 sequence, XML Track ID, weather/camera state | Không có nhãn xe dừng trong ROI; không thấy motorcycle trong class XML |

Kết luận định lượng phải lấy từ `reports/external_eda/*.csv`, không lấy từ bảng mô tả này.
