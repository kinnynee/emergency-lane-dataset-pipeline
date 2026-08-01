# Review metadata cảnh cho cross-test

Ngày review: 01/08/2026. Phạm vi: 10 sequence đang được đề xuất cho `CROSS_DATASET_TEST`. Đây là metadata ở cấp sequence và vẫn chờ Data Lead phê duyệt trước khi áp dụng split thật.

## Quy ước

- `road_type`: `HIGHWAY`, `URBAN_ROAD`, `INTERSECTION`, `EMERGENCY_LANE_LIKE`, `UNKNOWN`.
- `weather`: `CLEAR`, `CLOUDY`, `RAIN`, `RAIN_OR_WET_ROAD`, `SNOW`, `UNKNOWN`. Giá trị kết hợp được dùng khi ảnh chứng minh mặt đường ướt nhưng chưa đủ bằng chứng để khẳng định đang mưa.
- `lighting`: `DAY`, `TWILIGHT`, `NIGHT`, `UNKNOWN`.
- `camera_view`: góc quan sát hình học; cả 10 sequence hiện là `ELEVATED_OBLIQUE`.
- `traffic_density`: `LOW` nếu trung bình không quá 4 bbox xe/ảnh, `MEDIUM` nếu trên 4 đến 10, `HIGH` nếu trên 10.

AAU được kiểm tra bằng ba ảnh RGB đầu/giữa/cuối và thống kê annotation trên 100 ảnh RGB của mỗi sequence. UA-DETRAC dùng ảnh đại diện, metadata XML và toàn bộ bbox/frame của sequence. Giá trị XML `night` được dùng cho `lighting`, không dùng làm thời tiết.

## Kết quả

| Dataset | Sequence | Road type | Weather | Lighting | Camera view | Traffic | Mean box/image |
|---|---|---|---|---|---|---|---:|
| AAU RainSnow | Hjorringvej-4 | INTERSECTION | RAIN_OR_WET_ROAD | NIGHT | ELEVATED_OBLIQUE | MEDIUM | 6.48 |
| AAU RainSnow | Ringvej-3 | INTERSECTION | RAIN_OR_WET_ROAD | TWILIGHT | ELEVATED_OBLIQUE | LOW | 3.88 |
| UA-DETRAC | MVI_39311 | INTERSECTION | CLEAR | DAY | ELEVATED_OBLIQUE | HIGH | 14.92 |
| UA-DETRAC | MVI_39401 | INTERSECTION | CLEAR | DAY | ELEVATED_OBLIQUE | HIGH | 10.18 |
| UA-DETRAC | MVI_39781 | URBAN_ROAD | UNKNOWN | TWILIGHT | ELEVATED_OBLIQUE | MEDIUM | 4.10 |
| UA-DETRAC | MVI_39851 | URBAN_ROAD | UNKNOWN | NIGHT | ELEVATED_OBLIQUE | LOW | 3.61 |
| UA-DETRAC | MVI_40191 | HIGHWAY | CLOUDY | DAY | ELEVATED_OBLIQUE | HIGH | 15.39 |
| UA-DETRAC | MVI_40751 | URBAN_ROAD | UNKNOWN | TWILIGHT | ELEVATED_OBLIQUE | MEDIUM | 6.45 |
| UA-DETRAC | MVI_40851 | INTERSECTION | CLOUDY | DAY | ELEVATED_OBLIQUE | HIGH | 13.39 |
| UA-DETRAC | MVI_41063 | HIGHWAY | CLEAR | DAY | ELEVATED_OBLIQUE | MEDIUM | 6.68 |

## Kết luận ngắn

Road type cũ được giữ nguyên sau vòng review thứ hai. Cross-test có đa dạng ánh sáng và mật độ xe, nhưng chưa có sequence `EMERGENCY_LANE_LIKE`. Ba sequence UA-DETRAC ban đêm/chạng vạng không có nhãn thời tiết đáng tin cậy nên được giữ `UNKNOWN`.
