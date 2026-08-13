# UA-DETRAC replay protocol

`replay_ua_detrac_alerts.py` dùng track ID và bbox từ XML UA-DETRAC qua đúng
state machine `TrackState → update_speed_kmh → open_stop_event/close_event` của
`host_yolo_loop.py`. ROI là giả lập và video overlay có nhãn
`FAKE ROI - UA REPLAY ONLY`; kết quả không phải chứng cứ triển khai K230.

Ví dụ lệnh:

```powershell
python data_collection/scripts/replay_ua_detrac_alerts.py `
  --ua-xml <MVI_xxxxx.xml> --speed-calibration <synthetic_replay_calibration.json> `
  --frame-size 960x540 --fake-roi 0.55,0.45,0.98,0.92 --fps 25 `
  --alerts reports/ua_replay/alerts.csv --metrics reports/ua_replay/metrics.json `
  --images-dir <MVI_xxxxx> --annotated-video reports/ua_replay/replay.mp4
```

Để đo `false_alerts_per_hour`, `missed_stopped_vehicles` và
`detection_to_alert_latency_s`, phải thay template
`planning/ua_replay_expected_stop_events_template.csv` bằng các khoảng thời
gian xe dừng đã review và đặt `approval_status=APPROVED`. Nếu không có nhãn đó,
script bắt buộc ghi `NOT_MEASURED_MISSING_APPROVED_STOP_GROUND_TRUTH` thay vì
suy diễn metric từ track XML.
