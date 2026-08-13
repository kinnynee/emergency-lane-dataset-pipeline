# Distance and ROI calibration

The K230 does not have one fixed conversion such as `25 px = X m`. Perspective means that physical pixel scale changes with image position. The 25 px guard is a model-input-quality guard only; it must not be reported as the distance at which a vehicle is detected.

## Measuring vehicle distance in metres

1. Keep the K230 fixed and use the exact stream resolution that will run in production.
2. Measure four non-collinear points on the road in metres, including the image pixel coordinate and the matching ground coordinate, in the same order.
3. Measure `camera_ground_point_m`: the point on the road directly below the camera in that same coordinate system.
4. Copy `speed_calibration.template.json` outside Git and enter those values. Placeholder zeroes are deliberately invalid for a real calibration.
5. Run the detector with that calibration. The overlay and event CSV report the ground-plane distance from the bottom centre of a vehicle box to `camera_ground_point_m`.

For example, coordinates `(0, 0)`, `(3.5, 0)`, `(3.5, 20)`, `(0, 20)` describe a 3.5 m × 20 m road patch, but they only yield a camera distance when `camera_ground_point_m` is also measured. Never reuse one camera's calibration for a different camera position, lens, crop, or resolution.

## Operational ROI versus public-data proxy

`emergency_lane_roi.template.json` is intentionally a draft and cannot be used by the detector. Copy it outside Git, trace the emergency-lane boundary on the deployment camera, record the reviewer, then set `status` to `CALIBRATED`. A calibration config is bound to its `camera_id` and `frame_size_px`; the application rejects a stream with a different size.

Use it as follows:

```powershell
python host_yolo_loop.py `
  --video ".\evidence\k230\lane_a.mp4" `
  --model ".\runs\final\weights\best.pt" `
  --speed-calibration ".\evidence\k230\calibration\lane_a.json" `
  --roi-config ".\evidence\k230\calibration\lane_a_roi.json" `
  --classes 0
```

The existing public-dataset ROI proxy is only for exploratory statistics. It must be labelled as a proxy and can favour near/bottom-of-frame vehicles, which makes a low `<25 px` failure rate optimistic. Recompute the statistic after the real K230 ROI is calibrated; do not represent proxy results as deployment performance.
