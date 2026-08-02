# UA-DETRAC YOLO export smoke test

- Date: 2026-08-02
- Source: local raw `ua-detrac-orig.zip`
- Command scope: first 200 frames processed by the production exporter
- Exported image/label pairs: 200 / 200
- Exported vehicle boxes: 605
- Boxes clipped to image bounds and retained: 31
- Annotation manifest rows: 605
- Image and label stems matched: PASS
- Every YOLO class ID was `0` (`vehicle`): PASS
- Every normalized center/width/height value was inside `[0,1]`: PASS
- Every normalized width and height was positive: PASS
- `preserve_original_class`: true

The generated images and labels are intentionally kept outside Git. This report proves the exporter exercised real UA-DETRAC media and wrote clipped training labels; it is not a claim that a full export has completed.
