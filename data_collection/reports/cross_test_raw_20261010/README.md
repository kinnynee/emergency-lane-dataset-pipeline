# Cross-test evidence, 10 October 2026

This directory contains compact audit results derived from the user-supplied `PHAT_HIEN_THO_CROSS_TEST_20261010.zip`. The two prediction pickle files and the full exported image manifest stay outside Git.

- [Audit and limitations](cross_test_raw_detection_audit.md)
- [Per-sequence metrics](raw_detection_recall.csv)
- [Input hashes and structure](raw_detection_audit.json)
- [AP@50 per sequence after UA ignore](map50_after_ua_regions.csv) and [aggregate means](map50_after_ua_regions_summary.json)
- [Export count reconciliation](dataset_export_counts.json)
- [UA source ignore rectangles](ua_ignore_regions.csv)

Recompute with NumPy 2.x and the original prediction ZIP and export metadata:

```powershell
python data_collection/scripts/audit_cross_test_raw_detections.py `
  <PHAT_HIEN_THO_CROSS_TEST_20261010.zip> `
  <dataset-v1-full-reexport/metadata/images.csv> `
  data_collection/reports/cross_test_raw_20261010/ua_ignore_regions.csv `
  <output-directory>
```

The operating-point script verifies both pickle SHA-256 values before restricted loading. `raw_detection_recall.csv` records guard-25 measurements; for publication metrics use `fp_after_ua_regions` and `precision_after_ua_regions_pct`. The experiment owner confirmed all 140 GT/TP cells and all 100 H1/H3 FP-after-ignore cells against the official evaluator. The separate [AP@50 script](../../scripts/audit_cross_test_map50.py) recomputes sequence AP from all saved score≥0.01 detections, with UA ignore and no guard; its results match the 09 October aggregates and MVI_39311 example. [The source note](../references/SO_LIEU_09_10_2026.md) links the original kmodel and clustered McNemar JSON. Neither script reproduces kmodel simulation, McNemar, or board FPS.
