# Recommended fix plan

Production changes must be separate from this audit commit.

## Priority 1 — HIGH

### `fix(report)`: correct the UA clipping explanation

- File/line: `data_collection/docs/25_eda_distribution_quality_split.md:19` and generated narrative in `data_collection/scripts/run_external_dataset_eda.py`.
- Cause: the report assumes vehicles entering/leaving frame, while all observed overruns are right/bottom-only and no more than one pixel.
- Proposed patch: retain clipping; describe the pattern as a likely coordinate-convention/off-by-one issue unless image-level evidence proves another cause.
- Risk: documentation may become too definitive in the opposite direction; keep “hypothesis” wording.
- Retest: full UA XML geometry audit and selected boundary contact-sheet review.

### `fix(mapping)`: stratify UA `others` review

Status: **RESOLVED 2026-08-02** — review expanded to all 74 unique tracks; 73 vehicle tracks approved and `MVI_40172 / track 79` rejected through a 201-box track exclusion.

- File/line: `data_collection/configs/vehicle_class_mapping.yaml:31`, `reports/external_eda/ua_others_sample_review.csv`.
- Cause: 12 non-random first-occurrence examples cannot represent 20,641 heterogeneous annotations.
- Proposed patch: add deterministic stratified sampling across sequence, object scale, lighting, and clipping severity; retain sequence/frame/track/reviewer/date/evidence and keep `preserve_original_class=true`.
- Risk: changing `others` inclusion changes training counts and historical metrics.
- Retest: dual-review agreement, mapping conservation, and per-original-class evaluation.

### `fix(leakage)`: complete cross-split near-duplicate audit

- File/line: `data_collection/scripts/detect_external_duplicates.py:58-68`.
- Cause: pHash compares only consecutive sampled images within the same sequence.
- Proposed patch: compare full cross-split candidate sets using scalable pHash/embedding indexing, then SHA-256-confirm exact candidates.
- Risk: quadratic comparisons and false positives without indexing/threshold calibration.
- Retest: planted transformed duplicates plus a reviewed candidate sample; rerun split validation.

### `fix(parser)`: invalidate stale resume caches

- File/line: `data_collection/scripts/run_external_dataset_eda.py:1031-1046`.
- Cause: cache identity contains only dataset and sample size.
- Proposed patch: store and validate schema version, parser hash, mapping/config hash, raw archive identity, Python/package versions, and audited commit.
- Risk: more cache misses and expensive recomputation.
- Retest: unchanged rerun must reuse cache; parser/config/source mutation must reject it; counts must not duplicate.

## Priority 2 — MEDIUM

### `fix(report)`: separate raw, valid, mapped, and sampled totals

- File/line: report generation around `data_collection/scripts/run_external_dataset_eda.py:800-960`.
- Cause: `1,301,866` combines a MIO sample with broader AAU/UA scopes.
- Proposed patch: label it `analysis_scope_bbox_sum` and publish per-dataset raw/valid/excluded/invalid/final fields.
- Risk: downstream documents may expect the old label.
- Retest: conservation equations and cross-report consistency checks.

### `fix(report)`: add figure provenance

- File/line: `data_collection/scripts/run_external_dataset_eda.py:594`, `657`, and `1217-1222`.
- Cause: figures consume in-memory rows without a source-file/hash manifest.
- Proposed patch: save normalized CSVs first, render figures from those files, and write `figure_provenance.csv` with SHA-256 hashes.
- Risk: minor ordering/format differences in charts.
- Retest: clean rerun from the same CSVs must reproduce values and provenance hashes.

### `test(eda)`: promote independent raw invariants

- File: `data_collection/tests/test_independent_audit.py`.
- Cause: the original default suite does not read raw archives.
- Proposed patch: retain a fast tracked-evidence test and run the opt-in raw integration test in a data-enabled CI/manual gate.
- Risk: raw integration is slow and requires local licensed datasets.
- Retest: clean data-enabled run with `K230_RUN_RAW_AUDIT_TESTS=1`.

### `test(eda)`: constrain pytest discovery

- File: add/update repository pytest configuration.
- Cause: root `python -m pytest -q` took 83.24 s although individual test calls were at most 0.01 s; the explicit tests path completed in 0.54 s.
- Proposed patch: set `testpaths = data_collection/tests` and exclude raw/cache/report-reproduction directories from recursion.
- Risk: an incorrect exclusion could hide future tests outside the declared test directory.
- Retest: compare collected node IDs and require the same 47 tests with both commands.

### `docs(eda)`: lock environment and remove local paths

- Files: `data_collection/requirements-data.txt`, a new constraints/lock file, and generated `dataset_inventory.csv`.
- Cause: range-only dependencies and machine-specific absolute paths reduce portability.
- Proposed patch: record a tested dependency lock and emit dataset IDs/configured roots rather than personal absolute paths.
- Risk: platform-specific wheels require separate lock variants.
- Retest: clean environment install and sample-100 reproduction.
