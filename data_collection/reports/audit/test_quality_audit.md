# Test quality audit

## Conclusion

The original suite collected 45 tests and all passed, but that result does **not** prove the raw-data EDA is correct. The default suite does not need the MIO TAR, AAU videos/JSON, or UA ZIP/XML and therefore cannot independently detect a wrong raw count, stale cache, incomplete archive, or semantic label error.

## Classification of the original 45 tests

- **Unit tests:** bbox validation/clipping, class mapping, deterministic sequence split, discovery filters, cache helper behavior, and small EDA helper functions.
- **Synthetic/data-validation tests:** functions are exercised with in-memory dictionaries, temporary paths, or tiny fabricated inputs.
- **Tracked-report regression tests:** verify that expected files and selected aggregate values already committed to Git are present.
- **Raw-data integration tests:** none in the original 45-test suite.
- **Idempotency tests:** cache reuse is tested at helper level, but a complete clean-output production run followed by `--resume` is not covered.

## Claims the original suite does not prove

- It does not assert the independently parsed total `1,301,866`.
- It does not assert the exact UA out-of-bounds count `130,181`; the handoff test only checks that it is greater than zero.
- AAU `10 DAY / 11 NIGHT / 1 TWILIGHT` is asserted from committed configuration/report rows, not derived from a review of raw videos.
- `preserve_original_class` policy is checked, but end-to-end converted training labels are not validated against raw source labels.
- RADIATE exclusion is tested at discovery level; this does not prove every generated artifact is free of stale RADIATE rows.
- `K230_BACKLIT = NOT_AVAILABLE` is not validated end to end with ground truth and prediction absence.
- Semantic correctness of UA `others` and clipped boxes is not testable from structural XML checks alone.

## Audit tests added

- A fast audit-evidence regression test checks count conservation, the exact total, exact UA boundary count, the one-pixel maximum overrun, and absence of left/top overruns.
- An opt-in raw archive integration test reruns the independent parser when `K230_RUN_RAW_AUDIT_TESTS=1`. It is skipped by default because it is a multi-minute, high-I/O check.

Passing the revised suite means the code and tracked audit evidence are internally consistent. It still does not replace manual visual review or independent raw integration execution.

## Test execution performance

`python -m pytest -q` passed with `46 passed, 1 skipped` but took `83.24 s`; pytest reported every listed test at `<=0.01 s`. Running the same suite with the explicit path (`python -m pytest data_collection/tests -q`) took `0.54 s` inside pytest. This is consistent with repository-wide discovery traversing data-heavy local directories. Add pytest `testpaths`/`norecursedirs` configuration or use the explicit test path.
