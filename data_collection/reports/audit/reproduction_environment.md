# Reproduction environment

Audit date: 2026-08-02 (Asia/Bangkok)
Audited commit: `80e66fb`

## Runtime

- OS: Windows 11 (`10.0.26200`), 64-bit
- Python: `3.12.10`
- Python executable: `C:\Users\LEGION\AppData\Local\Programs\Python\Python312\python.exe`
- NumPy: `2.4.6`
- OpenCV: `4.13.0`
- Pillow: `12.3.0`
- PyYAML: `6.0.3`
- Matplotlib: `3.11.1`
- pytest: `9.1.1`
- `ffmpeg`: not found on `PATH`
- `ffprobe`: not found on `PATH`

## Determinism and dependencies

- Production requirements are version-range constrained; there is no tested lock/constraints file.
- Independent audit seeds: MIO `230`, AAU `231`, UA `232`, bbox reservoir `233`.
- Text input uses UTF-8/UTF-8-SIG where appropriate.
- The production CLI accepts `--workers`, but the audited execution path does not use it.
- Resume cache names include dataset and sample size only. They do not fingerprint the parser, mapping config, source archive, or audited commit.

These limitations mean a future environment or stale cache may produce a result that appears reproducible without being equivalent to this audit.

## Observed performance

- Independent MIO raw audit: `101.004 s` for 351,549 rows (about 3,480 boxes/s).
- Independent AAU raw audit: `6.262 s` for 13,297 rows (about 2,123 boxes/s), excluding separate visual review time.
- Independent UA raw audit: `162.281 s` for 1,274,055 boxes (about 7,851 boxes/s).
- Leakage/duplicate audit: `9.271 s`.
- Python `tracemalloc` peak: `370,739,893` bytes; externally observed process working set peaked at approximately `861 MB`.
- Production sample 100 fresh/resume: approximately `56.0 s` / `5.505 s`.
- Production sample 5,000 fresh/resume: `145.2 s` / `6.692 s`.
- Final root test command: `46 passed, 1 skipped` in `83.24 s`; explicit `data_collection/tests` path: same result in `0.54 s` pytest time.

A complete all-image quality scan was not run. It would decode roughly 277,874 MIO+UA still images plus AAU video samples and add substantial I/O. The required sample-100, sample-5,000, full UA/AAU annotation pass, and independent raw count audit were completed; all-image quality and full near-duplicate conclusions remain unverified.
