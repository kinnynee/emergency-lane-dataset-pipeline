# Full EDA Audit Report

## Scope and independence

This is a read-only audit of commit `80e66fb`. The independent script parses the raw MIO TAR, AAU JSON/videos, and UA ZIP/XML without importing the production dataset inspectors. Existing reports were compared only after raw counts were established. No production EDA logic or source data was changed.

## Dataset discovery and count reconciliation

All three intended datasets were found; RADIATE was not selected.

| Dataset | Raw boxes | Valid before clip | Excluded | Invalid | Final analysis scope |
|---|---:|---:|---:|---:|---:|
| MIO-TCD Localization | 351,549 | 351,544 | 11,738 | 5 | 16,017 |
| AAU RainSnow | 13,297 | 11,794 | 1,230 valid-class exclusions | 1,503 | 11,794 |
| UA-DETRAC Original | 1,274,055 | 1,274,055 | 0 | 0 structural | 1,274,055 |

The reported sum `1,301,866` is reproduced exactly (`difference=0`) as `16,017 + 11,794 + 1,274,055`. It is not a full-raw-data total: MIO is the deterministic 5,000-image analysis sample, while AAU and UA use broader/full annotation scopes. The number must be named an **analysis-scope sum**.

## Class mapping

The configured policy keeps motorized vehicles, including motorcycles/motorbikes, and excludes pedestrians, persons, bicycles, and non-motorized vehicles. `preserve_original_class=true` is present and reports/manifests retain an original-class column. The target detection class is `0: vehicle`.

This is only partially verified end to end: the audit checked raw class counts, mapping configuration, and output schemas, but did not train or reconvert every raw annotation. UA `others` remains heterogeneous and cannot be considered proven vehicle ground truth from 12 examples.

## AAU lighting and annotation scope

Raw inventory confirms 22 sequences and 44 videos. Configuration and report counts agree at `DAY=10`, `NIGHT=11`, `TWILIGHT=1`. The audit regenerated four contact sheets and visually rechecked all 22 decisions; no obvious mismatch was found. Brightness is not used as the official lighting decision in the current pipeline.

The audited commit still lacks the original named reviewer, review date, and retained evidence/checksum. Therefore the historical claim is `PARTIALLY_VERIFIED`, not fully auditable from Git alone.

AAU does contain detection bounding boxes. Independently parsed annotation counts are raw `13,297`, structurally valid `11,794`, and invalid `1,503`; this dataset is not treated as image-only classification data.

## UA boundary clipping

The independent full XML pass found exactly `130,181` out-of-bounds boxes. All remain positive after clipping; none is fully outside the image, none becomes smaller than 2 pixels, and no structural XML/class/frame error was found under the implemented checks.

The geometry is highly specific:

- LEFT overruns: `0`
- TOP overruns: `0`
- RIGHT overruns: `79,522`
- BOTTOM overruns: `60,465`
- Maximum overrun: `1.0` pixel
- `CLIP_MINOR`: `130,178`; `CLIP_MODERATE`: `3`; `CLIP_SEVERE`: `0`

Clipping to image bounds is numerically safe for this observed set and deleting these boxes would be unjustified. However, the repository statement that they are caused by vehicles entering/leaving the frame is not supported by this pattern. A one-pixel coordinate convention/off-by-one difference is a more plausible hypothesis. “Zero errors after clipping” is only true for the audited structural checks; it does not prove semantic correctness.

## UA `others` review

The 12 listed examples were reconstructed with sequence, frame, and track ID and visually reviewed from a regenerated contact sheet. Two were clearly motorized vehicles, two were likely vehicles, and eight remained indeterminate because of scale or boundary truncation. The original selection is non-random and covers first occurrences from only 12 sequences.

This evidence does not justify automatically retaining all `20,641` `others` annotations as vehicles. Current recommendation: `KEEP_WITH_CAUTION`, preserve the original class, and conduct a stratified review across sequences, scale, lighting, and clipping severity before using the class as research ground truth.

## Split leakage and duplicates

- Sequence exclusivity: passed.
- Source-path exclusivity: passed.
- Exact cross-split content: passed for full AAU RGB SHA-256 and full UA archive CRC/size candidates confirmed by SHA-256.
- MIO cross-split check: not applicable because MIO is train-only.
- Full near-duplicate cross-split check: not verified.

The production pHash check operates on sampled images and compares only consecutive images within the same sequence. It cannot prove the absence of transformed or non-adjacent near duplicates across splits. Therefore the broad claim “no leakage” is only partially verified.

## Reports, figures, Git, and reproducibility

- The main numeric totals and AAU condition table agree with independently checked evidence within the scope stated above.
- The UA causal explanation is inconsistent with the observed one-pixel right/bottom-only pattern.
- Figures are created from in-memory rows and have no manifest connecting each image to exact source CSV hashes.
- Production resume caches include dataset/sample-size names but no parser, config, archive, or commit fingerprint.
- `--workers` is exposed but unused in the audited execution path.
- Dependencies are range-constrained without a tested lock file.
- Tracked inventory reports contain local absolute Windows paths.
- Git contains no raw datasets/videos and no discovered token, cookie, or API key. Repository Git objects are approximately 14 MiB; the largest tracked artifact is approximately 1.74 MiB.

## Test quality

The original 45/45 passing tests mainly cover units, synthetic inputs, and committed report regressions. They do not require raw archives and cannot independently prove `1,301,866`, `130,181`, semantic `others` validity, or complete near-duplicate leakage absence. Audit-only tests add exact evidence invariants plus an opt-in multi-minute raw integration check. See `test_quality_audit.md`.

## K230 BACKLIT and research validity

`K230_BACKLIT` is correctly `NOT_AVAILABLE`: no real K230 backlit media, ground truth, predictions, or evaluable split was found. It must not be converted to a zero score.

External datasets support general vehicle detection and domain analysis. They do not establish stopped-vehicle ground truth in an emergency lane and cannot replace a locked, high-mounted K230 field test. Current results are suitable as engineering EDA evidence only after the report wording is corrected; HIGH findings must be addressed before using the dataset conclusions as strong research claims.

## Direct answers

1. **Do 45/45 tests prove the EDA correct?** No.
2. **Is 1,301,866 reproducible?** Yes, exactly, but only as a mixed analysis-scope sum.
3. **Are 130,181 UA boxes handled correctly?** The count and clipping are verified; deleting them would be wrong. The stated cause and semantic correctness are not verified.
4. **Do 12 `others` samples justify keeping the full class?** No.
5. **Is AAU 10/11/1 supported?** Counts and the new visual recheck agree, but historical review provenance is incomplete.
6. **Is class mapping safe?** Core motorized/person/bicycle policy is consistent; UA `others` remains a material uncertainty.
7. **Is there leakage?** No sequence/source/exact-content leakage was found in the audited scope; near-duplicate cross-split leakage remains unverified.
8. **Were images/annotations double-counted?** No duplicate counting was found in the independent raw parsers; the reported total is nevertheless a mixed-scope metric.
9. **Is K230_BACKLIT NOT_AVAILABLE?** Yes.
10. **Is this ready for a research report?** Not as an unqualified final claim. Resolve HIGH findings or clearly disclose them.
