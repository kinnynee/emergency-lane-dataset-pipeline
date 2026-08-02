# EDA Audit Executive Summary

- Audited commit: `80e66fb`
- Independent raw total matching the current mixed analysis scope: **1,301,866**
- Difference from 1,301,866: **+0**
- UA out-of-bounds boxes: **130,181**; difference from 130,181: **+0**
- AAU lighting decision counts: **{'NIGHT': 11, 'DAY': 10, 'TWILIGHT': 1}**
- Claim status: **{'PARTIALLY_VERIFIED': 4, 'VERIFIED': 4, 'NOT_VERIFIED': 1, 'INCORRECT': 1}**
- Finding severity: **{'HIGH': 5, 'MEDIUM': 6, 'LOW': 1}**

## Direct conclusions

- Passing 45 tests is **not sufficient** to prove the EDA is correct.
- The 1,301,866 total is reproducible, but it mixes a deterministic MIO sample with full AAU-valid and full UA XML counts.
- The UA boundary count is reproducible; structural validity after clipping does not prove semantic correctness. Clip severity must be reviewed.
- All UA overruns are right/bottom-only and at most one pixel. The reported entering/leaving-frame explanation is unsupported; a coordinate-convention issue is more plausible.
- The 12 `others` rows do not provide enough evidence to retain the entire heterogeneous class without caution.
- AAU 10/11/1 is recorded and the raw videos exist, but commit `80e66fb` lacks a named reviewer and retained original review evidence.
- Sequence and exact-content leakage checks pass only within their stated scope; cross-split near-duplicate leakage remains unverified.
- `K230_BACKLIT` is correctly `NOT_AVAILABLE`.
- Clean production runs analyzed 3/3 datasets. Sample 5,000 reproduced **1,301,866**; resume reused all three caches and changed only the timestamped executive summary.

See `FULL_EDA_AUDIT_REPORT.md`, `audit_findings.csv`, and `claim_verification.csv` for evidence and limitations.
