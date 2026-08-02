# UA-DETRAC `others` — Data Lead review

Review date: 2026-08-02  
Reviewer: `CODEX_ACTING_DATA_LEAD`  
Decision: `APPROVED_WITH_TRACK_EXCLUSION`

## Scope and evidence

- Parsed all 20,641 `others` annotations into 74 unique `(sequence_id, track_id)` tracks.
- Rechecked the 60-frame stratified queue covering 49 tracks and 48 sequences.
- Reviewed a representative annotated frame for each of the remaining 25 tracks.
- Rechecked ambiguous and boundary-clipped tracks using a representative frame from the middle of the track.

## Decision

- 73/74 tracks are motorized vehicles and may be mapped to `vehicle`.
- `MVI_40172 / track 79` is not a vehicle. It is a stationary roadside bus-stop or advertising structure with people and contains 201 boxes from frames 2066–2266.
- Exclude the complete rejected track through `ua_others_track_exclusions.csv`.
- The 60-frame queue resolves to 58 confirmed motor-vehicle samples and 2 rejected samples. Both rejected samples belong to the same rejected track.
- Keep `original_class=others` in companion metadata for every retained annotation.

Approval applies only when the track exclusion is enforced during dataset export. It does not make the complete dataset train-ready by itself.
