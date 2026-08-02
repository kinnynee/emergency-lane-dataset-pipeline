# AAU RainSnow lighting — Data Lead signoff

Review date: 2026-08-02  
Reviewer: `CODEX_ACTING_DATA_LEAD`  
Status: `DATA_LEAD_SIGNOFF_COMPLETED`

The RGB `cam1.mkv` video of every AAU sequence was reviewed again at 10%, 50% and 90% of its duration. This covers 66 representative frames across all 22 sequences. Mean brightness was not used to assign lighting.

Approved distribution:

- `DAY=10`
- `NIGHT=11`
- `TWILIGHT=1`
- `BACKLIT=0`

The sequence-level decisions in `configs/aau_sequence_lighting_review.yaml` match the visual evidence. In particular, wet-road reflections and street lighting in night sequences are not reclassified as daylight. `Ringvej-3` remains `TWILIGHT`; `Hjorringvej-4` remains `NIGHT`.
