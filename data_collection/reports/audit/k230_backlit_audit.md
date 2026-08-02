# K230_BACKLIT audit

- Config definition: `{'slice_id': 'K230_BACKLIT', 'dimension': 'lighting', 'value': 'BACKLIT', 'minimum_sessions': 1, 'collection_rule': 'Dedicated sequence with sun or strong light source facing the elevated K230 camera.'}`
- Readiness row: `{'metric': 'mAP', 'slice': 'BACKLIT', 'source': 'K230_SELF_RECORDED', 'status': 'BLOCKED_MISSING_DATA', 'current_evaluable_sequences': '0', 'current_map': 'NOT_AVAILABLE', 'required_action': 'Record a dedicated elevated-camera BACKLIT sequence; do not substitute bright DAY images.', 'leakage_rule': 'Keep entire session in MAIN_K230_TEST and out of train/validation.'}`
- Real K230 media/annotation files found: **0**
- Ground truth: **NOT_AVAILABLE**
- Model predictions: **NOT_AVAILABLE**
- mAP: **NOT_AVAILABLE**

Conclusion: the slice is a collection plan only; zero must not be reported as a score.
