# Host → K230 runtime contract

`runtime_contract.json` is the single source of truth for input size, class ID,
class name and confidence on both runtimes. Copy it to
`/sdcard/emergency_lane/runtime_contract.json` and copy the `.kmodel` compiled
from the matching team checkpoint to the path declared by `kmodel_path`.

`main.py` is a fail-closed CanMV bootstrap. It uses `nncase_runtime` and refuses
missing, COCO-named, or policy-mismatched models. It intentionally does not
print `INFERENCE_OK`: camera preprocessing, YOLO11 one-class output decoding,
tracking and the calibrated stopped-vehicle state machine still need a real
board implementation and test. Until then, the release validator must remain
`BLOCKED_BOARD_RUN_REQUIRED`.

The desktop `host_yolo_loop.py` is a replay/reference runtime. Its Ultralytics,
ByteTrack, OpenCV and NumPy dependencies are not copied to CanMV; only the
contract and verified algorithm parameters cross that boundary.
