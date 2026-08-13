# K230 deployment contract

`team_model_config.py` replaces the old “load COCO when configuration fails”
pattern. The board accepts only a K230 model whose SHA-256 is in the contract.

After final fine-tuning and K230 compilation, generate the contract:

```powershell
python data_collection/scripts/create_k230_deployment_contract.py `
  --team-model-manifest runs/final/team_model_manifest.json `
  --kmodel build/team_yolo11n_320.kmodel `
  --board-kmodel-path /sdcard/model/team_yolo11n_320.kmodel `
  --output build/team_deployment_contract.json
```

Copy the `.kmodel`, contract and `team_model_config.py` to the K230. At the
top of the board application's `main.py`, before KPU initialization, use:

```python
from team_model_config import DeploymentRejected, load_team_deployment

try:
    TEAM_DEPLOYMENT = load_team_deployment()
except DeploymentRejected as exc:
    print("MODEL_LOAD_REJECTED", exc)
    raise

MODEL_PATH = TEAM_DEPLOYMENT["model"]["board_path"]
CONFIDENCE = TEAM_DEPLOYMENT["detection"]["confidence"]
CLASS_NAMES = ["vehicle"]
```

Do not catch this error to load a COCO model. On successful boot, emit these
lines exactly, then run an actual inference before release validation:

```text
MODEL_LOAD_OK
TEAM_MODEL_MANIFEST_SHA256=<contract team_model_manifest_sha256>
KMODEL_SHA256=<contract model kmodel_sha256>
INFERENCE_OK
```

The board sends detections; `host_yolo_loop.py` owns the stopped-vehicle
hysteresis. This avoids attempting to run PC-only Ultralytics, ByteTrack,
OpenCV or NumPy on CanMV.
