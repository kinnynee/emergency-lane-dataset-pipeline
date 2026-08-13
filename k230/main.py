"""CanMV K230 team-model bootstrap with a hash-bound, fail-closed contract.

This intentionally imports no PC-side Ultralytics, ByteTrack, OpenCV or NumPy
dependency.  Camera capture, preprocessing and matching YOLO11 decode are
enabled only after this bootstrap has accepted the exact compiled artifact.
"""

from team_model_config import DeploymentRejected, load_team_deployment


def load_team_model(contract):
    """Load only the K230 artifact whose SHA-256 was verified by the contract."""
    import nncase_runtime as nn

    kpu = nn.kpu()
    kpu.load_kmodel(contract["model"]["board_path"])
    print("MODEL_LOAD_OK")
    print("TEAM_MODEL_MANIFEST_SHA256=" + contract["team_model_manifest_sha256"])
    print("KMODEL_SHA256=" + contract["model"]["kmodel_sha256"])
    return kpu


def main():
    try:
        contract = load_team_deployment()
    except DeploymentRejected as exc:
        print("MODEL_LOAD_REJECTED", exc)
        raise
    load_team_model(contract)
    print("BOARD_RUNTIME_READY")
    print("INFERENCE_NOT_STARTED: run camera/preprocess/YOLO11 decode before emitting INFERENCE_OK")


if __name__ == "__main__":
    main()
