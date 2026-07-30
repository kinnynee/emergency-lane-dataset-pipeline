# Emergency Lane Dataset Pipeline

Data engineering pipeline for the project **“Automatic emergency-lane stopped-vehicle warning using K230 and YOLOv8.”**

The repository contains:

- dataset collection plans and source registries;
- license and permission validation;
- safe download, video inspection and normalization scripts;
- frame extraction and duplicate detection;
- YOLO dataset preparation rules;
- metadata, quality-control reports and evidence.

Images, videos, downloaded archives, model weights, credentials and other large or private files are intentionally excluded from Git.

## Getting started

```powershell
cd data_collection
python -m pip install -r requirements-data.txt
python -m pip install -r requirements-online-data.txt
python -m pytest
python scripts/run_online_data_pipeline.py --dry-run
```

See [`data_collection/README.md`](data_collection/README.md) for the directory layout and complete workflow.

