# Reproduction commands

Run from repository root. Production reproductions use new audit-only output directories and do not overwrite `reports/external_eda`.

```powershell
git switch audit/eda-full-review
python data_collection/scripts/audit_eda_independent.py --output data_collection/reports/audit_reproduction --audit-dir data_collection/reports/audit

python data_collection/scripts/run_external_dataset_eda.py --sample-size 100 --output data_collection/reports/audit_reproduction/eda_sample_100 --skip-contact-sheets
python data_collection/scripts/run_external_dataset_eda.py --sample-size 100 --output data_collection/reports/audit_reproduction/eda_sample_100 --skip-contact-sheets --resume

python data_collection/scripts/run_external_dataset_eda.py --sample-size 5000 --output data_collection/reports/audit_reproduction/eda_sample_5000 --skip-contact-sheets
python data_collection/scripts/run_external_dataset_eda.py --sample-size 5000 --output data_collection/reports/audit_reproduction/eda_sample_5000 --skip-contact-sheets --resume

python -m compileall data_collection/scripts
python -m pytest -q
python -m pytest --collect-only
python data_collection/scripts/run_external_dataset_eda.py --help
```

The independent full raw-archive test can also be invoked explicitly:

```powershell
$env:K230_RUN_RAW_AUDIT_TESTS='1'
python -m pytest data_collection/tests/test_independent_audit.py -q
Remove-Item Env:K230_RUN_RAW_AUDIT_TESTS
```

The opt-in test is intentionally not part of the fast default suite because it parses the full raw archives and takes several minutes.
