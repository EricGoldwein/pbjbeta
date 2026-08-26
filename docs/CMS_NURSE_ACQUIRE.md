# PBJ nurse staffing CMS acquire

**Depends on:** Data Ops foundation (PR #64 / branch `cursor/pbj-data-ops-v0-6578`).
**Dataset:** `7e0d53ba-8f02-4c66-98a5-14a1c997c50d`
**Discovery:** `GET https://data.cms.gov/data-api/v1/dataset/{uuid}/resources` (Primary CSV)

```bash
python scripts/manage_cms_sources.py nurse acquire --dry-run
python scripts/cms_pbj_nurse_acquire.py --dry-run
python scripts/cms_pbj_nurse_acquire.py   # download + validate + standardize_pbj_files.py
```

Does **not** promote metrics, write pbj-root, deploy, or auto-approve.
