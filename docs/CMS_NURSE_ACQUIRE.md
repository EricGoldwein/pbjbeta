# PBJ nurse staffing CMS acquire

**Depends on:** Data Ops foundation (PR #64 / branch `cursor/pbj-data-ops-v0-6578`).
**Dataset:** `7e0d53ba-8f02-4c66-98a5-14a1c997c50d`
**Discovery:** `GET https://data.cms.gov/data-api/v1/dataset/{uuid}/resources` (Primary CSV)

```bash
python scripts/manage_cms_sources.py nurse acquire --dry-run
python scripts/cms_pbj_nurse_acquire.py --dry-run
python scripts/cms_pbj_nurse_acquire.py   # stream download + validate + standardize_pbj_files.py
```

## Identity / no-op rule

`CURRENT` / cryptographically identical requires `PBJcsv/_manifests/{CYyyyyQn}/acquisition.json`
whose `sha256` matches the on-disk raw file, plus matching CMS filename/quarter (and
CMS-reported `file_size` / `file_uuid` when present). Same filename alone is never enough.

Unmanifested historical raw is structurally validated when present but reported as
`LOCAL_UNMANIFESTED` — not `CURRENT`. Manifest SHA mismatch → `PROVENANCE_MISMATCH` (fail closed).

Downloads stream CMS → temp in 1 MiB chunks while hashing; partial failures delete temps only.

Does **not** promote metrics, write pbj-root, deploy, or auto-approve.
