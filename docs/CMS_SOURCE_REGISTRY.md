# CMS source registry + PBJ Data Ops v0

**Canonical home:** PBJapp (this repo).  
**UI:** Streamlit page `pages/9_Data_Ops.py` (password via `PBJ_DATA_OPS_PASSWORD`).  
**Code:** `cms_source_registry.py` (static inventory), `cms_data_ops.py` (status + Provider Info actions).

Do not invent CMS dataset IDs. Only Provider Information has a verified stable ID on main (`4pq5-n9py`).

## Schema (`CmsSourceRecord`)

| Field | Meaning |
|-------|---------|
| `source_id` | Stable internal ID (`cms.*`) |
| `human_name` | Display name |
| `source_family` | Enum family |
| `formats` | CSV / ZIP / ZIP-member / PDF / other / mixed |
| `cadence` | monthly / quarterly / irregular / follows_provider_info / unknown |
| `release_version_strategy` | How vintages are identified |
| `cms_dataset_id` | Stable CMS ID when verified (else null) |
| `cms_metadata_endpoint` | Metastore URL when verified |
| `cms_landing_url` | Landing/explorer URL when verified |
| `local_raw_location` | Raw path pattern |
| `normalized_derived_location` | Derived path pattern |
| `validator` | Validator entrypoint if any |
| `downstream_consumers` | Known consumers |
| `automation_level` | detection_exists / acquisition_exists / normalization_exists / validation_exists / fully_manual / broken_legacy |
| `actions_enabled` | Data Ops UI actions (`check_cms`, `acquire_process`) |

Runtime status (`SourceOpsSnapshot`) adds: `cms_latest`, `pbjapp_latest`, `status`, `last_checked`, `last_successful_local_processing`.

### Status values

`CURRENT` · `CMS_NEWER` · `LOCAL_RAW_ONLY` · `PROCESSING_REQUIRED` · `READY_FOR_HANDOFF` · `UNKNOWN` · `ERROR`

## Auth

```bash
export PBJ_DATA_OPS_PASSWORD='…'   # never commit
streamlit run PBJ_Dashboard.py
# open Data Ops page in sidebar
```

Fail closed if env unset.

## Provider Info actions

Call `scripts/cms_provider_info_acquire.py` only (PR #63):

- Refresh/check CMS → `cms_data_ops.check_provider_info_cms`
- Acquire/process → `cms_data_ops.acquire_provider_info` → `acquire_and_process`

Other families are read-only in v0.

## Inventory audit (main)

See `CMS_SOURCE_REGISTRY` in `cms_source_registry.py`. Summary:

| Source | CMS ID | Format | Cadence | Automation |
|--------|--------|--------|---------|------------|
| Provider Information | `4pq5-n9py` | CSV (+ zip path) | monthly | validation_exists (acquire pilot) |
| PBJ nurse staffing | — | CSV | quarterly | normalization_exists |
| PBJ non-nurse | — | CSV/ZIP | quarterly | broken_legacy |
| PBJ EIN detail | — | ZIP/CSV | quarterly | broken_legacy |
| SNF All Owners | — | CSV | unknown | fully_manual |
| SNF Enrollments | — | — | unknown | fully_manual (no drop on main) |
| SNF CHOW | — | other/CSV | unknown | fully_manual |
| Chain Performance | — | CSV | irregular | detection_exists |
| SFF | — | CSV column (+ PDF allowed) | follows Provider Info | normalization_exists (via PI) |
