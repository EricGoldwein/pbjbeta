# PBJapp data source map

**Purpose:** Where data lives, what is input vs output, and what must stay out of git.  
**Resolver:** `data_path_resolver.py` and `cms_data_paths.py` (env → `data_paths.local.json` → repo-relative default).

---

## PBJ nurse staffing data

| Role | Path | Git on main | Notes |
|------|------|-------------|-------|
| Raw CMS PBJ nurse CSVs | `PBJcsv/` | Gitignored | ~10 GB; local or external junction |
| Standardized nurse files | `standardized_PBJ/` | Gitignored | Output of `standardize_pbj_files.py` |
| National lite rollups | `pbj_lite/national_lite_metrics.csv`, `state_lite_metrics.csv` | Tracked | Small deploy slices |
| Facility lite rollup | `pbj_lite/facility_lite_metrics.csv` | Tracked | ~46 MB; intentional deploy input |
| National facility-quarter metrics | `facility_quarterly_metrics.csv` (repo root) | Gitignored | Input to `geo_distribution_lib.py` |
| Per-facility complete slice | `deployments/pbj320-<CCN>/facility_<CCN>_complete_data.csv` | Mixed | Generated from national pipeline |
| Legacy slice | `pbj320-495177/facility_495177_complete_data.csv` | Tracked | Older bundle layout |

**Canonical for AI:** Pipeline scripts at repo root + `pbj_lite/*.csv` for bundled metrics. Raw `PBJcsv/` is read-only input on dev machines.

---

## PBJ non-nurse staffing data

| Role | Path | Git on main | Notes |
|------|------|-------------|-------|
| Raw CMS non-nurse CSVs | `NonNursecsv/` | Gitignored | ~19 GB |
| Standardized non-nurse | `standardized_NonNurse/` | Gitignored | Output of `standardize_nonnursepbj_files.py` |
| National non-nurse metrics | `non_nurse_*_metrics.csv` (repo root) | Gitignored | From `generate_non_nurse_*.py` |
| Per-facility non-nurse daily | `deployments/pbj320-<CCN>/facility_<CCN>_nonnurse_daily.csv` | Local only | Example: 315461 untracked |

**MDS census:** There is no separate MDS file tree. `MDScensus` is a column in PBJ nurse and non-nurse daily data (resident days). Referenced in `DATA_PIPELINE_DOCUMENTATION.md`, templates, and metric generators — not a standalone dataset folder.

---

## Provider / facility info

| Role | Path | Git on main | Notes |
|------|------|-------------|-------|
| Raw CMS Provider Info snapshots | `provider_info/NH_ProviderInfo_*.csv` | **Tracked** (8 files) | Folder listed in `.gitignore` but files pre-indexed |
| Combined provider CSVs | `provider_info_combined.csv`, `provider_info_combined_with_quarters.csv` | Gitignored | ~700–800 MB |
| Normalized provider | `provider_info_normalized/` | Gitignored | Rebuildable |
| Extracted provider | `provider_info_extracted/` | Gitignored | Interim pipeline output |
| Transform logic | `prov_info.py` | Tracked | Canonical code |
| Per-facility provider slice | `deployments/pbj320-<CCN>/facility_<CCN>_provider_info_data.csv` | In bundles | Generated at publish time |

**Canonical for AI:** `prov_info.py` + `config/` + tracked `provider_info/` snapshots. Do not commit new extract/normalized trees.

---

## Ownership / chain data

| Role | Path | Git on main | Notes |
|------|------|-------------|-------|
| CMS chain performance measures | `ownership/Nursing_Home_Chain_Performance_Measures_*.csv` | Tracked | National reference |
| Per-facility ownership longitudinal | `deployments/pbj320-335513/ownership/facility_335513_entity_longitudinal.csv` | Tracked | Deploy slice |
| Entity lookup | `deployments/pbj320-335513/ownership/entity_lookup.csv` | Tracked | Deploy slice |
| Ownership scripts | `scripts/Ownership.py` | Tracked | Processing helper |
| UI | `static/js/pbj_ownership_disclosures.js`, `templates/partials/v2/chow_ownership_modal.html` | Tracked | Display layer |

---

## Citation config / runtime

| Role | Path | Git on main | Notes |
|------|------|-------------|-------|
| Taxonomy config | `config/citation_topic_registry.json`, `citation_severity_rank.json`, `citation_narrative_role_map.json` | Tracked | Edit with care |
| PBJ bridge config | `config/citation_pbj_bridge.json` | Tracked | Links citations to PBJ metrics |
| Python runtime | `citation_taxonomy.py`, `citation_lib.py`, `citation_pbj_bridge.py`, `citation_date_confidence.py` | Tracked | Canonical |
| Per-facility citations CSV | `deployments/pbj320-<CCN>/facility_<CCN>_citations.csv` | Local only | Example: 315461 |
| Streamlit tools | `scripts/citations_app.py`, `scripts/citations_dashboard.py` | Tracked | Dev/QA apps |

---

## Geo distribution metrics

| Role | Path | Git on main | Notes |
|------|------|-------------|-------|
| Server library | `geo_distribution_lib.py` | Tracked | Reads `facility_quarterly_metrics.csv` |
| Client JS | `static/js/pbj_v2_geo_distribution.js` | Tracked | V2 geo card |
| Region mapping | `cms_region_state_mapping.csv` | Untracked on slim worktree | Small reference; copied into deploys |
| County centroids | `data/gis/*_county_centroids.json` | Not on slim worktree | Small registry when present |
| MACPAC state standards | `macpac/macpac_state_standards_clean.csv`, `pbj_lite/macpac_state_standards_clean.csv` | Tracked | Duplicated for deploy convenience |

---

## EIN (employee identification) data

| Role | Path | Git on main | Notes |
|------|------|-------------|-------|
| Raw EIN tree | `EIN/monolithic/`, `EIN/quarters/` | Gitignored | National CMS PUF |
| Per-facility EIN analytics | `deployments/pbj320-335513/facility_335513_ein_*.csv/.parquet` | Tracked in 335513 bundle | Large; candidate to untrack later |
| Libraries | `facility_ein_lib.py`, `facility_ein_employee_analytics.py` | Tracked | Canonical code |

---

## Report Builder (V3) data dependencies

| Role | Path | Notes |
|------|------|-------|
| Runtime | `report_builder_v3.py`, `report_item_registry.py` | Canonical |
| Templates | `templates/partials/v2/report_builder_v3_*` | Canonical |
| Client | `static/js/pbj_report_builder_v3.js`, `pbj_report_item_registry.js` | Canonical |
| Generated HTML reports | `deployments/pbj320-335513/reports/` | Deploy output — do not edit as source |

---

## Local-only data (never commit)

- `data_paths.local.json` — machine-specific path overrides
- `deployments/pbj320-315461/` — local deploy slice CSVs
- `pr2a_pre_rebase_diff_report.txt` — extraction scratch
- `backups/`, `Archive/`, `metrics_backups/`, `.ein_quarter_cache/`
- `outputs/`, `Facility Reports/`, `facility_red_flag_report/reports/`
- `column_mapping_reports/` — regenerable; still in git index historically
- Playground JSON/JS at repo root — regenerable via `generate_playground_*.py`

---

## Generated data that should not be committed (new files)

Even if similar paths are already tracked from before `.gitignore` guardrails:

- New files under `provider_info/`, `provider_info_extracted/`, `provider_info_normalized/`
- New `deployments/pbj320-*` bundles
- National rollup CSVs at repo root (`facility_quarterly_metrics.csv`, etc.)
- Parquet, zip, xlsx, sqlite artifacts
- `data/sec/cache/` and other cache dirs

**Exception (intentional tracked slices):** `pbj_lite/*.csv`, `ownership/*.csv`, existing `provider_info/NH_ProviderInfo_*.csv`, and the `deployments/pbj320-335513/` reference bundle until a dedicated untrack PR.
