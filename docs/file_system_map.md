# PBJapp file system map

**Purpose:** Help Cursor and humans navigate the repo without moving files.  
**Scope:** `origin/main` at `b6703cc` (367 tracked files). This worktree is a slim clone — large CMS trees may exist only on full dev machines via junction or `data_paths.local.json`.

See also: `docs/data_source_map.md`, `docs/repo_cleanup_plan.md`, `data_path_resolver.py`.

---

## Canonical source areas (edit here; re-bootstrap deploys)

| Area | Representative paths | Git | Cursor index |
|------|---------------------|-----|--------------|
| Live app entrypoints | `PBJ_Dashboard.py`, `main_app.py`, `dynamic_facility_dashboard.py`, `pages/` | Tracked | Yes |
| V2/V3 dashboard shell | `templates/superdynamic_dashboard_v2.html` | Tracked | Yes |
| V2 partials | `templates/partials/v2/*` (~45 files) | Tracked | Yes |
| Legacy V1 template | `templates/dynamic_facility_dashboard.html` | Tracked | Yes |
| Client JS | `static/js/*` | Tracked | Yes |
| Shared Python libs | `facility_report_lib.py`, `facility_ein_lib.py`, `geo_distribution_lib.py`, `report_builder_v3.py`, `report_item_registry.py`, `data_path_resolver.py`, `cms_data_paths.py` | Tracked | Yes |
| Identifiers / utils | `pbj_identifiers/`, `utils/` | Tracked | Yes |
| Citation runtime | `citation_*.py`, `config/citation_*.json` | Tracked | Yes |
| Pipeline scripts | `standardize_pbj_files.py`, `generate_metrics.py`, `run_pipeline_update.py`, `create_vercel_deployment.py`, `prov_info.py` | Tracked | Yes |
| Deploy guardrails | `deployment_entrypoint_guard.py`, `packaging_refresh_gates.py`, `scripts/bootstrap_superdynamic_v2_facility.py`, `scripts/preflight_v2_facility_deploy.py`, `scripts/check_v2_*.py` | Tracked | Yes |
| Tests | `tests/` | Tracked | Yes |
| Small reference data (intentional) | `pbj_lite/*.csv`, `ownership/*.csv`, `macpac/macpac_state_standards_clean.csv`, `state_legislation_standards.csv` | Tracked | Yes |
| Docs / runbooks | `docs/`, `DATA_PIPELINE_DOCUMENTATION.md`, `DEPLOYMENT_RULES.md` | Tracked | Yes |

**Future AI tasks:** Treat repo-root copies above as canonical. Regenerate deployment bundles from these — do not hand-edit bundle copies.

---

## Generated / local artifact areas (do not treat as source)

| Area | Representative paths | Git | Cursor index |
|------|---------------------|-----|--------------|
| Per-CCN Vercel bundles | `deployments/pbj320-<CCN>/` | Mixed (335513 tracked; others local) | No — `.cursorignore` |
| Legacy root bundle | `pbj320-495177/` | Tracked (9 files) | Low priority |
| National CMS raw inputs | `PBJcsv/`, `NonNursecsv/`, `EIN/` | Gitignored; absent on slim worktree | No |
| Processed national data | `standardized_PBJ/`, `standardized_NonNurse/`, `provider_info_normalized/`, `provider_info_extracted/` | Gitignored | No |
| National metric rollups | `facility_quarterly_metrics.csv`, `state_quarterly_metrics.csv`, etc. (repo root) | Gitignored | No |
| Pipeline mapping reports | `column_mapping_reports/` (64 txt files) | **Tracked** (pre-ignore index) | No |
| Playground artifacts | `playground_data.json`, `playground_distributions.json`, `real_mobile_data.js` | **Tracked** (should be local) | No |
| QA / verify output | `.verify_output/`, `playwright-report/` | Gitignored | No |
| Caches / interim | `.ein_quarter_cache/`, `indexed/`, `metrics_backups/`, `data/sec/cache/` | Gitignored | No |
| Outputs / reports | `outputs/`, `Facility Reports/`, `facility_red_flag_report/reports/` | Gitignored | No |
| Local path overrides | `data_paths.local.json` | Gitignored | No |

**Commit policy:** Do not stage new generated CSVs, deployment folders, provider extracts, or scratch reports.

---

## Deployment bundle areas

| Bundle | Path | Status |
|--------|------|--------|
| Reference V2 deploy (Seagate) | `deployments/pbj320-335513/` | 30 tracked files — Flask app, EIN parquets, complete_data CSV, copied libs, legacy templates |
| Local-only deploy slice | `deployments/pbj320-315461/` | Untracked (4 CSVs on pr2a worktree) |
| Legacy outside `deployments/` | `pbj320-495177/` | Tracked — older Vercel layout |

**Rule:** Edit canonical repo files (`templates/`, `static/js/`, root `*_lib.py`), then run `create_vercel_deployment.py` / `scripts/bootstrap_superdynamic_v2_facility.py`. Never casually edit copied files inside bundles.

---

## Backup / stale / duplicate areas

| Area | Notes | Action |
|------|-------|--------|
| `deployments/pbj320-335513/facility_335513_complete_data.csv.bak_*` | Session backup inside bundle | Leave; do not edit |
| `macpac_state_standards_clean.csv` | Triplicated: `macpac/`, `pbj_lite/`, deploy copies | Canonical: `macpac/`; others are publish copies |
| `pbj320-495177/` vs `deployments/` | Parallel deploy layouts | Prefer `deployments/` for new CCNs |
| `templates/dynamic_facility_dashboard.html` vs deploy copies | V1 shell duplicated in bundles | Canonical: repo `templates/` |
| `tatus --porcelain`, `debug_contract.py` | Accidental root files still tracked | Flag for future untrack PR |
| `.cursorrules.backup` | Local backup of rules | Low risk; not runtime |
| `backups/`, `Archive/`, `archives/` | Gitignored session snapshots | Local only when present |
| `pr2a_pre_rebase_diff_report.txt` | Extraction diff report | Local scratch — do not commit |

---

## What Cursor should ignore

See `.cursorignore`. Broadly:

- `deployments/` (generated bundles)
- `backups/`, `outputs/`, provider extract/normalized trees
- Audit archives, deploy audits, cursor state tools, mockups
- `data/sec/cache/` when present

**Do not ignore:** `templates/`, `static/js/`, `scripts/`, `config/`, `pbj_lite/`, `ownership/`, root Python libs, `docs/`.

---

## What not to edit casually

| File / area | Why |
|-------------|-----|
| `templates/superdynamic_dashboard_v2.html` | Primary V2/V3 shell (~57K lines); bootstrap copies to deploys |
| `deployments/**` copied `*_lib.py`, `static/js/*`, templates | Regenerate from repo canonical |
| `pbj_lite/facility_lite_metrics.csv` | Intentional deploy slice (~46 MB tracked) |
| `provider_info/NH_ProviderInfo_*.csv` | CMS snapshots (tracked; folder is gitignored for new files) |
| `render.yaml`, `vercel.json`, `deployments/*/vercel.json` | Production entry / routing |
| `config/citation_*.json` | Citation taxonomy and bridge config |
| `.gitignore`, `data_path_resolver.py` | Path contracts for pipeline + deploy |

---

## Satellite apps (tracked, lower traffic)

- `Admin_Dashboard/` — internal admin Streamlit
- `RN_Compliance_Dashboard/` — RN compliance analysis
- `225500/` — legacy single-facility app
- `streamlit-extras/` — vendored Streamlit components
- `api/` — minimal API stub

---

## Expected benefits

**Cursor:** Smaller index, less confusion between canonical source and deploy copies, clearer guardrails for AI edits.  
**Humans:** One-page map of “edit here vs regenerate there” without a repo reorg.
