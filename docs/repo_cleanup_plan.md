# PBJapp repo cleanup plan

**Phase 1:** 2026-06-13 — Audit + `.gitignore` guardrails (no moves, no behavior changes)  
**Phase 2:** 2026-06-13 — Role classification + tailored target structure (plan only)  
**Related:** `docs/DATA_PATHS_AUDIT.md`, `docs/SAFE_CLEANUP_EXECUTION_PLAN.md`, `docs/testing_backlog.md`

---

## Repo layers (context)

PBJapp is **not** a web-app-only repo. Active or semi-active layers:

| Layer | Description | Default git policy |
|-------|-------------|-------------------|
| Live app / dashboards | Streamlit national + Flask facility V2/V3 | Track source; not generated bundles |
| PBJ pipelines | Extract → standardize → metrics → deploy slices | Track scripts; not raw CMS |
| Large CMS/source data | PBJ, EIN, NonNurse, provider info | **Gitignore**; local or external via junction |
| Generated deploy artifacts | Per-CCN Python/HTML, lite slices, facility CSVs | Mixed — some tracked today |
| Test / QA | pytest, browser QA scripts, audits | Track or document in `docs/testing_backlog.md` |

**Principles for all phases:** large ≠ bad; old ≠ dead; uncertain → flag, don't guess.

---

## Phase 2 — Role classification

Legend for **safe to move now**:

| Value | Meaning |
|-------|---------|
| **Yes** | Documentation-only or git-index-only; no runtime path dependency |
| **No** | Active entrypoint, hardcoded path, or deploy contract |
| **Later** | Safe in principle after resolver / bootstrap / deploy manifest update |
| **Never** | Production entry or intentionally tracked deploy input |

### Live app code

| Current path | Role | Proposed path | Move now? | Breaks imports/routes? | Notes |
|--------------|------|---------------|-----------|------------------------|-------|
| `PBJ_Dashboard.py` | `live_app` | `app/routes/pbj_dashboard_streamlit.py` | No | **Yes** — `render.yaml` start command | Production Render entry |
| `main_app.py` | `live_app` | `app/routes/main_app.py` | No | **Yes** | Streamlit wrapper / SEO shell |
| `dynamic_facility_dashboard.py` | `live_app` (+ mixed concerns — see §B.3) | `app/routes/dynamic_facility_dashboard.py` | No | **Yes** | Canonical **legacy** Flask app; renders `dynamic_facility_dashboard.html` (not V2 shell) |
| `staffing_query_app.py`, `staging_app.py`, `staging_dashboard.py` | `live_app` | `app/routes/staging/` | Later | **Yes** | Staging / experiments; low prod traffic |
| `pages/` | `live_app` | `app/routes/streamlit_pages/` | Later | **Yes** | Streamlit multipage |
| `facility_report_lib.py` | `live_app` | `app/services/facility_report_lib.py` | No | **Yes** — imported by Flask apps + deployments | Report/memo generation core |
| `facility_ein_lib.py`, `facility_ein_employee_analytics.py` | `live_app` | `app/services/ein/` | No | **Yes** | EIN bridge + roster analytics |
| `geo_distribution_lib.py`, `report_findings_engine.py`, `report_builder_v3.py` | `live_app` | `app/services/` | No | **Yes** | V2/V3 dashboard services |
| `pbj_federal_compliance.py`, `nonnurse_staffing_lib.py` | `live_app` | `app/services/` | No | **Yes** | Compliance + non-nurse API backing |
| `deployment_entrypoint_guard.py` | `live_app` | `app/services/deploy/entrypoint_guard.py` | Later | **Maybe** — imported by packaging | Protects V2 superdynamic entrypoints |
| `facility_deploy_guardrails.py` | `live_app` | `app/services/deploy/guardrails.py` | Later | **Maybe** | Vercel bundle validation |
| `data_path_resolver.py`, `cms_data_paths.py`, `file_path_utils.py` | `live_app` | `app/config/data_paths.py` | Later | **Yes** — 100+ refs | Path contract; migrate as one unit |
| `utils/` | `live_app` | `app/utils/` | Later | **Yes** | Small shared helpers |
| `api/` | `live_app` | `app/routes/api/` | Later | **Uncertain** | Minimal; verify usage |
| `static/` | `live_app` | `app/static/` | No | **Yes** — script tags in templates | Canonical client assets |
| `pbj_identifiers/` | `live_app` | `app/packages/pbj_identifiers/` | Later | **Yes** | URL validators used by Flask |
| `config/` | `live_app` | `config/` (keep) | No | **Maybe** | Deploy timestamps, facility config |
| `streamlit-extras/` | `live_app` | vendor boundary (keep) | No | **Yes** | Separate component repo |

### Dashboard templates (V2/V3 active — not disposable)

| Current path | Role | Proposed path | Move now? | Breaks imports/routes? | Notes |
|--------------|------|---------------|-----------|------------------------|-------|
| `templates/superdynamic_dashboard_v2.html` | `dashboard_template` | `app/templates/dashboard_v2/shell.html` | No | **Yes** | **Primary live V2/V3 shell** (~57K lines). See §B.1 |
| `templates/dynamic_facility_dashboard.html` | `dashboard_template` | `app/templates/dashboard_v1/shell.html` | No | **Yes** | Legacy Flask template (~29K lines) |
| `templates/partials/v2/*` (~45 files) | `dashboard_template` | `app/templates/dashboard_v2/partials/` | No | **Yes** | Active extract pattern; already partial |
| `templates/partials/superdynamic_url_macros.html` | `dashboard_template` | `app/templates/shared/url_macros.html` | Later | **Yes** | Shared URL helpers |
| `templates/report_builder_v3_*.html` | `dashboard_template` | `app/templates/dashboard_v3/report_builder/` | Later | **Yes** | V3 report builder panes |
| `deployments/pbj320-*/templates/superdynamic_dashboard_v2.html` | `generated_artifact` | `deployments/generated/<CCN>/templates/...` | No | **Yes** — Vercel bundle | Copy of repo canonical; bootstrap refreshes from repo |
| `deployments/pbj320-*/templates/partials/v2/*` | `generated_artifact` | same | No | **Yes** | Synced by `create_vercel_deployment.py` / bootstrap |

### Pipeline — extract

| Current path | Role | Proposed path | Move now? | Breaks imports/routes? | Notes |
|--------------|------|---------------|-----------|------------------------|-------|
| `standardize_pbj_files.py` | `pipeline_extract` | `pipelines/extract/standardize_pbj.py` | Later | **Yes** — CLI + `run_pipeline_update` | PBJcsv → standardized_PBJ |
| `standardize_nonnursepbj_files.py` | `pipeline_extract` | `pipelines/extract/standardize_nonnurse.py` | Later | **Yes** | NonNursecsv → standardized_NonNurse |
| `scripts/ingest_cms_quarter.py` | `pipeline_extract` | `pipelines/extract/ingest_cms_quarter.py` | Later | **Maybe** | EIN quarter ingest |
| `scripts/ingest_cms_nonnurse_quarter.py` | `pipeline_extract` | `pipelines/extract/ingest_nonnurse_quarter.py` | Later | **Maybe** | |
| `scripts/ingest_ein_quarter_csv.py` | `pipeline_extract` | `pipelines/ein/ingest_quarter_csv.py` | Later | **Maybe** | |
| `scripts/manage_cms_sources.py` | `pipeline_extract` | `pipelines/extract/manage_cms_sources.py` | Later | **Maybe** | Download / status CLI |
| `scripts/organize_ein_folder.py`, `scripts/sync_ein_quarters.py` | `pipeline_extract` | `pipelines/ein/` | Later | **Maybe** | EIN tree hygiene |
| `column_mapper.py`, `non_nurse_column_mapper.py` | `pipeline_extract` | `pipelines/extract/` | Later | **Maybe** | Schema mapping helpers |

### Pipeline — transform

| Current path | Role | Proposed path | Move now? | Breaks imports/routes? | Notes |
|--------------|------|---------------|-----------|------------------------|-------|
| `run_pipeline_update.py` | `pipeline_transform` | `pipelines/run_pipeline_update.py` | Later | **Yes** — documented CLI entry | Orchestrator |
| `normalize_provider_info.py` | `pipeline_transform` | `pipelines/provider_info/normalize.py` | Later | **Yes** | provider_info → provider_info_normalized |
| `generate_metrics.py`, `generate_national_metrics.py` | `pipeline_transform` | `pipelines/transform/` | Later | **Yes** | Lite metric rollups |
| `generate_non_nurse_*.py` | `pipeline_transform` | `pipelines/transform/` | Later | **Yes** | |
| `generate_state_rankings.py`, `generate_quarterly_medians.py` | `pipeline_transform` | `pipelines/transform/` | Later | **Maybe** | |
| `prov_info.py`, `prov_info_quarter_map.py` | `pipeline_transform` | `pipelines/provider_info/` | Later | **Yes** | Combined provider-info logic |
| `pbj_staffing_normalize.py` | `pipeline_transform` | `pipelines/transform/` | Later | **Maybe** | |

### Pipeline — publish (deploy slices + reports)

| Current path | Role | Proposed path | Move now? | Breaks imports/routes? | Notes |
|--------------|------|---------------|-----------|------------------------|-------|
| `create_vercel_deployment.py` | `pipeline_publish` | `pipelines/publish/create_vercel_deployment.py` | Later | **Yes** | Copies libs/templates/JS into `deployments/` |
| `scripts/bootstrap_superdynamic_v2_facility.py` | `pipeline_publish` | `pipelines/publish/bootstrap_superdynamic_v2.py` | Later | **Maybe** | Cold-start V2 bundle from ref CCN |
| `scripts/deploy_vercel_facility.py` | `pipeline_publish` | `pipelines/publish/deploy_vercel_facility.py` | Later | **Maybe** | |
| `scripts/ensure_facility_county_lite_slice.py` | `pipeline_publish` | `pipelines/publish/ensure_lite_slice.py` | Later | **Maybe** | Writes deploy `pbj_lite/` slice |
| `scripts/extract_facility_ein_from_zip.py` | `pipeline_publish` | `pipelines/publish/extract_facility_ein.py` | Later | **Maybe** | Facility EIN slice into deploy dir |
| `generate_facility_report_attorney.py`, `internal_report_generator.py` | `pipeline_publish` | `pipelines/publish/reports/` | Later | **Maybe** | HTML report export |
| `facility_red_flag_report/` (generator scripts) | `pipeline_publish` | `pipelines/publish/red_flag_reports/` | Later | **Maybe** | **Reports output** gitignored |

### Generated deploy artifacts

| Current path | Role | Proposed path | Move now? | Breaks imports/routes? | Notes |
|--------------|------|---------------|-----------|------------------------|-------|
| `deployments/pbj320-<CCN>/facility_<CCN>_superdynamic_dashboard.py` | `generated_artifact` | `deployments/generated/pbj320-<CCN>/` | **Never** manual move | **Yes** — `vercel.json` entry | Treat as **deploy output**; edit repo canonical, re-bootstrap. See §B.4 |
| `deployments/pbj320-*/facility_*_flask_app.py` | `generated_artifact` | same | No | **Yes** | Legacy entry; some CCNs |
| `deployments/pbj320-*/facility_*_complete_data.csv` | `generated_artifact` | `data/deploy_slices/<CCN>/` | Later | **Yes** — Flask reads local path | Per-facility PBJ extract |
| `deployments/pbj320-*/pbj_lite/facility_lite_metrics.csv` | `generated_artifact` | `data/deploy_slices/<CCN>/pbj_lite/` | Later | **Yes** | State-scoped slice; regenerated by publish pipeline |
| `deployments/pbj320-335513/facility_335513_ein_employee_detail.csv` | `generated_artifact` | `data/deploy_slices/335513/` | Later | **Maybe** | 13 MB; candidate to untrack from git |
| `deployments/pbj320-*/static/js/*` | `generated_artifact` | stay in bundle | No | **Yes** | Mirrors `static/js/` |
| `deployments/pbj320-*/facility_report_lib.py` (copies) | `generated_artifact` | generated from canonical | No | **Yes** | Do not hand-edit; sync from root |
| `pbj320-495177/` | `generated_artifact` | `deployments/generated/pbj320-495177/` | Later | **Uncertain** | Legacy deploy outside `deployments/` |

### Raw / processed data

| Current path | Role | Proposed path | Move now? | Gitignored? | Notes |
|--------------|------|---------------|-----------|-------------|-------|
| `PBJcsv/` | `raw_data` | `data/raw/pbj_nurse/` | Later (junction) | **Yes** | ~10 GB; pipeline read |
| `NonNursecsv/` | `raw_data` | `data/raw/pbj_nonnurse/` | Later (junction) | **Yes** | ~19 GB |
| `provider_info/` | `raw_data` | `data/raw/provider_info/` | Later (junction) | **Yes** | ~3 GB; 9 small CSVs still **tracked** |
| `EIN/` | `raw_data` | `data/raw/ein/` | Later (junction) | **Yes** | ~9 GB; quarters + monolithic |
| `provider_info_combined.csv` (root) | `processed_data` | `data/processed/provider_info/combined.csv` | Later (junction) | **Yes** | ~834 MB untracked on disk |
| `provider_info_combined_with_quarters.csv` | `processed_data` | `data/processed/provider_info/` | Later | **Yes** | ~695 MB |
| `standardized_PBJ/` | `processed_data` | `data/processed/pbj_nurse/` | Later (junction) | **Yes** | ~9 GB |
| `standardized_NonNurse/` | `processed_data` | `data/processed/pbj_nonnurse/` | Later (junction) | **Yes** | ~17 GB |
| `provider_info_normalized/` | `processed_data` | `data/processed/provider_info/normalized/` | Later | **Yes** | Rebuildable |
| `provider_info_extracted/` | `interim` → `data/interim/provider_info_extracted/` | Later | **Listed in .gitignore but ~98 CSVs still in git index** | See §B.6 |
| `indexed/`, `.ein_quarter_cache/`, `metrics_backups/` | `interim` | `data/interim/` | Later | **Yes** | Cache / index layers |
| `pbj_lite/facility_lite_metrics.csv` | `deploy_slices` | `data/deploy_slices/pbj_lite/facility_lite_metrics.csv` | **No** | **Tracked (46 MB)** | **Intentional** deploy input — see §B.5 |
| `pbj_lite/national_lite_metrics.csv`, `state_lite_metrics.csv` | `deploy_slices` | `data/deploy_slices/pbj_lite/` | Later | **Tracked (small)** | Bundled into Vercel deploys |
| `data/gis/*_county_centroids.json` | `samples` | `data/samples/gis/` | Later | Untracked/new | Small map reference |
| `data/players/`, `data/sec/`, `data/threshold_registry.json` | `samples` | `data/samples/registries/` | Later | Partially tracked | App registries |
| `macpac/macpac_state_standards_clean.csv` | `samples` | `data/samples/macpac/` | Later | **Tracked** | Small reference |
| `state_legislation_standards.csv` | `samples` | `data/samples/` | Later | **Tracked** | Compliance reference |
| `cms_region_state_mapping.csv` (root) | `samples` | `data/samples/pbj_lite/` | Later | Untracked | Used by geo card + deploy copy |

### Test / QA (not junk)

| Current path | Role | Proposed path | Move now? | Notes |
|--------------|------|---------------|-----------|-------|
| `tests/` | `test_or_qa` | `tests/` (keep) | No | pytest + JS unit tests; growing |
| `tests/fixtures/` | `test_or_qa` | `tests/fixtures/` | No | Small committed fixtures |
| `scripts/qa_*.py`, `scripts/smoke_test_*.py` | `test_or_qa` | `tests/qa/` or keep `scripts/` | Later | Browser/HTTP QA — see `docs/testing_backlog.md` |
| `scripts/audit_*.py`, `scripts/check_v2_*.py` | `test_or_qa` | `tests/qa/audits/` | Later | Guardrail audits; not CI yet |
| `scripts/verify_*.py`, `scripts/preflight_*.py` | `test_or_qa` | `tests/qa/` | Later | Pre-deploy verification |
| `pbj_facility_coverage_audit.py` | `test_or_qa` | `tests/qa/` | Later | **Uncertain** — imports missing `nonnurse_index_lib` |
| `test_minimal.py`, `ultra_minimal_test.py` (root) | `test_or_qa` | `tests/` or `archive_candidate` | Later | Ad-hoc smoke |
| `.verify_output/` | `test_or_qa` | local only | No | Gitignored QA artifacts |

### Documentation

| Current path | Role | Proposed path | Move now? |
|--------------|------|---------------|-----------|
| `docs/` | `documentation` | `docs/` | No |
| `DATA_PIPELINE_DOCUMENTATION.md`, `DEPLOYMENT_RULES.md` (root) | `documentation` | `docs/` | **Yes** — doc-only moves |
| `docs/DATA_PATHS_AUDIT.md`, `docs/SAFE_CLEANUP_EXECUTION_PLAN.md` | `documentation` | keep | No |
| `docs/testing_backlog.md` | `documentation` | keep | No |

### Archive candidates (confirm before any move)

| Current path | Role | Proposed path | Move now? | Notes |
|--------------|------|---------------|-----------|-------|
| `backups/` | `archive_candidate` | `archive/backups/` | Later | Gitignored session snapshots |
| `Archive/` | `archive_candidate` | `archive/legacy/` | Later | Old dashboard variants |
| `debug_contract.py`, `tatus --porcelain`, `Untitled`, `tmp_*.txt` | `archive_candidate` | `archive/junk/` | Later | No imports |
| Root `pbj_v2_dashboard_extras.js` | `uncertain` | delete or archive after diff | **No** | Stale duplicate? 5,633 vs 6,314 lines in `static/js/` |
| `scripts/pbj_dashboard2.py`, `scripts/nurse_staffing_app.py` | `archive_candidate` | `archive/legacy_apps/` | Later | Legacy app copies in scripts/ |
| `225500/`, `superdynamic/` | `uncertain` | TBD | **No** | Purpose unclear |

---

## §B — Special attention items

### B.1 `templates/superdynamic_dashboard_v2.html`

**Status:** **Live, active V2/V3 dashboard shell** — not obsolete.  
**Size:** ~57,410 lines / ~2.7 MB.  
**Canonical source:** Repo `templates/` (bootstrap copies to deployments; see `scripts/bootstrap_superdynamic_v2_facility.py`).

**Already extracted** (via `{% include %}`):

- Brand / nav chrome: `brand_styles`, `guided_nav*`, `floating_controls_panel`, `premium_banner`
- Modals: `ai_data_pack_modal`, `chow_ownership_modal`, `facility_events_modal`, `dashboard_how_to_modal`, compliance modals
- Major sections: `pbj_staffing_core_shell`, `staffing_patterns_workforce_section`, `risk_timeline_v3_shell`, `benchmark_view_chrome`
- V3: `report_builder_v3_*`, `v3_handoff_card`, `v3_pane_header`
- Macros: `metric_context_strip`, `chart_events_toolbar`, rollup accordions

**Recommended future splits** (phase 3+, zero behavior change per PR):

| Lines (approx.) | Section | Proposed partial |
|-----------------|---------|------------------|
| 1–220 | `<head>` meta, JSON config blobs, GA | `dashboard_v2/head_config.html` |
| 220–3730 | Inline CSS blocks | `dashboard_v2/inline_styles.html` or move to `static/css/dashboard_v2.css` |
| 3730–14270 | Additional styles / script setup | `dashboard_v2/styles_extension.html` |
| 14270–14290 | Top chrome includes | already partials ✓ |
| 14288–14492 | Report builder pane (`#pbjReportBuilderPane`) | extend `partials/v2/report_builder_v3_pane.html` |
| 14492–14853 | Facility header + summary KPI strips | `dashboard_v2/facility_hero.html` |
| 14853–15108 | Staffing core + charts scaffold | already `pbj_staffing_core_shell` ✓; split chart grid further |
| 15108–15200 | Workforce / patterns | `staffing_patterns_workforce_section` ✓ |
| 15196+ | Risk timeline V3 | `risk_timeline_v3_shell` ✓ |
| 15462+ | Compliance modals/settings | partials exist ✓ |
| 16112+ | Period screening / appendices | `period_screening_card` + new partials |
| 57403 | Closing scripts | `guided_nav_shell_scripts` ✓ |

**Do not** archive or replace with V1 template. Production Vercel bundles use this file when `PBJ_SUPERDYNAMIC_TEMPLATE=v2`.

---

### B.2 `static/js/pbj_v2_dashboard_extras.js`

**Status:** **Live V2 client bundle** — staffing hub layout, scope labels, chart chrome, CHOW/AI helpers.  
**Size:** ~6,314 lines. **Canonical:** `static/js/` (also copied to each deployment; root copy is **uncertain/stale**).

**Logical modules** (recommended future files under `static/js/dashboard_v2/`):

| Module | Approx. lines | Responsibility | Existing overlap |
|--------|---------------|----------------|------------------|
| `plotly_chrome.js` | 1–700 | Plotly legend, axis, mobile layout, dynamic Y | partial split possible from `pbj_v2_chart_scope_toolbar.js` |
| `viewport_copy.js` | 30–120 | Mobile breakpoint, citation chip labels | |
| `scope_labels.js` | 1200–2100 | Scope label DOM, filter info strings | overlaps `pbj_v2_scope_config.js` |
| `floating_period.js` | 1900–2100 | Floating control center period sync | overlaps `floating_controls_panel` |
| `controls_dock.js` | 3400–4000 | Draggable controls dock position/persist | |
| `census_context.js` | 4800–5100 | Census/occupancy context strips, certified beds | |
| `spark_summary.js` | 4700–4820 | KPI sparkline painting from chart traces | overlaps `pbj_sparkline.js` |
| `ai_context_pack.js` | 5900–6030 | AI export JSON/CSV | overlaps `pbj_rb3_ai_bridge.js` |
| `guided_nav_wire.js` | 6050–6314 | How-to modal, guided nav tool buttons, init | overlaps `guided_nav_shell_scripts` |

**Already split out** (do not duplicate work): `pbj_v2_peer_comparison.js`, `pbj_facility_events.js`, `pbj_v2_chart_scope_toolbar.js`, `pbj_v3_panes.js`, `pbj_v3_handoffs.js`, `pbj_report_builder_v3.js`.

**Phase 3 approach:** extract one module at a time; keep thin `pbj_v2_dashboard_extras.js` re-export shim until all call sites updated in templates **and** deployment copies (or add sync script to publish pipeline).

---

### B.3 `dynamic_facility_dashboard.py`

**Verdict:** **Live app code with pipeline and data-layer mixing** — high-risk monolith.

| Concern | Evidence |
|---------|----------|
| **Routing** | 40+ `@app.route` handlers (`/`, `/api/*`, favicon, data-matching page) |
| **Rendering** | `render_template("dynamic_facility_dashboard.html")` — **legacy V1 shell**, not `superdynamic_dashboard_v2.html` |
| **Data loading** | `initialize_data()`, `create_dynamic_dashboard()`, reads standardized CSVs, provider info, lite metrics |
| **Pipeline logic embedded** | `create_facility_complete_csv()`, `create_facility_provider_info_csv()` — reads `standardized_PBJ/`, writes facility CSVs (same logic needed for deploy slices) |
| **Transform in request path** | Heavy pandas in API handlers (charts, compliance, EIN bridge) |
| **Deploy awareness** | sys.path hack when run from `deployments/pbj320-<CCN>/` (lines 13–25) |

**Relationship to V2:** `deployments/.../facility_*_superdynamic_dashboard.py` is a **fork/evolution** — adds `_superdynamic_dashboard_template_name()`, V2/V3 panes, renders `superdynamic_dashboard_v2.html`. `deployment_entrypoint_guard.py` explicitly preserves these entrypoints during packaging.

**Danger flags:** Any refactor must not assume `dynamic_facility_dashboard.py` == production V2. Production V2 is the **deployment superdynamic entrypoint**. Canonical shared logic should migrate to `app/services/`, not delete the legacy file prematurely.

---

### B.4 `deployments/.../facility_*_superdynamic_dashboard.py`

**Treat as:** **Generated / deploy-specific artifacts** (bootstrap + packaging), **not** hand-edited canonical source.

| Evidence | |
|----------|--|
| `scripts/bootstrap_superdynamic_v2_facility.py` creates bundle from ref CCN `315461` | |
| `create_vercel_deployment.py` detects V2 and **preserves** entrypoint | |
| `deployment_entrypoint_guard.py` blocks accidental overwrite | |
| `vercel.json` `builds[].src` points at this file | |

**Workflow:** Edit shared libs + templates at repo root → run bootstrap/deploy script → verify bundle.  
**Do not** manually diverge deployment copies without a merge-back plan.

---

### B.5 `pbj_lite/facility_lite_metrics.csv`

**Status:** **Intentionally tracked (46 MB)** — deploy pipeline input.

| Consumer | Usage |
|----------|-------|
| `create_vercel_deployment.py` | `write_state_facility_lite_metrics_slice()` copies state-scoped slice into `deployments/.../pbj_lite/` |
| `scripts/ensure_facility_county_lite_slice.py` | Ensures slice exists before deploy |
| `geo_distribution_lib.py` | Peer geography / county column |
| `dynamic_facility_dashboard.py` | `_resolve_pbj_lite_csv()` |

**Do not** remove from git without: (1) alternative hosting for deploy slices, (2) packaging script update, (3) Vercel smoke on regional county features.

**Future:** move to `data/deploy_slices/pbj_lite/` but keep committed (or commit a compressed sample + document full file download).

---

### B.6 `provider_info_extracted/ProviderInfo_*.csv`

**Status:** ~98 files, **~870 MB**, **still in git index** despite `.gitignore` entry (pre-ignore commit).

**Safe migration plan (phase 2b — separate PR, no local delete):**

1. Confirm `.gitignore` contains `provider_info_extracted/` (phase 1 ✓).
2. Confirm local files remain after index removal (`git ls-files` empty, files still on disk).
3. Remove from index only (commands deferred to that PR's runbook — not executed in planning phase).
4. Verify dashboards still resolve paths via `geo_distribution_lib.py` / `data_path_resolver` (local disk or junction).
5. Optional: add `data_paths.local.json.example` documenting external path.
6. Communicate: clones will not include these CSVs; pipeline can regenerate into `provider_info_extracted/` from `provider_info/` zips.

**Also consider untracking:** `provider_info/NH_ProviderInfo_*.csv` (9 files, raw layer).

---

### B.7 `EIN/`, `NonNursecsv/`, `provider_info_combined.csv`

| Path | Gitignored | On disk | Conceptual home |
|------|------------|---------|-----------------|
| `EIN/` | **Yes** | ~9 GB | `data/raw/ein/` |
| `NonNursecsv/` | **Yes** | ~19 GB | `data/raw/pbj_nonnurse/` |
| `PBJcsv/` | **Yes** | ~10 GB | `data/raw/pbj_nurse/` |
| `provider_info_combined.csv` | **Yes** | ~834 MB (root) | `data/processed/provider_info/combined.csv` |

**Migration:** Prefer Windows junctions per `docs/DATA_PATHS_AUDIT.md` — no code changes until `cms_data_paths.py` delegates to `data_path_resolver.py`. **Do not delete** local copies during cleanup.

---

## §C — Tailored target structure

Adjusted for what actually exists today (V2/V3 active, `deployments/` special, `pbj_identifiers/`, `streamlit-extras/`):

```txt
app/
  routes/
    pbj_dashboard_streamlit.py      # PBJ_Dashboard.py
    dynamic_facility_dashboard.py   # legacy Flask (V1 template)
    staging/
  services/
    facility_report_lib.py
    facility_ein/
    geo_distribution_lib.py
    report_findings_engine.py
    report_builder_v3.py
    deploy/
      entrypoint_guard.py
      guardrails.py
  templates/
    dashboard_v1/                   # dynamic_facility_dashboard.html
    dashboard_v2/                   # superdynamic_dashboard_v2.html + partials
    dashboard_v3/                   # report builder, risk timeline shells
    shared/                         # url macros, brand
  static/
    js/
      dashboard_v2/                 # split from pbj_v2_dashboard_extras.js over time
      dashboard_v3/
      shared/                       # sparkline, utils
    css/
  config/
    data_path_resolver.py           # moved with cms_data_paths.py
  packages/
    pbj_identifiers/

pipelines/
  extract/
    standardize_pbj.py
    standardize_nonnurse.py
    manage_cms_sources.py
  transform/
    run_pipeline_update.py
    generate_metrics.py
    normalize_provider_info.py
  publish/
    create_vercel_deployment.py
    bootstrap_superdynamic_v2.py
    deploy_vercel_facility.py
  provider_info/
    prov_info.py
  ein/
    ingest_quarter_csv.py
    organize_ein_folder.py

data/
  raw/              # gitignored — PBJcsv, NonNursecsv, EIN, provider_info
  interim/          # gitignored — provider_info_extracted, indexed, caches
  processed/        # gitignored — standardized_*, provider_info_normalized, combined CSVs
  deploy_slices/    # committed selective — pbj_lite/*.csv, per-CCN slices (policy TBD)
  samples/          # committed small — gis, macpac, registries, legislation standards

deployments/        # KEEP at repo root (Vercel contract)
  pbj320-<CCN>/     # generated bundles — treat as publish output
  generated/        # future: optional subfolder for new bundles (migration TBD)

scripts/            # KEEP — operational QA/audits (see testing_backlog.md)
tests/
  fixtures/
  qa/               # future home for scripts/qa_*.py (optional)

docs/
  repo_cleanup_plan.md
  testing_backlog.md
  DATA_PATHS_AUDIT.md
  runbooks/

archive/
  backups/
  legacy/
  junk/

streamlit-extras/   # vendor — unchanged
config/             # deploy JSON — unchanged at root or under app/config/
```

**What stays at repo root during transition:** `render.yaml`, `requirements.txt`, `deployments/`, entrypoint symlinks or thin wrappers until imports migrate.

---

## PR #1: Documentation + guardrails (this PR)

**Scope — include only:**

- `docs/repo_cleanup_plan.md`
- `docs/testing_backlog.md`
- `.gitignore`

**Explicitly excluded:** file moves, JS/template splits, provider data untracking, pipeline restructuring, deployment edits, V2/V3 behavior changes.

**Stage and verify before commit:**

```powershell
git add .gitignore docs/repo_cleanup_plan.md docs/testing_backlog.md
git status --short
git diff --cached --stat
git diff --cached -- .gitignore docs/repo_cleanup_plan.md docs/testing_backlog.md
```

Confirm `git diff --cached --stat` lists **only** those three paths.

---

## PR #2: Untrack provider info artifacts from git index

**Status:** Planned only — **do not run** until PR #1 is merged.

### Pre-flight (read-only dependency checks)

```bash
git ls-files provider_info_extracted
git ls-files "provider_info/NH_ProviderInfo_*.csv"

git grep -n "provider_info_extracted"
git grep -n "NH_ProviderInfo_"
```

**Snapshot (2026-06-13):**

| Check | Result |
|-------|--------|
| `git ls-files provider_info_extracted` | **96 files** tracked (~870 MB index) |
| `git ls-files "provider_info/NH_ProviderInfo_*.csv"` | **8 files** in `provider_info/` |
| `provider_info_extracted` code refs | Runtime **optional** — `dynamic_facility_dashboard.py`, `geo_distribution_lib.py`, `cms_data_paths.py`, `data_path_resolver.py` resolve **local disk** paths; dashboards fall back to CMS ZIP URLs when absent |
| `NH_ProviderInfo_` code refs | **Pipeline** (`normalize_provider_info.py`, `manage_cms_sources.py`) reads from `provider_info/` on disk; **not** a Vercel bundle requirement |

### Future commands (after pre-flight passes — not executed yet)

```bash
git rm --cached -r provider_info_extracted
```

Optionally, **only if** raw `provider_info/NH_ProviderInfo_*.csv` files are confirmed non-runtime/non-deploy dependencies (pipeline reads local `provider_info/`; clones can re-download):

```bash
git rm --cached provider_info/NH_ProviderInfo_*.csv
```

### Post-untrack verification

```powershell
python scripts/audit_data_paths.py --quick
python -m pytest tests/test_data_path_resolver.py -q
git status
# Confirm files still exist on disk:
Test-Path provider_info_extracted
Get-ChildItem provider_info/NH_ProviderInfo_*.csv | Select-Object -First 3
```

### Important notes

- Removes files from the **git index only** — local files are untouched.
- Does **not** delete working-tree files.
- Does **not** rewrite old git history.
- Reduces future repo churn and prevents accidental recommits.
- Does **not** by itself shrink historical clone size (blobs remain in history until a separate history rewrite, which is out of scope).
- **Do not** untrack `pbj_lite/facility_lite_metrics.csv` — intentionally tracked for deployment slices (`create_vercel_deployment.py`).

---

## Phase 1 deliverables (PR #1 — complete)

- [x] `docs/repo_cleanup_plan.md`
- [x] Expanded `.gitignore`
- [x] `docs/testing_backlog.md`
- [x] No file moves / import changes / behavior changes

### Verify after any cleanup PR

```powershell
cd C:\Users\egold\PycharmProjects\PBJapp
python scripts\audit_data_paths.py --quick
python -m pytest tests\test_data_path_resolver.py -q
git status
```

---

## Appendix: live app surfaces

| Surface | Entry | Template | Deploy |
|---------|-------|----------|--------|
| National Streamlit | `PBJ_Dashboard.py` | Streamlit pages | Render |
| Facility V2/V3 Flask | `deployments/pbj320-<CCN>/facility_<CCN>_superdynamic_dashboard.py` | `superdynamic_dashboard_v2.html` | Vercel |
| Facility legacy Flask | `dynamic_facility_dashboard.py` | `dynamic_facility_dashboard.html` | Local / legacy |
| Pipeline CLI | `run_pipeline_update.py` | — | CLI |

## Appendix: path resolver

`data_path_resolver.py` — env → `data_paths.local.json` → repo-relative default. Any `data/` rehome requires updating `DATA_ROOT_KEYS` and `cms_data_paths.py` together.
