# Testing & QA backlog

**Purpose:** Track semi-active QA scripts and audit tools that are **not junk** and are **not yet in CI**.  
**Policy:** Prefer moving wired tests to `tests/`; keep operational scripts in `scripts/` until a `tests/qa/` migration is planned. Do **not** archive by default.

**Related:** `docs/repo_cleanup_plan.md` (role: `test_or_qa`)

---

## In CI today (`tests/`)

| File | Type | Coverage area |
|------|------|---------------|
| `test_data_path_resolver.py` | unit | Data path resolver |
| `test_pbj_federal_compliance.py` | unit | Federal compliance payloads |
| `test_red_flag_quarter_resolver.py` (+ `.js`) | unit | Red-flag quarter logic |
| `test_pre_post_quarter_anchor.py` (+ `.js`) | unit | Pre/post quarter anchors |
| `test_ein_employee_id_continuity.py` | unit | EIN employee ID continuity |
| `test_ein_sustained_work_patterns.py` | unit | Sustained work flags |
| `test_daily_staffing_flags.py` | unit | Daily staffing flags |
| `test_admin_turnover_review.py` | unit | Admin turnover review |
| `test_audit_facility_pbj_coverage.py` | unit | Facility PBJ coverage audit |
| `test_cms_pbj_daily_explorer_url_golden.py` | golden | CMS explorer URLs |
| `test_facility_ein_extract_plan.py` | unit | EIN extract plan |
| `test_weighted_summary_contract_pct.py` | unit | Contract % weighting |
| `test_ownership_disclosures.js` | unit | Ownership disclosures UI |
| `test_pbj_*` (display name, events, roster, apply quarter) | unit/JS | Dashboard client helpers |
| `fixtures/` | data | Small JSON/Python fixtures |

**Run:** `python -m pytest tests/ -q`

---

## Pre-deploy guardrails (`scripts/check_*`, `scripts/preflight_*`)

| Script | Role | Suggested CI stage |
|--------|------|-------------------|
| `check_v2_deployment_bootstrap.py` | V2 bundle layout | pre-deploy |
| `check_v2_deployment_import.py` | Import sanity in deploy dir | pre-deploy |
| `check_v2_static_js_bundle.py` | JS bundle parity | pre-deploy |
| `check_v2_inline_js.py` | Inline JS guardrail | pre-deploy |
| `check_v2_evidence_layout.py` | Evidence layout | manual |
| `check_pbj_events_hover_legend_guardrail.py` | Chart legend | pytest candidate |
| `check_facility_cms_data_ready.py` | CMS data readiness | pipeline gate |
| `preflight_v2_facility_deploy.py` | Full preflight | pre-deploy |
| `pbj_premium_verify_auth.py` | Premium auth | manual |

---

## Browser / HTTP QA (`scripts/qa_*`, `scripts/smoke_*`, `scripts/verify_*`)

| Script | Role | Notes |
|--------|------|-------|
| `qa_phase1_perf_http.py` | HTTP perf | Phase 1 perf audit |
| `qa_phase1_perf_browser.py` | Browser perf | Playwright-style |
| `qa_phase2_browser.py` | Browser QA | Phase 2 |
| `qa_phase_ein1_browser.py` | EIN browser QA | CCN 315461 |
| `qa_ein1_focused_browser.py` | Focused EIN | |
| `qa_v3_mobile_smoke.py` | V3 mobile | Active V3 work |
| `qa_v3_panes_browser.py` | V3 panes | |
| `qa_v3_export_smoke.py` | V3 export | |
| `qa_v3_risk_timeline_smoke.py` | Risk timeline | |
| `smoke_test_ny_minimum_insights_flask.py` | Flask smoke | NY insights |
| `verify_premium_315461_live.py` | Live premium | Production check |
| `verify_red_flag_resolver_315461_smoke.py` | Red flag smoke | |
| `verify_ny_minimum_insights_report.py` | Report verify | |

**Output:** often writes to `.verify_output/` (gitignored). Keep scripts; document runs here until CI job exists.

---

## Data / pipeline audits (`scripts/audit_*`)

| Script | Role |
|--------|------|
| `audit_data_paths.py` | Path resolver + disk layout |
| `audit_disk_dedupe.py` | Duplicate file detection |
| `audit_facility_pbj_coverage.py` | Facility coverage |
| `audit_ein_extraction.py` | EIN extraction |
| `audit_nonnurse_extraction.py` | Non-nurse extraction |
| `audit_report_builder_v3.py` | Report builder V3 |
| `audit_report_findings_quality_315461.py` | Findings quality |
| `cms_quarter_status.py` | Quarter inventory |

**Run after cleanup/data moves:** `python scripts/audit_data_paths.py --quick`

---

## Root-level audit scripts (candidates to move → `tests/qa/`)

| File | Status |
|------|--------|
| `pbj_facility_coverage_audit.py` | **Blocked** — imports missing `nonnurse_index_lib` |
| `benchmark_hour_gap.py` | Ad-hoc metric validation |
| `packaging_refresh_gates.py` | Deploy packaging gates — **live**; not test junk |

---

## Legacy app copies in `scripts/` (archive only after confirmation)

| File | Notes |
|------|-------|
| `scripts/pbj_dashboard2.py` | Old dashboard variant |
| `scripts/nurse_staffing_app.py` | Old app |
| `scripts/citations_app.py` | Citations prototype |
| `scripts/streamlit_pbj_dashboard.py` | Streamlit experiment |

---

## Future CI tiers (proposal)

1. **Tier A (every PR):** `pytest tests/ -q` + `audit_data_paths.py --quick`
2. **Tier B (pre-deploy):** `preflight_v2_facility_deploy.py` + `check_v2_*` guardrails
3. **Tier C (manual/nightly):** browser QA scripts + live `verify_*` against staging CCNs

---

## Adding new QA

When adding a script:

1. Prefer `tests/` if it asserts behavior with fixtures.
2. Use `scripts/` if it needs live data, browser, or deploy credentials.
3. Add a row to this file — **do not** leave orphaned scripts at repo root.
4. Do **not** move to `archive/` unless confirmed dead and documented here.
