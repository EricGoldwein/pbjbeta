# CMS source registry + PBJ Data Ops (control plane)

**Canonical home:** PBJapp.  
**Entrypoint:** `python data_ops_app.py` (Flask + Jinja).  
**Auth:** `PBJ_DATA_OPS_PASSWORD` (required). Optional `PBJ_DATA_OPS_SECRET` for session signing.  
**Not Streamlit.** `PBJ_DASHBOARD_PASSWORD` remains customer-dashboard auth only.

## Architecture

```
UI (Flask/Jinja) → cms_data_ops / data_ops_* services → canonical pipelines (PR #63 acquire, etc.)
```

Lifecycle: detect → acquire → raw preserve → structural validation → normalize → **Zweli Check** → readiness → human approval → downstream.

Model: **source → release → artifact → derived signal → consumer**.

## Brand assets

| Asset | Status in this environment |
|-------|----------------------------|
| `320_CONSULTING_BRAND_INSTRUCTIONS.md` | **Unavailable** |
| 320 wordmark image/SVG | **Unavailable** |
| Local DM Sans / DM Mono font files | **Unavailable** |
| Used instead | Google Fonts CDN DM Sans + DM Mono; PBJ320 CSS text mark (`brand_macros` / `brand_styles`) |

## Verified CMS IDs (encoded from pbj-root watcher; no runtime dependency)

See `cms_source_registry.py` constants. Includes Provider Info, nurse/non-nurse/EIN, SNF All Owners, SNF Enrollments (separate), SNF CHOW, Chain Performance, Health Citations (`r5ix-sfxw`).

SFF: **signal** `signal.sff_status` from Provider Info; **source** `cms.sff_pdf_list` (PDF/list, UNMODELED).

## Zweli Check

`data_ops_zweli.py` — states PASS / REQUIRES_REVIEW / BLOCKED / NOT_RUN. Provider Info profile v0 + synthetic 60×/1/60 regression.

## Dashboard Builder

Existing V2 refresh: `bootstrap_superdynamic_v2_facility.run_bootstrap` + preflight.  
Cold/new V2: **unresolved** without local reference. No Deploy button in V0.
