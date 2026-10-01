# Data Ops completion: September 30, 2026

This continuation is confined to Data Ops. PBJapp and Vercel deployment state were not changed.

## Repository and checkpoint

Repository: `C:\Users\egold\PycharmProjects\pbj-data-ops`; branch: `feat/cms-nursing-home-catalog-20260930`; starting SHA: `09a1313eb2792dcec04ca6489bb4dcdd960e256c`. Ending implementation SHA: `4a8a1863c219b52448d5eb345a4e6a91ab069ea3`, successfully pushed to the existing upstream. A documentation-only follow-up records this exact SHA; final repository SHA is recorded in `_scratch/data-ops-ending-sha.txt`. Exact commit/push evidence is in `_scratch/data-ops-checkpoint.log` and `_scratch/data-ops-push.log`.

The preexisting dirty files and untracked datasets were preserved. `cms_data_ops.py` already had unrelated edits: only the three-line Survey Summary dispatch was staged against HEAD. No broad staging, reset, stash, clean, PR, merge, or PBJapp edits occurred.

Changed implementation paths: `cms_source_registry.py`, `cms_nh_catalog.py`, `source_family_inventory.py`, `source_evidence.py`, `survey_summary.py`, `data_ops_app.py`, and the selected dispatch in `cms_data_ops.py`; `schemas/cms_survey_summary.json`; `templates/data_ops/survey_summary.html` and `templates/data_ops/partials/survey_summary_panel.html`; tests `test_survey_summary.py`, `test_survey_summary_ui.py`, `test_source_family_inventory.py`, `test_source_evidence.py`, `test_cms_data_ops.py`, and `test_release_review_live_html.py`; this handoff document.

## Survey Summary result

Official identity: [CMS Survey Summary, tbry-pc2d](https://data.cms.gov/provider-data/dataset/tbry-pc2d). The official metastore was re-resolved before acquisition. It reported modified **2026-08-01**, released **2026-08-26**, planned update **2026-09-30**. Planned date was not treated as a new release.

Resolved resource: `072c3220af015d4b4fb4d471cf790c78_1786724152/NH_SurveySummary_Aug2026.csv`. Publisher artifact identity: `9b534d95f43a446f15d81e427cd3ab4fafe6453a1afcc9e239dd0f53d5dec547`.

Candidate: `cms.survey_summary`, release `2026-08-9b534d95f43a-76ad33cab36e`, lifecycle **VALIDATED**, validation **PASS**, review-ready, **no ACTIVE release**. No activation was performed.

Raw SHA256: `76ad33cab36e5a07b0046669eefb1cdaf3f2e3e0766345a772a90498bde52844`; 9,925,239 bytes; **43,931 rows**, **14,690 CCNs**, **41 columns**. Grain is provider inspection cycle; **CCN + Inspection Cycle is unique**. CCN + Health Survey Date has **one duplicate row**, confirming that date is unsuitable as the key. Letter-bearing CCNs are preserved. Missing survey dates and corresponding missing fire-safety counts remain missing, rather than being converted to zero.

Immutable content-addressed raw file and metadata are retained under `state/source_artifacts/cms.survey_summary/76ad33cab36e5a07b0046669eefb1cdaf3f2e3e0766345a772a90498bde52844/`. The canonical candidate in `state/release_candidates.json` points to the source and immutable validation receipt. `_scratch/data-ops-source-audit.json` records their paths, provenance and verification. Runtime raw/state artifacts are retained locally and deliberately not broadly staged into Git.

Validation checks the exact schema contract, row width/nonempty rows, six-character numeric or letter-bearing CCN shape, inspection cycles 1–3, unique cycle key, ISO dates/date bounds/processing month, and nonnegative integer counts with narrowly allowed missing fire-safety values. The receipt contains schema hash, raw hash, counts, key diagnostics, CMS metadata and URL provenance. The adapter refuses overwrite of retained bytes and preserves ACTIVE registry bytes in regression tests.

Existing Provider retention was inspected first: July Survey Summary already exists under PBJapp's `provider_info/_retained/2026-07/`; the shared Provider archive extracts inspected contain Provider/interval files. There was no current August standalone governed Summary candidate. The new slice acquires the officially resolved CSV directly and adds semantic validation; it does not create another theme ZIP extraction pipeline. A matching existing retained file can be supplied with `--retained-source`; the adapter verifies it against the official resource bytes before reuse.

Reproducible preparation: `python survey_summary.py --root <Data Ops repo>`. This detects, acquires, retains, validates and records a candidate; it does not promote. Sources and the dedicated Survey Summary page/modal show review state, counts, key and SHA. The prepare action is authenticated and does not expose an activation action.

Summary remains conceptually separate from Health Deficiencies, Fire Safety Deficiencies, Inspection Dates, Provider Information and Penalties. No foreign-key relationships or row-level citation linkage are manufactured.

## Inventory: dynamic versus static

`source_family_inventory.py` is now a projection, not a parallel registry of status claims. Governance comes from actual ACTIVE/pending control records, not dependency-graph placeholder presence. Lifecycle and candidate validation come from canonical candidate state/receipt fields. Releases come from ACTIVE or pending records, probe observations, retained-source evidence, or an explicitly labelled CMS catalog observation. Health comes from candidate validation, ACTIVE health/probes, or bounded artifact/receipt observations. Next actions follow those states and the HCRIS receipt's actual limitations. Runtime absence is labelled only for the checked canonical location.

Static metadata remains in `cms_source_registry.py`: source identity/display name, official stable CMS ID, conceptual family, publisher, expected cadence, format, semantic key/schema contract and adapter capability references. Those are descriptive contracts, not release/health/merge claims. Adjacent-source identities contain no hardcoded governance, merge, evidence or health status. `ADAPTER_REGISTERED` means a capability reference exists; it does not claim successful runtime execution. Git MERGED/UNMERGED assertions were removed rather than inferred from data files.

## Bounded adjacent-source verification and one next step each

- **Penalties:** `D:\PBJapp-data\Penalties\NH_Penalties_Aug2026.csv` exists, **15,696 rows**; raw SHA and headers are in the source audit. No Penalties ACTIVE entry was found in the current registry and no adapter is registered in this Data Ops runtime. Raw presence does not prove semantic validation or a Git merge. **Next:** validate that retained release against its source contract and record a receipt.
- **HCRIS:** existing `snf-hcris-raw-contract-pilot-v1` receipt explicitly reports **NON_PRODUCTION_RAW_CONTRACT_PILOT**, 63 selected reports and 1,622 normalized measures. Both retained source ZIP hashes match the receipt (SNF10 FY2023 and SNF24 FY2025). The receipt's unproven revision chain, Form-10 amount mapping and rate scope limitations remain explicit. **Next:** obtain/verify one prior snapshot to prove the revision/replacement chain.
- **NPI/NPPES:** no NPPES artifact was observed at `D:\PBJapp-data\clinicians\_sources\nppes`, and no corresponding governed release exists in the current Data Ops control state. Other clinician source directories exist and are not relabelled NPPES. This is scoped runtime absence, not a claim of universal absence across all worktrees. **Next:** verify any existing external NPPES acquisition receipt before adding a pipeline.

No adjacent-source pipeline was rebuilt or added.

## Tests and browser acceptance

Final result: **124 passed, 2 skipped** (30.82 seconds). The skipped live cases require pending Health Citations or ownership review items absent from current ACTIVE state. Tests ran against the existing working tree with its preserved preexisting work, not a claimed clean checkout. Full test output is saved to `_scratch/data-ops-final-tests.log`; targeted adapter/inventory/registry/catalog output also exists in `_scratch/data-ops-targeted-tests.log`. The relevant suite covers malformed/duplicate source keys, schema drift, CCN preservation, dates/counts, immutable retention, unchanged ACTIVE registry, candidate/probe projections, authenticated UI preparation, source review page/modal rendering, catalog discovery and release-control behavior.

An existing live ownership test incorrectly treated the catalog preceding Active releases as part of Needs attention. Its assertion now isolates the actual attention section, so descriptive PECOS catalog text does not masquerade as an actionable ownership row.

Authenticated browser acceptance used the actual local app at `http://127.0.0.1:8512`, with the current source/registry state. Sources showed Survey Summary **GOVERNED / VALIDATED / VALIDATION_PASS**, the correct release and explicit review action; Penalties raw evidence; HCRIS non-production pilot receipt state; and scoped NPPES absence. Clicking the inventory review link opened the Summary page and showed **VALIDATED**, **No ACTIVE release**, **PASS**, **43931 / 14690**, **CCN + Inspection Cycle; 0 duplicates**, and the correct raw SHA. Server evidence is in `_scratch/data-ops-browser-server.log`. Modal rendering/authenticated action behavior is tested; an additional browser modal attempt was interrupted by the browser control backend and is not claimed as accepted.

## Remaining scope

The requested Data Ops candidate/inventory slice is complete; Survey Summary is deliberately awaiting explicit review, not ACTIVE. Its local raw artifact/state must be retained when moving runtimes. NPPES external evidence and HCRIS pilot limitations are accurately surfaced, not resolved by creating extra pipelines. The earlier PBJapp ownership-modal deployment blocker and PBJapp upstream divergence remain outside this continuation.
