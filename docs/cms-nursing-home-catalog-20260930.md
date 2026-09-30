# September 30 CMS Data Ops checkpoint

## A. Governance checkpoint

Branch: `checkpoint/sep30-data-ops-20260930`. Start: `d18c62323e9e6e6bfbcf16f03c96f32b2d8851bd`. Commit: `8e730366edb801c63ba56a52a4d8c1c0b8859289`. Pushed to origin.

Focused exact-index verification: 115 passed, one native-library skip, four live-state tests deselected. The native synthetic builder test passed separately with native dependencies available. Changed Python syntax checked.

Committed files:

- `active_release_client.py`
- `cms_data_ops.py`
- `cms_source_registry.py`
- `data_ops_app.py`
- `derived_provenance.py`
- `generate_metrics.py`
- `generic_cms_csv.py`
- `lite_report.py`
- `operator_freshness.py`
- `ownership_downstream_rebuild.py`
- `ownership_pairing.py`
- `premium_source_contract.py`
- `provenance_freshness.py`
- `release_check.py`
- `release_control_plane.py`
- `release_review_policy.py`
- `release_source_catalog.py`
- `scripts/preflight_v2_facility_deploy.py`
- `sff_release.py`
- `static/data_ops/data_ops.css`
- `templates/data_ops/ownership_pair_detail.html`
- `templates/data_ops/partials/ownership_pair_panel.html`
- `templates/data_ops/release_review.html`
- `templates/data_ops/sources.html`
- `tests/test_cms_data_ops.py`
- `tests/test_cms_source_registry.py`
- `tests/test_derived_provenance.py`
- `tests/test_operator_freshness.py`
- `tests/test_ownership_downstream_rebuild.py`
- `tests/test_ownership_downstream_rebuild_integration.py`
- `tests/test_ownership_same_release_revision.py`
- `tests/test_release_check.py`
- `tests/test_release_review_live_html.py`
- `tests/test_sep30_activation_safety.py`
- `tests/test_sff_release.py`
- `tests/test_sources_release_actions.py`

Unrelated dashboard/quarter-map work remains uncommitted. No runtime, downloaded artifacts, recovery snapshots, `.cursor/`, `ownership/_sources/`, `sff/`, or `state/` files were staged. The unrelated `.gitignore` edit is preserved.

## B. CMS architecture

The Provider Data page uses [structured search](https://data.cms.gov/provider-data/api/1/search?theme=Nursing%20homes%20including%20rehab%20services&page-size=100) for complete theme membership. Discovery validates pagination, totals, theme membership and duplicate stable IDs.

[Expanded dataset metastore](https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/y2hd-n93e?show-reference-ids=true) supplies `identifier`, `modified`, `released`, `nextUpdateDate`, distributions, resource IDs and version/reference tokens. `modified` is logical data vintage; `released` is posting date; `%modified` is administrative metadata time. Planned dates never establish publication. Resource tokens and local metadata fingerprints are not content checksums.

[Theme archive history](https://data.cms.gov/provider-data/api/1/archive/aggregate/theme/nursing-homes/relative) records dated theme ZIPs and annual snapshots. [Current archive metadata](https://data.cms.gov/provider-data/api/1/archive/aggregate/current/theme/all/relative) identifies the current whole-theme download. Neither surface proves atomic per-dataset publication. A changed archive cannot mark every dataset NEWER. Per-dataset archive endpoint returned 404; metastore revisions returned 401, so public per-dataset historical access was not established.

## C–D. Live catalog

Checked `2026-09-30T11:37:54.039554-04:00`. All 18 authoritative dataset lookups and archive metadata succeeded. Two successive observations: **0 NEWER, 0 REVISED, 13 PLANNED_TODAY, 5 CURRENT/not due, 0 errors**. The first observation established a baseline (18 NEW_DATASET first observations); this does not mean CMS added 18 datasets today. Change counts cover this session comparison, not unseen earlier history.

Identity below is the current resource ID/version. Full publisher fingerprint, distribution/reference identities, previous successful metadata and evidence URLs are in the local snapshot and expandable UI.

| Dataset | Stable ID | Publisher resource/version | Modified | Released | Planned | Status | Coverage |
|---|---|---|---|---|---|---|---|
| Citation Code Look-up | tagd-9999 | 6d684830d328710dabe0a20c096bcab7_1786724152 | 2026-08-01 | 2026-08-26 | 2026-09-30 | PLANNED_TODAY | Governed via Provider bundle |
| Fire Safety Deficiencies | ifjz-ge4w | 3d8cda3e909d835f20cecf639d7e21ad_1786724148 | 2026-08-01 | 2026-08-26 | 2026-09-30 | PLANNED_TODAY | Governed via Provider bundle |
| FY 2026 SNF VBP Aggregate Performance | ujcx-uaut | 9edb8fc369e3adecc5c8d50906d2ffc9_1764720378 | 2025-12-02 | 2025-12-10 | 2026-12-02 | CURRENT | Detected only |
| FY 2026 SNF VBP Facility-Level Dataset | 284v-j9fz | 24625f7ba546a31aafbd5057da94a0e2_1767384370 | 2025-11-01 | 2025-12-10 | 2026-12-02 | CURRENT | Detected only |
| Health Deficiencies | r5ix-sfxw | 600f5d1861dd2e0280b2e961e8396245_1786724148 | 2026-08-01 | 2026-08-26 | 2026-09-30 | PLANNED_TODAY | Governed |
| Inspection Dates | svdt-c123 | c695a10fcf74e3a839879a307008563e_1786724153 | 2026-08-01 | 2026-08-26 | 2026-09-30 | PLANNED_TODAY | Governed via Provider bundle |
| MDS Quality Measures | djen-97ju | be86252fa6f8cc3035ec7cd9d3a14371_1786724149 | 2026-08-01 | 2026-08-26 | 2026-09-30 | PLANNED_TODAY | Built, not integrated |
| Medicare Claims Quality Measures | ijh5-nb2v | 6f3001dacc01644d9f284248ed6af75f_1786724149 | 2026-08-01 | 2026-08-26 | 2026-09-30 | PLANNED_TODAY | Built, not integrated |
| Nursing Home Data Collection Intervals | qmdc-9999 | 0ac4d8d665411a251c4d3c4156d0dfd1_1786724152 | 2026-08-01 | 2026-08-26 | 2026-09-30 | PLANNED_TODAY | Governed via Provider bundle |
| Ownership | y2hd-n93e | cc4bedc83a7b3752efb24f7817a856ce_1786724150 | 2026-08-01 | 2026-08-26 | 2026-09-30 | PLANNED_TODAY | Governed via Provider bundle |
| Penalties | g6vv-u9sr | 22b16c9dfd718b1b2a003eb8d4222046_1786724150 | 2026-08-01 | 2026-08-26 | 2026-09-30 | PLANNED_TODAY | Detected only |
| Provider Information | 4pq5-n9py | 328596835e6db31b2564cd733c3795f4_1786724150 | 2026-08-01 | 2026-08-26 | 2026-09-30 | PLANNED_TODAY | Governed |
| Skilled Nursing Facility Quality Reporting Program - National Data | 5sqm-2qku | 535db1952768202fd80d81285ae16a65_1784297729 | 2026-07-01 | 2026-07-29 | 2026-10-28 | CURRENT | Detected only |
| Skilled Nursing Facility Quality Reporting Program - Provider Data | fykj-qjee | 3078f089eb03a8f5caf2de0d7bbcbe1b_1786724151 | 2026-08-01 | 2026-08-26 | 2026-10-28 | CURRENT | Built, not integrated |
| Skilled Nursing Facility Quality Reporting Program - Swing Beds - Provider Data | 6uyb-waub | b260bfaa27393e8b12bfa95d3bb18c96_1784297730 | 2026-07-01 | 2026-07-29 | 2026-10-28 | CURRENT | Detected only |
| State US Averages | xcdc-v8bm | c8aea0b9df2b5da58a9ed58cb24abffa_1786724151 | 2026-08-01 | 2026-08-26 | 2026-09-30 | PLANNED_TODAY | Detected only |
| State-Level Health Inspection Cut Points | hicp-9999 | 995d42ad054917707fbb07b41e22cf4d_1786724153 | 2026-08-01 | 2026-08-26 | 2026-09-30 | PLANNED_TODAY | Out of scope |
| Survey Summary | tbry-pc2d | 072c3220af015d4b4fb4d471cf790c78_1786724152 | 2026-08-01 | 2026-08-26 | 2026-09-30 | PLANNED_TODAY | Governed via Provider bundle |

PECOS is independently checked: `cms.snf_all_owners` REVISED for logical release `2026-07-31`; `cms.snf_enrollments` CURRENT. Owners version `48ca01a7-7176-4882-9273-fef560dc763c`, file UUID `93535382-e382-4338-9307-a18ce27d041f`, publisher modified September 29. Provider Data Ownership `y2hd-n93e` is PLANNED_TODAY and is never used for PECOS freshness.

Owners ACTIVE SHA remains `4346a8d44a06ab5214f8d23f86c23f2ddf88de6d14e339a1580aa95dc7c3f435`; revised candidate SHA `dc2d46642b81e5dd78998bf03519a39cbbdce932ef6433ef03f0af0188f5ba7f`, 295083 rows, remains VALIDATED. Enrollments ACTIVE is unchanged; no fabricated candidate.

## E. Coverage audit

Read-only claims worktree verified at `codex/data-ops-claims-qm-governance-20260919`, `f09c7cb6a4bc583d8c35cb38531ee4f9d471927b`. Nothing merged. Classes: A operational; B Provider extraction/retention member; C lifecycle elsewhere; D detection only; E intentional scope exclusion.

| Dataset | Class / operational ID | Lifecycle / validator | Consumer | Gap |
|---|---|---|---|---|
| Citation Code Look-up | B / cms.provider_info | Provider extraction manifest; citation description member hashes | Citation code descriptions | No independent ACTIVE dataset |
| Fire Safety Deficiencies | B / cms.provider_info | Provider retained member inventory/extraction; hashes only | Retained source archive; no dedicated downstream consumer | No standalone semantic validator or ACTIVE lifecycle |
| FY 2026 SNF VBP Aggregate Performance | D / none | CMS catalog discovery only | None governed operationally | No governed lifecycle; some files retained as adjacent CMS program members |
| FY 2026 SNF VBP Facility-Level Dataset | D / none | CMS catalog discovery only | None governed operationally | No governed lifecycle; some files retained as adjacent CMS program members |
| Health Deficiencies | A / cms.health_citations | health_citations_acquire; schema/provenance validation; explicit ACTIVE | Facility citations and public citation surfaces | Acquisition and review remain source-specific |
| Inspection Dates | B / cms.provider_info | Provider retained survey member inventory/extraction; hashes only | Retained source archive | No standalone semantic validator or ACTIVE lifecycle |
| MDS Quality Measures | C / cms.quality_measures_mds | mds_quality_measures_lifecycle.prepare_mds_quality_measures_candidate | Normalized MDS measure candidates in claims worktree | cms.quality_measures_mds lifecycle at f09c7cb; no operational integration |
| Medicare Claims Quality Measures | C / cms.quality_measures_claims | claims_quality_measures_lifecycle.prepare_claims_quality_measures_candidate | Normalized claims measure candidates in claims worktree | cms.quality_measures_claims lifecycle at f09c7cb; no operational integration |
| Nursing Home Data Collection Intervals | B / cms.provider_info | Provider extraction manifest; interval CSV member hashes | Provider processing-month/quarter mapping | No independent ACTIVE dataset |
| Ownership | B / cms.nh_ownership | Provider extraction/promotion bundle member; member hash + ownership schema | Care Compare ownership in facility bundles | Co-versioned Provider member; distinct from PECOS Owners |
| Penalties | D / none | CMS catalog discovery only | None governed operationally | No governed lifecycle; some files retained as adjacent CMS program members |
| Provider Information | A / cms.provider_info | ProviderInfo adapter; normalized bundle + Zweli; explicit ACTIVE | Provider charts and public/provider facility slices | Required region/CMI builders block activation until proven |
| Skilled Nursing Facility Quality Reporting Program - National Data | D / none | CMS catalog discovery only | None governed operationally | No governed lifecycle; some files retained as adjacent CMS program members |
| Skilled Nursing Facility Quality Reporting Program - Provider Data | C / cms.snf_qrp_provider | snf_qrp_provider_lifecycle.prepare_snf_qrp_provider_candidate | Normalized SNF QRP candidates in claims worktree | cms.snf_qrp_provider lifecycle at f09c7cb; no operational integration |
| Skilled Nursing Facility Quality Reporting Program - Swing Beds - Provider Data | D / none | CMS catalog discovery only | None governed operationally | No governed lifecycle; some files retained as adjacent CMS program members |
| State US Averages | D / none | CMS catalog discovery only | None governed operationally | No governed lifecycle; some files retained as adjacent CMS program members |
| State-Level Health Inspection Cut Points | E / none | Provider inventory policy: documentation_reference | Reference archive | Documentation reference; no operational source lifecycle |
| Survey Summary | B / cms.provider_info | Provider retained survey member inventory/extraction; hashes only | Retained source archive | No standalone semantic validator or ACTIVE lifecycle |

All catalog detectors use the same authoritative search/metastore adapter. A sources retain source-specific acquisition/validation/review/ACTIVE actions. B members are acquired through Provider bundle code (`scripts/cms_provider_release_lib.py`), with extraction/retention inventory and member hashes; retained survey members do not have standalone semantic validation or ACTIVE. C worktree lifecycle functions prepare validated candidates from retained bundles, and are not operationally integrated. D/E have no acquisition or ACTIVE action. CMS State US Averages is distinct from internal PBJ benchmarks.

## F. UI verification

One compact, initially collapsed section was added to `/sources`; no new navigation. Click Check CMS releases, expand Inspect datasets and publisher evidence, then expand a title to inspect current/previous identities, filenames, resource versions, dates, status explanation, lifecycle and CMS links. Governed rows link to existing source workflows; unsupported rows have no Acquire form.

Verified in an authenticated browser against the exact staged source tree with copied registry/candidate state: Sources rendered; catalog expansion displayed all 18 rows; Ownership evidence showed matching identities and correct PECOS separation; unsupported VBP/Penalties/State US Averages had no acquisition button; Owners candidate remained VALIDATED. The isolated preview needed explicit PBJ_ROOT because its scratch location has no sibling data checkout. No source action or activation was clicked.

Screenshot: `C:/Users/egold/.codex/visualizations/2026/09/30/01a0f2b4-0427-7700-86fe-dc0d90689d0b/catalog-sources.jpg`.

## G. Whole-theme ZIP

Current archive row ID `6281`, dated `2026-08-26`, HEAD length `39127248`, Last-Modified `Wed, 26 Aug 2026 17:01:14 GMT`. No ETag or guaranteed content checksum. Latest dated publication: `{'id': '9266', 'name': 'Theme: nursing-homes (2026-08-26)', 'date': '2026-08-26', 'url': '/provider-data/sites/default/files/dataset-archives/unknown-type/nursing-homes_2026-08-26.zip', 'size': '34053466'}`.

A bounded central-directory/manifest range audit found 20 entries: 18 CSVs, dictionary PDF and manifest. The manifest matches all 18 live stable IDs/resource filenames, including mixed monthly/quarterly/annual vintages. Routine checks use metadata and HEAD, not ZIP bodies. The UI provides an explicit CMS download link. Existing Provider bundle code extracts five primary members and inventories/retains other members; a future reviewed acquisition bundle could reuse this, but would require explicit member validation and publication consistency checks. No atomicity contract was established.

## H. Verification

Exact staged source tree: 137 passed, one native DuckDB dependency skip. Covers catalog membership/additions, next month, same-month replacement, planned date, failure preservation, removal/regression, identity separation, unsupported actions, archive independence, compatibility discovery without ZIP downloads, and governance/review regressions. The isolated native builder test passed separately (1 passed); 138 tests verified in total. The final workflow-link correction passed the catalog/source-action rerun (24 passed). Existing live-state tests are excluded from synthetic unit proof; live CMS and browser checks are reported above.

## I. Catalog Git checkpoint

Branch `feat/cms-nursing-home-catalog-20260930`, starting at governance checkpoint `8e730366edb801c63ba56a52a4d8c1c0b8859289`. Commit contains the canonical catalog module, metadata-only theme compatibility, detect-only check integration, compact template/CSS, tests and this report. Ending SHA/push result are reported in the final response.

No real source activation, ownership downstream rebuild, dashboard packaging or deployment occurred.

Preserved working-tree files:

- `.gitignore`
- `cms_data_ops.py`
- `data_ops_app.py`
- `data_ops_dashboard.py`
- `prov_info_quarter_map.py`
- `pytest.ini`
- `scripts/cms_provider_release_lib.py`
- `scripts/preflight_v2_facility_deploy.py`
- `scripts/start_local_data_ops.ps1`
- `static/data/interval_quarter_mapping.json`
- `static/data_ops/data_ops.css`
- `static/data_ops/data_ops.js`
- `templates/data_ops/base.html`
- `templates/data_ops/dashboard_builder.html`
- `templates/data_ops/partials/source_detail_panel.html`
- `tests/test_cms_provider_release.py`

Preserved untracked work: `.cursor/`, `data_ops_dashboard_jobs.py`, `provider_quarter_mapping.py`, `scripts/check_provider_quarter_flow.py`, `tests/test_data_ops_dashboard.py`, `tests/test_data_ops_dashboard_jobs.py`, `tests/test_provider_quarter_mapping.py`, `ownership/_sources/`, `sff/`, `state/`.

Browser verification additionally followed the Ownership catalog link successfully into the existing Provider Information bundle source detail. The standalone Ownership route is deliberately not used.

