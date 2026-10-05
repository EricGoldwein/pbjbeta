# CMS nursing-home release identity audit — October 5, 2026

## Scope and result

This is a code-path and authoritative metadata audit, with narrow changes to publisher checks and their display. No live source was acquired, reacquired, activated, staged, or published. Existing ACTIVE metadata was not rewritten. Existing intentional workspace changes were preserved.

The prior Ownership fix applied only to `cms.snf_all_owners` and `cms.snf_enrollments`. Its exact-byte rule now lives in `cms_release_identity.assess_raw_identity`. Seven operational sources use it: Provider Information, PBJ nurse, PBJ non-nurse, Health Citations, SNF All Owners, SNF Enrollments, and SFF posting. Provider retains its source-specific extraction-manifest adapter; SFF retains its PDF posting adapter; PBJ treats a dataset-version date as a reporting-period label rather than a publication date.

This is **not proof that all catalog data is current**. Historical PBJ ACTIVE records lack a raw-to-normalized binding; same-period checks return UNKNOWN unless such evidence is available. Catalog-only datasets remain discovery-only and cannot detect an in-place byte revision without a content observation. They no longer assert CURRENT from matching metadata.

## Root causes and corrected paths

- `release_check.production_handlers` returned CURRENT immediately for Provider month, nurse quarter, and Health Citations month equality. These shortcuts now defer to the shared raw assessment.
- `generic_cms_csv._assess_found` ran the Ownership byte check only for the two SNF sources; non-nurse could return CURRENT with unchanged identifiers. All generic feeds now use the shared assessment.
- `sff_release.check_sff_cms` compared posting months, despite discovery already reading the PDF. Discovery now retains the SHA and the shared assessment compares it against the raw PDF bound to the normalized ACTIVE artifact. A missing PDF Updated label does not establish a release month from the URL alone.
- `cms_nh_catalog.compare_observation` called a matching resource fingerprint CURRENT and derived chronology from modified month. It now distinguishes the CMS released month, processing/modified metadata, resource ID/version, and an unobserved snapshot. Matching resource metadata is METADATA_UNCHANGED; it does not establish bytes or ACTIVE currentness. A changed resource under the same released month is REVISED.
- Sources availability cleared `new_release_available` when ACTIVE/pending IDs matched, and theme processing-month hints took priority over checks. An authoritative check now overrides those hints and restores same-period revisions. Old date-only CURRENT rows are unverified.
- Dedicated Provider/nurse/SFF check buttons now retain their check evidence separately from lifecycle registries and display CURRENT / REVISED / NEWER / UNKNOWN rather than a date-based `newer=False` claim.

The common assessment streams remote bytes without retaining an artifact. It validates the governed ACTIVE hash and the raw-source hash. A transformed primary requires a bound raw record; normalized and raw SHA values are never compared as interchangeable identities. Equal authoritative bytes are CURRENT; changed bytes under the same period are REVISED; an advanced authoritative vintage/period is NEWER. Missing proof is UNKNOWN, and a failed fetch is ERROR. Check time does not fill a missing acquisition timestamp.

## Exact official metadata observed

The complete Provider Data theme search returned **18 datasets**. Exact metastore distributions, reference resource IDs, versions, URLs, and null publisher checksums are retained in `tests/fixtures/cms_nh_release_metadata_20261005.json`. The fixture is an observation for tests, not a rewrite of historical source state.

Official listing: https://data.cms.gov/provider-data/api/1/search?theme=Nursing%20homes%20including%20rehab%20services&page-size=100

Each row was confirmed through `https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/<stable-id>?show-reference-ids=true`.

| Stable ID | CMS title | released | modified | filename period |
|---|---|---|---|---|
| 4pq5-n9py | Provider Information | 2026-08-26 | 2026-08-01 | Aug2026 |
| r5ix-sfxw | Health Deficiencies | 2026-08-26 | 2026-08-01 | Aug2026 |
| tbry-pc2d | Survey Summary | 2026-08-26 | 2026-08-01 | Aug2026 |
| y2hd-n93e | Ownership | 2026-08-26 | 2026-08-01 | Aug2026 |
| qmdc-9999 | Nursing Home Data Collection Intervals | 2026-08-26 | 2026-08-01 | Aug2026 |
| tagd-9999 | Citation Code Look-up | 2026-08-26 | 2026-08-01 | Aug2026 |
| ifjz-ge4w | Fire Safety Deficiencies | 2026-08-26 | 2026-08-01 | Aug2026 |
| svdt-c123 | Inspection Dates | 2026-08-26 | 2026-08-01 | Aug2026 |
| djen-97ju | MDS Quality Measures | 2026-08-26 | 2026-08-01 | Aug2026 |
| ijh5-nb2v | Medicare Claims Quality Measures | 2026-08-26 | 2026-08-01 | Aug2026 |
| fykj-qjee | SNF QRP Provider Data | 2026-08-26 | 2026-08-01 | Aug2026 |
| g6vv-u9sr | Penalties | 2026-08-26 | 2026-08-01 | Aug2026 |
| hicp-9999 | State-Level Health Inspection Cut Points | 2026-08-26 | 2026-08-01 | Aug2026 |
| xcdc-v8bm | State US Averages | 2026-08-26 | 2026-08-01 | Aug2026 |
| 5sqm-2qku | SNF QRP National Data | 2026-07-29 | 2026-07-01 | Jul2026 |
| 6uyb-waub | SNF QRP Swing Beds Provider Data | 2026-07-29 | 2026-07-01 | Jul2026 |
| 284v-j9fz | FY 2026 SNF VBP Facility-Level Dataset | 2025-12-10 | 2025-11-01 | FY 2026 |
| ujcx-uaut | FY 2026 SNF VBP Aggregate Performance | 2025-12-10 | 2025-12-02 | FY 2026 |

The VBP observations specifically disprove a universal modified-month = publication-vintage rule. Fiscal year and measurement windows are source-specific; no effective date was guessed from them.

Provider resource: `328596835e6db31b2564cd733c3795f4`, version `1786724150`, distribution `7a4f1244-c697-5d9d-8f84-1e10347a3a7e`, `NH_ProviderInfo_Aug2026.csv`.

Survey Summary resource: `072c3220af015d4b4fb4d471cf790c78`, version `1786724152`, distribution `dfea198b-3907-5af3-b26a-2eecd32830d1`, `NH_SurveySummary_Aug2026.csv`.

PBJ was additionally observed through its official stable slug and resources routes:

| Source | Product UUID | Current node UUID | Node version label | CMS last modified | Primary file UUID | Reporting file |
|---|---|---|---|---|---|---|
| Nurse | 7e0d53ba-8f02-4c66-98a5-14a1c997c50d | 6e5d5e28-66fd-41bc-a36c-db54dcbffd3e | 2026-01-01 | 2026-07-29 | 7e121b8b-8bba-4f74-8f01-82a13560e81b | PBJ_dailynursestaffing_CY2026Q1.csv |
| Non-nurse | b497431a-5b57-42c0-9016-90105b51841e | 723cd724-db2c-4252-b694-40f1fc7ee589 | 2026-01-01 | 2026-07-29 | 4bf59a18-1939-482b-9450-5fe35cec7ee2 | PBJ_dailyNonnurseStaffing_CY2026Q1.csv |

These node labels and file quarters do not establish a CMS publication vintage. The two primary files are 234,273,667 and 353,387,241 bytes respectively. This audit did not stream those files or assert new live byte currentness.

## Source matrix

“Metadata only” means URL/resource/version changes can be detected, but an in-place byte revision cannot. It never authorizes CURRENT or activation. All 18 catalog members inherit the common released/modified separation and honest metadata-only observation status.

| SOURCE | RELEASE IDENTITY BASIS | SNAPSHOT BASIS | BYTE CHECK | REVISION SAFE | CHANGE NEEDED |
|---|---|---|---|---|---|
| cms.snf_all_owners | Stable product → current node; CMS release label; file UUID/URL/SHA | Filename Jul 31 snapshot, separate from Aug vintage | Shared streaming SHA vs raw ACTIVE | Yes | Shared extraction; existing pair key preserved |
| cms.snf_enrollments | Same as Owners | Filename Jul 31 snapshot | Shared streaming SHA vs raw ACTIVE | Yes | Inherits shared fix |
| cms.provider_info / 4pq5-n9py | CMS released month/date + distribution resource/version/URL/SHA | Provider processing date; normalized bundle period | Shared SHA vs raw member bound by extraction manifest to ACTIVE | Yes when bound; UNKNOWN otherwise | Common check replaces date shortcut; manifest adapter retained |
| cms.health_citations / r5ix-sfxw | CMS released month/date + distribution URL/resource version/SHA | File period; CMS modified separate; survey dates are row-level | Shared SHA vs raw ACTIVE | Yes | Common check replaces month shortcut |
| cms.pbj_nurse_staffing | Stable product/current node + Primary UUID/URL/SHA; node label preserved | CY reporting quarter; publication vintage unobserved | Shared SHA, requires raw-to-normalized binding | Fail closed UNKNOWN without binding; hash revisions detected with binding | Reporting-period adapter; historical ACTIVE needs provenance review |
| cms.pbj_non_nurse_staffing | Same PBJ product/node semantics | CY reporting quarter | Shared generic assessment, requires binding | Same as nurse | Inherits common comparator and version discovery; historical binding unproven |
| cms.sff_pdf_list | Official PDF + Updated posting label + raw PDF SHA | No independent data snapshot established | Shared SHA vs stored source_pdf_hash/URI and governed normalized ACTIVE | Yes | PDF adapter now retains already-read SHA; no URL-month proof |
| cms.survey_summary / tbry-pc2d | CMS released metadata + resource/version + candidate raw SHA | Processing Date validated against CMS modified; inspection dates remain row-level | Exact official bytes during candidate preparation | Yes on preparation; metadata-only discovery cannot see in-place bytes | Existing immutable candidate logic retained; metadata/UI additions; review only |
| cms.nh_ownership / y2hd-n93e | Catalog distribution + Provider bundle member provenance | Provider processing/file period | Parent bundle member hashes; catalog metadata only | Resource revisions observed; in-place remote bytes unverified | No independent publisher-byte assertion; local derived CURRENT means built from ACTIVE upstream |
| qmdc-9999 Data Collection Intervals | CMS released + resource ID/version | Actual interval columns; modified separate | Catalog metadata; Provider bundle local member hashes | Metadata only | No standalone adapter/lifecycle added |
| tagd-9999 Citation Code Look-up | Same catalog basis | No independent snapshot inferred | Same parent/local distinction | Metadata only | Inherits catalog correction |
| ifjz-ge4w Fire Safety Deficiencies | Same catalog basis | Inspection dates are row-level | Retained Provider member; metadata discovery | Metadata only | Inherits catalog correction |
| svdt-c123 Inspection Dates | Same catalog basis | Row-level inspection dates | Retained Provider member; metadata discovery | Metadata only | Inherits catalog correction |
| djen-97ju MDS Quality Measures | Same catalog basis | Measure-specific windows; not inferred | Catalog only here | Metadata only | Candidate code elsewhere is not operationally integrated |
| ijh5-nb2v Medicare Claims Quality Measures | Same catalog basis | Measure-specific windows | Catalog only here | Metadata only | Same integration limit |
| fykj-qjee SNF QRP Provider | Same catalog basis | Measure-specific windows | Catalog only here | Metadata only | Same integration limit |
| g6vv-u9sr Penalties | Same catalog basis | Event dates; no single snapshot inferred | None in discovery | Metadata only | Discovery only; no new lifecycle |
| hicp-9999 State Health Inspection Cut Points | Same catalog basis | File period; no guessed effective date | None in discovery | Metadata only | Reference/out-of-scope workflow unchanged |
| xcdc-v8bm State US Averages | Same catalog basis | Measure-specific periods | None in discovery | Metadata only | Discovery only; distinct from internal PBJ benchmarks |
| 5sqm-2qku SNF QRP National | Same catalog basis | Measure-specific periods | None in discovery | Metadata only | Discovery only |
| 6uyb-waub SNF QRP Swing Beds Provider | Same catalog basis | Measure-specific periods | None in discovery | Metadata only | Discovery only |
| 284v-j9fz FY2026 VBP Facility | released Dec 10 + resource/version; FY is a reporting label | No effective/snapshot date inferred | None in discovery | Metadata only | Fiscal/measure semantics require an adapter if operationalized |
| ujcx-uaut FY2026 VBP Aggregate | Same released Dec 10 basis; modified Dec 2 separate | FY reporting label, not publication date | None in discovery | Metadata only | Same source-specific limit |
| cms.pbj_employee_ein_detail | Existing ZIP/member/quarter inventory; not a production check handler | Employee/quarter periods | No authoritative publisher-byte check here | Unverified | Broken legacy workflow remains explicit; not generalized into a new pipeline |
| cms.snf_chow | Unmodeled/manual cross-repository source | Not established by this audit | None | Unverified | No new adapter or currentness assertion |
| cms.chain_performance | Local inventory; not a production release-check handler | Not established by this audit | None | Unverified | No new adapter or currentness assertion |

## UI wording and remaining local shortcuts

- Sources CMS freshness requires byte evidence bound to ACTIVE, not `new_available=False`.
- A selected local CMS artifact is labelled ACTIVE; selection alone no longer creates a publisher CURRENT badge.
- Provider/Health expose CMS released date/vintage separately from file/processing metadata.
- PBJ says **CMS reporting quarter: CY2026Q1 · CMS release vintage: Not observed** when publication semantics are not established.
- Source detail exposes vintage, snapshot/file/reporting period, version date, modified date, raw SHA, byte-check time, and acquisition time separately; unavailable evidence stays “Not observed” / “Not recorded”.
- Ownership still shows **CMS release: Aug 2026 · Data snapshot / file date: Jul 31, 2026** where its observed proof supports it.
- Catalog says **METADATA_UNCHANGED** and **bytes not checked**, rather than CURRENT / unchanged bytes. Previously saved date-only CURRENT rows are displayed as metadata observations, without rewriting saved state.
- Survey Summary separately labels **CMS release / publication date** and **CMS processing date (modified)**. Its catalog ACTIVE lifecycle description now says **Review only; activation and publication unavailable**, matching the already-enforced policy.

Theme-publication modified-month values remain governed processing-period hints. They cannot establish byte CURRENT and cannot override authoritative same-period revisions. Provider/nurse acquisition libraries still have local reuse/dry-run checks; those are not publisher-currentness evidence. The production check no longer trusts their CURRENT labels. Same-period revisions in those bespoke acquisition adapters are held for source-specific immutable acquisition review rather than automatically overwriting a file.

## Changed files in this pass

`cms_release_identity.py` (new); `generic_cms_csv.py`; `release_check.py`; `cms_data_ops.py`; `health_citations_acquire.py`; `sff_release.py`; `operator_freshness.py`; `data_ops_app.py`; `cms_nh_catalog.py`; `survey_summary.py`; `templates/data_ops/partials/nh_catalog.html`; `templates/data_ops/partials/source_detail_panel.html`; `templates/data_ops/partials/survey_summary_panel.html`; `templates/data_ops/partials/ownership_pair_panel.html`; `tests/test_cms_release_identity.py` (new); `tests/fixtures/cms_nh_release_metadata_20261005.json` (new); `tests/test_release_check.py`; `tests/test_cms_nh_catalog.py`; `tests/test_cms_data_ops.py`; `tests/test_sff_release.py`; this report.

## Validation and unchanged live state

Regression coverage includes all seven operational source identities, all 18 official catalog members, separate vintage/snapshot, identical raw CURRENT, stable date/filename/IDs with changed bytes REVISED, newer vintage NEWER, missing/tampered provenance fail closed, production-handler delegation, PBJ reporting-version semantics, same-month SFF PDF revision, same-metadata Survey candidate byte revision, and check-only persistence leaving lifecycle registries unchanged. Ownership pairing and website-stage adapter tests remain covered; staging in those tests uses isolated fixture state.

The broad relevant suite passed **198 tests, with one existing skip**. After the final ACTIVE badge/label adjustment and its additional regression, the affected UI/control/catalog/identity subset passed **142 tests**. `git diff --check` passed for the changed implementation, templates, and tests.

An additional Survey Summary probe test fails on Windows URI handling of an extended-path candidate (`local_raw_present=False`). The same failure was reproduced after loading `survey_summary.py` directly from HEAD, so it predates this pass. The other eleven Survey Summary tests pass with extended Windows test paths. No unrelated URI refactor was made.

The following protected files are byte-for-byte unchanged from the completed Ownership audit:

- ACTIVE registry SHA: `749a7de37ee37733e0fc3ea84fa2034c509a819f254ad814b7d08def7d208408`
- Candidates registry SHA: `e7d31bd6de27cd8740e16829acb5bc1c16f05a46b9de49833b9c6538add4dc07`
- Existing Ownership website stage SHA: `c5f33aea06c5f9e755a28aa2237e2d4755ad1eee4af665273c9a4271143f257f`
- Live release-check state SHA: `76c61390649bf95b124a6f7b8f58f1942133fc01650fe94407c2ea12207154f5`; all rows equal the prior audit's reconstructed completed state.

No live content-currentness verdict was refreshed in this generalization audit. The earlier Ownership byte proof remains intact; catalog metadata observations alone do not extend it to other datasets.
