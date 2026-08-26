# CMS provider-information monthly release workflow (PBJapp)

PBJapp is the canonical home for CMS **monthly Nursing Home Provider Information** national releases. The public site repo (`pbj-root`) receives only a separately validated normalized export (`ProviderInfoNorm_*.csv`) via an explicit handoff — not raw monthly zips.

## Raw source model

| Layer | Location | Role |
|-------|----------|------|
| Yearly outer ZIP | `provider_info/nursing_homes_including_rehab_services_YYYY.zip` | Durable CMS source of record (gitignored) |
| Monthly inner ZIP | Member `nursing_homes_including_rehab_services_MM_YYYY.zip` inside the yearly archive | **Not** a separate download path |
| Release manifests | `provider_info/_manifests/YYYY-MM/` | Tracked provenance (`release_manifest.json`, `pbj_root_handoff.json`) |

Do not create a standalone monthly ZIP beside the yearly archive for convenience.

## Release command

```bash
# Preferred Pilot (Provider Information CSV via stable dataset 4pq5-n9py):
python scripts/manage_cms_sources.py provider acquire
# or:
python scripts/cms_provider_info_acquire.py

# Full monthly archive path (yearly nursing_homes_including_rehab_services_YYYY.zip):
python scripts/manage_cms_sources.py provider ingest-release --year 2026 --month 6
```

`provider acquire` resolves the current CMS distribution dynamically from metastore
`4pq5-n9py`, downloads `NH_ProviderInfo_*.csv` only when newer than local, validates,
runs existing `normalize_provider_info.py`, and writes
`provider_info/_manifests/YYYY-MM/{acquisition,pbj_root_handoff,release_diff}.json`.
It does **not** write to pbj-root or deploy.

The archive command:

1. Locates the inner monthly archive inside the yearly outer ZIP
2. Validates expected members exist
3. Extracts **active pipeline files only** (idempotent; refuses to overwrite non-identical files)
4. Writes/updates `provider_info/_manifests/YYYY-MM/release_manifest.json`
5. Runs `normalize_provider_info.py` unless `--no-normalize`

### Actively extracted (June 2026 example)

```text
provider_info/NH_ProviderInfo_Jun2026.csv
provider_info/NH_DataCollectionIntervals_Jun2026.csv
ownership/NH_Ownership_Jun2026.csv
Citations/NH_HealthCitations_Jun2026.csv
Citations/NH_CitationDescriptions_Jun2026.csv
```

### Retained-unmodeled members (not extracted to active pipeline paths)

QM, SNF QRP, VBP, surveys, penalties, fire-safety citations, state averages, cutpoints, and documentation files are **classified** in `release_manifest.json` → `source_members` with `ingestion_status: retained_unmodeled`. They do **not** block provider-info promotion when explicitly classified.

Raw copies land under `provider_info/_retained/YYYY-MM/` (CSV members at top level; PDF/readme under `documentation/`). Each member records `source_sha256`, schema or text fingerprint, row counts, and `retained_artifact` path in the manifest.

Promotion is blocked only when a **new unclassified** CSV appears (`unmapped_new_source`), an **ingested** source changes schema, or a prior **ingested** member disappears from the archive.

### Facility bundle lite/geo assets

Before v2 deploy packaging, run:

```bash
python scripts/ensure_facility_bundle_assets.py 315128 315461 335513
python scripts/check_facility_bundle_assets.py 315128 315461 335513
```

`deploy_vercel_facility.py --package` runs the check gate for the target CCN automatically.

## Processing month vs PBJ staffing quarter

CMS **provider processing month** (e.g. `2026-06`, processing date `2026-06-01`) is **not** a PBJ calendar quarter.

Staffing and case-mix intervals come from `NH_DataCollectionIntervals_*.csv`. For June 2026 processing month, the staffing-level window remains **2025 Q4** (`10/01/2025`–`12/31/2025`).

Mappings:

- `prov_info_quarter_map.py` — manual processing-month → PBJ quarter label
- `static/data/interval_quarter_mapping.json` — UI/API interval metadata

Never label June 2026 provider information as PBJ Q2 2026 staffing.

## Derived artifacts

| Artifact | Build command |
|----------|----------------|
| `provider_info_normalized/ProviderInfoNorm_YYYY_MM.csv` | Automatic via ingest-release normalize step |
| `provider_info_combined.csv` | `python scripts/build_provider_info_combined.py --output provider_info_combined.csv` |

Combined semantics: one row per `(processing_date, ccn)` from all `ProviderInfoNorm_*.csv` inputs (dedupe keeps last).

### June 2026 combined-table evidence (tracked in `release_manifest.json`)

| Metric | Value |
|--------|-------|
| Pre-June reconstruction (through 2026-05) | 1,412,822 rows — exact key match with prior combined artifact |
| June-inclusive rebuild | 1,427,517 rows |
| June-inclusive SHA-256 | `2e7567c56545831cc1e2e8f6a75e5547a7f7a7946699022101b319fbf8222428` |

Rebuild validation gates (fail on duplicate keys within a Norm file, missing required months, or column schema drift across Norm snapshots): `python scripts/build_provider_info_combined.py --verify-only --through 2026-05`

## National sources vs deployment bundles

Ingest-release validates **national sources** and local slice-generation logic. It does **not** refresh `deployments/pbj320-<CCN>/` bundles. Existing deployed facility sites continue to serve whatever slice generation last wrote until `create_vercel_deployment.py` (or equivalent) is run locally.

## Related ownership source families (separate CMS drops)

These are **not** inside the monthly provider-information archive:

| Family | Doc / path |
|--------|------------|
| CMS CHOW (change of ownership transactions) | `ownership/_sources/cms_chow/` — see README there |
| CMS chain performance measures | `ownership/_sources/cms_chain_performance/` — see README there |
| Monthly facility ownership contacts | `ownership/NH_Ownership_*.csv` (from provider-info zip) |
| CMS enrollment all-owners | `ownership/SNF_All_Owners_*.csv` (separate CMS dataset) |

**CHOW note:** The currently consumed PBJapp CHOW index was built in pbj-root from a legacy ZIP container whose SHA-256 differs from PBJapp’s canonical raw ZIP. However, both archives have identical member inventories and identical payload hashes for every member, including SNF_CHOW_2026.04.01.csv. The underlying CMS source content is therefore verified as equivalent, and the currently served source release is Q1 2026. The remaining issue is architectural rather than source freshness: PBJapp consumes a pbj-root-built derived `chow_index.json` through a fallback path, without a formal PBJapp-to-pbj-root source handoff and index-validation contract. See `ownership/_sources/cms_chow/README.md`.

## pbj-root handoff

```bash
python scripts/sync_to_pbj_root.py checklist --release-key 2026-05
python scripts/sync_to_pbj_root.py provider-release --release-key 2026-05 --force
```

**Contract:** `provider-release` copies **both** `ProviderInfoNorm_YYYY_MM.csv` (committed in pbj-root) and paired `NH_ProviderInfo_MonYYYY.csv` (local/gitignored). Norm-only sync is treated as a bug.

`pbj_root_handoff.json` records `provider_promotion.ready_for_pbj_commit` — must be `true` (Norm + NH in PBJapp) before handoff.

Gates in pbj-root (sync sets `PBJAPP_ROOT` automatically): backfill → validate → simulate → **`verify_provider_release_handoff.py`**

Derived (run by sync): `build_state_page_aggregates.py`, **`generate_search_index.py`**

Manual before pbj-root commit: rebuild `provider_info_combined_latest.csv`, `validate_release.py`

Full routing: `docs/PBJ_ROOT_HANDOFF.md`, `docs/PBJ_ROOT_DATA_LAYERS.md`
