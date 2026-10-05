# Ownership CMS release identity audit — October 5, 2026

## Conclusion

Both ACTIVE raw artifacts match the exact current August 2026 CMS distributions byte-for-byte. No acquisition, activation, website staging, or publication was performed. ACTIVE and candidate records remain byte-for-byte unchanged. Only the two ownership rows in `state/release_checks.json` were refreshed.

CMS publication vintage is August 2026. The July 31, 2026 date is embedded in the official CSV filenames and remains the legacy governed pair/snapshot key; it is not the CMS release vintage. The version-date field is also distinct from the dataset-last-updated date and from acquisition time. Existing ACTIVE acquisition timestamps are absent and were not invented.

## Live metadata and raw-byte proof

### cms.snf_all_owners

- Official current resource: [SNF_All_Owners_2026.07.31_update.csv](https://data.cms.gov/sites/default/files/2026-09/fa718478-b275-41b4-83bf-d69ed9adc428/SNF_All_Owners_2026.07.31_update.csv)
- Official resource title: `SNF All Owners Aug 2026`
- Stable product UUID: `afe44b85-cc6d-40d7-b5df-00ae8910d1d2`
- Current dataset-version UUID: `48ca01a7-7176-4882-9273-fef560dc763c`
- CMS version date: `2026-08-02`
- CMS dataset last updated: `2026-09-29`
- CMS file UUID: `93535382-e382-4338-9307-a18ce27d041f`
- File size: `54340025` bytes
- HTTP Last-Modified: `Tue, 29 Sep 2026 13:14:17 GMT`
- Local immutable raw file: `ownership\_sources\cms_snf_all_owners\raw\downloaded\SNF_All_Owners_2026.07.31_update.csv`
- Local file SHA-256: `dc2d46642b81e5dd78998bf03519a39cbbdce932ef6433ef03f0af0188f5ba7f`
- ACTIVE registry SHA-256: `dc2d46642b81e5dd78998bf03519a39cbbdce932ef6433ef03f0af0188f5ba7f`
- Current CMS raw-resource SHA-256: `dc2d46642b81e5dd78998bf03519a39cbbdce932ef6433ef03f0af0188f5ba7f`
- Discovery: [stable CMS product](https://data.cms.gov/data-api/v1/slug?path=%2Fprovider-characteristics%2Fhospitals-and-other-facilities%2Fskilled-nursing-facility-all-owners) → [version metadata](https://data.cms.gov/jsonapi/node/dataset?fields%5Bnode--dataset%5D=field_dataset_version,field_last_updated_date,field_re_release_select,field_re_release_version&filter%5Bfield_dataset_type.name%5D=Skilled%20Nursing%20Facility%20All%20Owners&sort=-field_dataset_version,-field_re_release_version) → [resources](https://data.cms.gov/data-api/v1/dataset/48ca01a7-7176-4882-9273-fef560dc763c/resources)

### cms.snf_enrollments

- Official current resource: [SNF_Enrollments_2026.07.31.csv](https://data.cms.gov/sites/default/files/2026-08/76343e4f-f990-41fc-90d9-33a3284ad997/SNF_Enrollments_2026.07.31.csv)
- Official resource title: `SNF Enrollments Aug 2026`
- Stable product UUID: `5f2c306f-3b1c-42cd-b037-187b2ce22126`
- Current dataset-version UUID: `d756f18b-96d9-4abf-938c-4940c82e83a5`
- CMS version date: `2026-08-01`
- CMS dataset last updated: `2026-08-17`
- CMS file UUID: `3d90b1ba-ee57-400d-9cfa-85463b450313`
- File size: `3891152` bytes
- HTTP Last-Modified: `Thu, 13 Aug 2026 14:09:57 GMT`
- Local immutable raw file: `ownership\_sources\cms_snf_enrollments\raw\downloaded\SNF_Enrollments_2026.07.31.csv`
- Local file SHA-256: `387e0caf0f2e017c5c8bddfda4b4351182e85ea7fc355b8a003f2f06d2ff2527`
- ACTIVE registry SHA-256: `387e0caf0f2e017c5c8bddfda4b4351182e85ea7fc355b8a003f2f06d2ff2527`
- Current CMS raw-resource SHA-256: `387e0caf0f2e017c5c8bddfda4b4351182e85ea7fc355b8a003f2f06d2ff2527`
- Discovery: [stable CMS product](https://data.cms.gov/data-api/v1/slug?path=%2Fprovider-characteristics%2Fhospitals-and-other-facilities%2Fskilled-nursing-facility-enrollments) → [version metadata](https://data.cms.gov/jsonapi/node/dataset?fields%5Bnode--dataset%5D=field_dataset_version,field_last_updated_date,field_re_release_select,field_re_release_version&filter%5Bfield_dataset_type.name%5D=Skilled%20Nursing%20Facility%20Enrollments&sort=-field_dataset_version,-field_re_release_version) → [resources](https://data.cms.gov/data-api/v1/dataset/d756f18b-96d9-4abf-938c-4940c82e83a5/resources)

The current Owners node now has version date `2026-08-02`, updated `2026-09-29`, and serves the `_update.csv`. The August 1 / August 17 observation in the request is superseded for Owners by this revision. Enrollments still has version date `2026-08-01`, updated `2026-08-17`. Both resource titles explicitly identify Aug 2026.

## Exact code path and root cause

1. Both Sources Check CMS forms POST to `action_control_panel_check_releases` in `data_ops_app.py`.
2. The route calls `release_check.check_releases(acquire=False, external_handlers=production_handlers())`.
3. `production_handlers` builds `ownership_csv_feeds`; each calls `generic_cms_csv.run_feed`.
4. `detect` resolves the stable product slug and its current dataset UUID, cross-checks sorted official version metadata, then loads the current node resources.
5. `_release_id` takes July 31 from the filename. This is the existing pair/snapshot key, but it was also incorrectly exposed as the publisher release/latest date.
6. `_assess_found` formerly returned CURRENT when the pair key and known resource identity fields matched, without fetching and hashing current resource bytes. Same-URL/UUID in-place revisions could therefore be missed.
7. `build_release_availability_context` also cleared `new_release_available` whenever IDs matched; `operator_freshness` rendered the snapshot as CMS latest and unchanged.

This was case A: current August resources were observed, but their filename dates were mislabeled. Resource discovery did not miss August. Byte verification has now independently established that both local raw files are current.

## Corrections

- Separate `cms_release_vintage`, `snapshot_date`, exact CMS version/resource identity, raw SHA, byte-check time, and acquisition time.
- For both ownership feeds, CURRENT requires matching remote raw SHA and on-disk ACTIVE SHA against the governed registry hash. Fetch failure and local tampering never imply CURRENT.
- Different bytes at the same filename/snapshot/resource identity are REVISED; a newer publication vintage or snapshot is NEWER. Detection remains separate from acquisition and activation.
- Same-filename acquisition retains a new immutable hash-suffixed artifact instead of replacing ACTIVE. Validation rejects a resource that changes between detection and acquisition.
- Matching bytes do not produce a candidate or reactivate data merely because publisher identity/metadata changed.
- Pair keys remain unchanged. New ownership website stages require verified observations for both members, aligned CMS vintages, and carry each member’s CMS vintage, filename date, resource identity, exact SHA and separate timestamps.

## Existing website candidate

The existing ownership stage has matching source hashes but lacks these explicit vintage/snapshot provenance fields. It is shown as PROVENANCE REVIEW REQUIRED rather than release-ready. Its manifest and publication receipt were deliberately preserved: a pre-existing receipt records committed/pushed YES, deployed UNKNOWN and production_verified NO. This audit made no website release or verification claim.

Existing stage SHA-256: `c5f33aea06c5f9e755a28aa2237e2d4755ad1eee4af665273c9a4271143f257f`
The receipt fingerprint differs from this existing stage, so it does not prove this candidate was published.

Existing receipt stage SHA-256: `25c07646f7ffec97247f8bccb7ef06696cc0f10aeb9baf92c893ccc208c149cf`

## Validation

Focused ownership detection, byte identity, immutable revision acquisition, pair validation/promotion (isolated fixtures only), display, provenance, stage and regression tests: 67 passed, 1 existing skip.
Browser: real current Sources card and both detail pages render August CMS vintage, July 31 file date, exact hash proof and separate check/acquisition timestamps. The Sources website card correctly requires provenance review.

Protected live files, unchanged during observation refresh:

- `C:\Users\egold\PycharmProjects\pbj-data-ops\state\active_releases.json`: `749a7de37ee37733e0fc3ea84fa2034c509a819f254ad814b7d08def7d208408`
- `C:\Users\egold\PycharmProjects\pbj-data-ops\state\release_candidates.json`: `e7d31bd6de27cd8740e16829acb5bc1c16f05a46b9de49833b9c6538add4dc07`
- `C:\Users\egold\PycharmProjects\pbj-data-ops\state\pbj320_stages\cms.snf_ownership_pair\2026-07-31.json`: `c5f33aea06c5f9e755a28aa2237e2d4755ad1eee4af665273c9a4271143f257f`

## Files changed for this audit

Detection / acquisition (ownership-specific behavior):
- `generic_cms_csv.py`
- `release_check.py`
- `ownership_pairing.py`

Read-only display and website provenance gates:
- `cms_data_ops.py`
- `operator_freshness.py`
- `source_operator_guidance.py`
- `pbj320_stage_ownership.py`
- `templates/data_ops/sources.html`
- `templates/data_ops/partials/source_detail_panel.html`
- `templates/data_ops/partials/ownership_pair_panel.html`

Focused tests:
- `tests/test_ownership_release_semantics.py` (new)
- `tests/test_release_check.py`
- `tests/test_pbj320_stage_adapters.py`

Audit documentation / local observation cache:
- `docs/ownership-cms-release-identity-audit-20261005.md` (this report)
- `state/release_checks.json` (only Owners and Enrollments observations)

All other working-tree changes predate this audit and were preserved. No commit, push, activation, staging run, or publication occurred.
