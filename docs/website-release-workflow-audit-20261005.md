# Website-release workflow audit — October 5, 2026

Scope: PECOS ownership and SFF website readiness, verification, Sources actions, and review UI. No source acquisition, activation, restaging, publication, merge, push, or deployment was performed. Bounded CMS byte observations wrote only two new verification evidence records.

## Audit and shared model

Previously, `source_operator_guidance.public_update_guidance` and `website_release_review.website_review_context` separately computed readiness. Ownership had a special verification gate; SFF did not. A generic source-details link stood in for a source-verification action. Stage-layer flags could be displayed as publication evidence, even though a stage is not a receipt.

`website_release_readiness.py` now owns the shared result consumed by cards, review, the Sources CMS-result column, and source-detail currentness. `WebsiteFamily.required_source_evidence` declares the requirements; it does not impose a two-source rule on SFF.

| Layer | Evidence required | Operator display |
|---|---|---|
| Local selection | Governed ACTIVE record; aligned local release IDs for the pair | Selected locally / ACTIVE |
| Website candidate | Recorded STAGED manifest, correct family/release, ACTIVE input fingerprints, existing contract eligibility, exact cached commit-artifact SHA matches | STAGED; readiness evaluated separately |
| Commit | Receipt bound to this manifest SHA; publication commit SHA and commit evidence | Committed locally, not pushed |
| Push | Matching receipt, commit evidence, successful push and timestamp | Pushed; deployment not verified |
| Deployment | Matching receipt plus successful deployment observation for its commit, or actual candidate artifact provenance verification | Deployment observed; provenance not verified / Live deployment verified |
| Production provenance | Matching receipt; successful timestamped production-origin checks whose expected and actual hashes cover the proposed candidate artifact hashes | Live deployment verified |

Stage `canonical: CURRENT`, `committed`, `pushed`, `deployed`, and `production_verified` flags are retained in the complete-manifest disclosure but do not establish these publication layers. A receipt with a different manifest fingerprint, or contradictory recorded source hashes, is explicitly identified as belonging to a different candidate. Deployment is never inferred from stage, commit, or push.

Source verification distinguishes **Byte verified current**, **Metadata checked; bytes not verified**, **Verification required**, **Publisher bytes differ**, and **Verification failed/unavailable**. A metadata observation cannot satisfy the byte gate. Publication blockers remain independent of source currentness: verified sources do not override failed validation or changed candidate artifact bytes.

## Verification action

`POST /actions/website-releases/<family>/<release_id>/verify-sources` runs the declared observations, records a complete attempt bound to the exact staged manifest SHA, and re-evaluates readiness. Both Ownership members must match the exact staged/ACTIVE source hashes; an unavailable member or same-vintage revision fails closed.

CSV observations use existing `generic_cms_csv.assess_feed` and `cms_release_identity.assess_raw_identity`, **not** `run_feed`, which can record lifecycle candidates. Current-product and current-version discovery must agree, and there must be one unambiguous primary resource. Only official HTTPS CMS URLs are accepted. Observations have byte/time bounds and do not save downloaded raw artifacts.

SFF uses its own adapter: resolve a unique current PDF link from CMS's official Nursing Homes page, parse its Updated release label, compare its exact PDF hash, and use existing immutable raw-PDF provenance bound to the staged normalized ACTIVE source. Monthly URL guessing does not establish website currentness. Ambiguous links or missing release labels block verification. SFF's old configured landing URL returned 404; the source definition now points to the authoritative CMS page containing the posting download.

Durable records contain source ID, publisher resource/version/release fields where supplied, observation timestamp, observed SHA, staged/ACTIVE/raw SHA, comparison result, and candidate manifest SHA. Check timestamps are not acquisition timestamps. Stage manifests and historical source metadata were not rewritten.

## Live before/after

| Family | Before | After bounded official observation | Publication evidence |
|---|---|---|---|
| PECOS ownership | Metadata checked; 0/2 byte verified | 2/2 byte verified current; validation passed; candidate ready; Review website release | Current candidate not published; older pushed receipt belongs to another candidate |
| SFF | Card could call the candidate prepared without a byte gate | 1/1 byte verified current; validation passed; candidate ready; Review website release | No publication receipt for this candidate; not published |

Current PECOS observations:

| Source | CMS version UUID | CMS version label | CMS modified metadata | File/snapshot date | Current resource file UUID | Local = remote SHA256 |
|---|---|---|---|---|---|---|
| Owners | `48ca01a7-7176-4882-9273-fef560dc763c` | `2026-08-02` | `2026-09-29` | `2026-07-31` | `93535382-e382-4338-9307-a18ce27d041f` | `dc2d46642b81e5dd78998bf03519a39cbbdce932ef6433ef03f0af0188f5ba7f` |
| Enrollments | `d756f18b-96d9-4abf-938c-4940c82e83a5` | `2026-08-01` | `2026-08-17` | `2026-07-31` | `3d90b1ba-ee57-400d-9cfa-85463b450313` | `387e0caf0f2e017c5c8bddfda4b4351182e85ea7fc355b8a003f2f06d2ff2527` |

Both vintages are August 2026. The full version labels and modified dates above are fresh CMS observations, not revisions to historical stage metadata.

Current Owners URL: https://data.cms.gov/sites/default/files/2026-09/fa718478-b275-41b4-83bf-d69ed9adc428/SNF_All_Owners_2026.07.31_update.csv

Current Enrollments URL: https://data.cms.gov/sites/default/files/2026-08/76343e4f-f990-41fc-90d9-33a3284ad997/SNF_Enrollments_2026.07.31.csv

Current SFF PDF: https://www.cms.gov/files/document/sff-posting-candidate-list-september-2026.pdf, linked by https://www.cms.gov/medicare/health-safety-standards/certification-compliance/nursing-homes

SFF's raw local/current PDF SHA is `a590a2266648c7d077c61965a2d06ad4beb886653de587ef8707bd1726a441a2`; the bound staged normalized ACTIVE source SHA is `5cb35b1ab07c58c1d13150f8693829cc4df23985e438b132a54b69d9c3ac3de1`. Its release label is September 2026. A separate snapshot date is not supplied by this observation and is not invented.

Evidence files: `state/website_source_evidence/cms.snf_ownership_pair/2026-07-31.json` and `state/website_source_evidence/cms.sff_pdf_list/2026-09.json`.

Ownership stage SHA stays `c5f33aea06c5f9e755a28aa2237e2d4755ad1eee4af665273c9a4271143f257f`. The old receipt references `25c07646f7ffec97247f8bccb7ef06696cc0f10aeb9baf92c893ccc208c149cf`, so its pushed flag does not describe this candidate. SFF stage SHA stays `ff6674a795a9278de11a348b9429e3a8dd5c9b76fd5969e1de3e77ec56ecd998`.

## UI and acceptance

Cards have one obvious state-driven primary action: Verify CMS source bytes, Review website release, Verify publication, or no urgent CTA. Source details is secondary; inspection of blocked release evidence is inside a disclosure. The review uses a wider desktop modal, a hash-free summary, expanded blockers/destinations/gates, and collapsed source/provenance forensics. Existing modal close, Escape, and focus restoration are preserved. Long evidence safely wraps on mobile.

Chromium acceptance cases verified with isolated state:

1. PECOS before verification: 0/2, verification CTA, blocked review.
2. Successful Owners + Enrollments verification via the actual POST route: 2/2, ready review.
3. One member unverifiable: 1/2, blocked.
4. Same-vintage changed bytes: revision surfaced, blocked.
5. Old mismatched receipt: explicit different-candidate notice; not published.
6. Staged without commit: not published.
7. Committed without push: committed locally, not pushed.
8. Pushed without deployment proof: deployment not verified; Verify publication CTA.
9. Matching production artifact proof: live verified; no urgent CTA.
10. SFF staged/ready: its own 1/1 evidence requirement.
11. SFF failed validation: publication blocked despite 1/1 verified source.
12. Desktop Sources: compact balanced cards and consistent table statuses.
13. Desktop review: wider than 1000px at a 1440px viewport; progressive disclosure.
14. Mobile Sources/review: responsive at 390px; no review horizontal overflow.

There were no browser errors. Browser mutations were limited to one verification POST against isolated fixture state. Live read-only Sources/review checks are recorded separately in `.tmp/website-workflow/`.

## Tests, shared consumers, and limits

The broader regression suite passed 140 tests; final affected UI/readiness coverage passed 49 tests. Coverage includes candidate hash/cache tampering, pair failures/revisions, metadata-only and non-CMS evidence rejection, publication state transitions, stale receipt rejection, source-detail/card consistency, read-only GET behavior, and source-specific SFF raw-PDF identity. The Provider Information stale-stage test now explicitly covers both its existing byte-verification precondition and the restage action; production PI lifecycle code was not changed.

The new readiness engine has two concrete registered families: Ownership and SFF. Provider Information retains its existing independent publication routes and contract; PBJ nurse/MACPAC stage adapters are not added to Sources website cards. Survey Summary remains review-only. Common stage-contract evaluation is reused without modifying the other families' lifecycle rules.

All 16 existing protected state files (ACTIVE, lifecycle candidates/checks, staged manifests, publication receipts and other top-level state) remained byte-identical. Only the two new live source-verification records were written. Current raw/source files were rechecked against the preserved ACTIVE hashes.

The actual website deployment and live provenance remain unobserved for these candidates. A stage or old receipt cannot establish them. Ownership/SFF have no executable publication or production-verification handler in Data Ops; the review states this explicitly and provides a gated governed-workflow handoff. No publication or deployment state was manufactured.

## Files changed in this task

- `website_release_readiness.py`: shared family registry, readiness, publication evidence, bounded observers, durable verification.
- `website_release_review.py`: shared context plus progressive-disclosure presentation.
- `source_operator_guidance.py`: delegate website readiness to the shared model; preserve Survey guidance.
- `data_ops_app.py`: verification POST, shared Sources and source-detail status integration.
- `cms_source_registry.py`: authoritative SFF landing page.
- `templates/data_ops/sources.html`, `partials/website_release_cards.html`, `partials/website_release_review.html`, `partials/source_detail_panel.html`: shared actions/statuses and review UX.
- `static/data_ops/data_ops.js`, `static/data_ops/sources.css`: website-only wider modal and responsive presentation.
- `tests/test_website_release_readiness.py`, `tests/test_website_release_review.py`, `tests/test_website_source_detail_consistency.py`, `tests/test_source_operator_guidance.py`, `tests/test_survey_summary_ui.py`, `tests/test_pi_stage_manifest_stale.py`: focused and shared-consumer regression coverage.
- This report and the two source-verification JSON records listed above. Browser fixtures/screenshots/logs remain under ignored `.tmp/website-workflow/`.
