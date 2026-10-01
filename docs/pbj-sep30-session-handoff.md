# PBJ September 30 session evidence and remaining work

Session stopped at the user's request to wrap up. Phase 1 completed; phase 2 failed its ownership acceptance gate. Phases 3–6 were not completed. This document records evidence, not a claim that the original task is finished.

## Repositories and checkpoints

- PBJapp: `C:\Users\egold\PycharmProjects\PBJapp`, branch `feat/conferences-evidence-layer-20260930`; starting SHA `247c59221efeaaaf7f1c727b7999042a318cadeb`; ending SHA `4aa868ded783132803a74e800c8a1c6868c3a517`.
- PBJapp commits: `4064442`, `f21628f`, `4730e05`, `23ceef6`, `bd9c6e6`, `4aa868d`. Normal upstream push was rejected as non-fast-forward. Fetch showed unrelated upstream commit `f89c5c4`; no merge, rebase, or force push performed. All session commits remain local.
- Data Ops: `C:\Users\egold\PycharmProjects\pbj-data-ops`, branch `feat/cms-nursing-home-catalog-20260930`; starting implementation SHA `4826e685ebd7bfab8dd02d8b63405fe3c6ff1ae9`. The only session checkpoint here is this document; resolve its ending SHA with `git log -1 --format=%H -- docs/pbj-sep30-session-handoff.md`.
- Both repositories had substantial preexisting dirty work. No broad staging, reset, stash, clean, PR, or merge was performed. PBJapp acceptance files with preexisting edits were staged by selected patch hunks. Generated packages remain unstaged. Full initial status: `D:\Temp\pbj-sep30-audit\PBJapp-status.txt` and `pbj-data-ops-status.txt`.

## Changed PBJapp files

`scripts/accept_existing_vercel_candidate.py`, `scripts/check_v2_owner_link_acceptance.py`, `scripts/check_v2_production_acceptance.py`, `scripts/check_v2_production_browser_acceptance.py`, `scripts/check_v2_vercel_auth_env.py`, `scripts/deploy_vercel_facility.py`, `scripts/pbj_premium_verify_auth.py`, `tests/test_deploy_roster_manifest_order.py`, `tests/test_package_finalization.py`, `tests/test_preview_protection_auth.py`, `tests/test_upload_asset_gates.py`.

## Package finalization result

The geo step overwrote `pbj_lite/facility_quarterly_metrics.parquet` and regenerated peer metadata after the old manifest generation point. Peer metadata's `generated_at` changed its hash. County/geo/EIN mutations now precede package checks and final manifest generation. Timing diagnostics now go outside package content.

Only 335581 was repackaged. Two non-repair validations and artifact-contract checks passed. The 301-file snapshots were byte-identical across validation; no changed files. Manifest SHA256: `ec86f96ea67e48e7653b48d9c8aa73b3305bf23831a68a728f9f7fba88e9774c`. Ownership provenance checks passed against the active release registry; 13 unique resolved associate IDs. This proves package validation idempotence, not byte-identical rebuilds.

Evidence directory: `D:\Temp\pbj-sep30-audit`. Files: `335581-before-validation.log`, `335581-idempotence.json`, `335581-validate_v2_deployment_package.py-1.log`, `335581-validate_v2_deployment_package.py-2.log`, `335581-check_v2_facility_artifact_contract.py-1.log`, `335581-check_v2_facility_artifact_contract.py-2.log`, `335581-routing-repackage.log`, `routing-idempotence.log`, `335581-preflight.log`.

Targeted tests: **38 passed**, recorded in `pbj-final-tests.log`. The final responsive-picker wait adjustment was subsequently exercised by live browser acceptance. The candidate-resumption script ran and stopped at the failing owner gate; its successful promotion path is unverified.

## Exact deployment state

- Failed first preview: `dpl_DZBqqeVSohMSdJCaY7CFQMA8UGeW`, `https://pbj320-335581-8dkfj7ct7-ericgoldweins-projects.vercel.app`. Standalone API base incorrectly included `/premium/335581`; browser requests received HTML 404s. Root API base fixed before final manifest regeneration.
- User authorized replacing that failed candidate. Corrected preview: `dpl_AvLteS6ozQ3rusVCGiJzBkTWZK9X`, `https://pbj320-335581-gzgm8uhpv-ericgoldweins-projects.vercel.app`.
- Corrected preview API, auth, and general desktop/mobile browser acceptance passed on the final resumption (`335581-resume.log`). Ownership acceptance failed: the modal displayed “Ownership and control disclosures could not be loaded” and produced no owner links. Its underlying cause remains unproven. Exact-owner hyperlink clicks and unresolved-owner plain-text browser behavior are therefore **not accepted**. API identity checks alone are insufficient.
- Accepted preview ID: **none**. Promoted deployment ID: **none**. No independent production build and no third preview were created.
- Final read-only production inspection confirmed unchanged deployment `dpl_hvfjca5tqNytcuwNDxAnwcCpX3LB`, READY, aliased to `https://pbj320-335581.vercel.app`. Evidence: `production-inspect-final.log`. Production acceptance of the session's change did not occur.
- Protection cookies were carried separately from application login cookies. Credentials and private cookie files are intentionally excluded from this handoff.

## Data Ops inventory audit: preliminary only

Read `source_family_inventory.py`, `cms_source_registry.py`, `cms_nh_catalog.py`, and `release_control_plane.py`. Registry-backed rows already derive some active/control/probe state dynamically, but implementation status uses static automation maturity. Penalties/HCRIS/NPI supplemental rows hardcode governance, implementation, evidence, health, and next-action claims. Those claims were not independently established in this session and were not changed.

Human names, publisher, conceptual family, expected cadence, and official dataset identity can legitimately be descriptive static metadata. Governance/lifecycle, releases, receipts, evidence availability, and health should be derived from canonical runtime state. The full field audit and implementation remain outstanding; no claim that the inventory is repaired.

## Survey Summary and bounded adjacent sources

Survey Summary (`tbry-pc2d`) was **not acquired or implemented**. Official current catalog/resource, release, schema, row counts, grain, and uniqueness were not reverified. The prior CCN + Inspection Cycle key finding is handoff context only; CCN + Survey Date must not be adopted without evidence. No release, immutable raw artifact, validation receipt, review-ready candidate, or ACTIVE promotion was created. Data Ops Sources UI was not browser-checked.

Penalties, HCRIS, and NPI/NPPES: current canonical status **not independently verified**. Smallest next steps: Penalties—compare its current retained release with canonical control state and latest validation receipt; HCRIS—inspect existing pilot artifacts and branch checkpoint before deciding one validation gap; NPI/NPPES—inventory existing acquisition artifacts/receipts before adding any pipeline. These are bounded verification steps, not additional pipeline projects.

## Real blockers and remaining work

1. Ownership modal fails on the exact corrected preview. Diagnose that failure before acceptance or promotion; reuse this candidate where its immutable contents suffice, otherwise explicitly record a replacement requirement.
2. PBJapp upstream diverged, blocking a normal push; user excluded merges. Keep local checkpoints intact.
3. Inventory refactor, governed Survey Summary slice, bounded adjacent-source verification, and Data Ops UI/regressions remain unfinished because the session was stopped. They are outstanding work, not proven external blockers.

No repository is described as clean. No new deployment is described as accepted or promoted. No Survey Summary source is described as ACTIVE.
