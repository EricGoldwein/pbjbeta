# Canonical release check

Run from `pbj-data-ops`:

```powershell
python release_check.py check-releases
```

Use `--detect-only` to query without downloading and `--json` for machine output.
The command never promotes. It writes observations and candidates to the existing
control-plane state, leaving ACTIVE unchanged until the established explicit
promotion policy succeeds.

## Source classes

- External recurring: Provider Information, PBJ nurse, PBJ non-nurse, SNF All
  Owners, and SNF Enrollment.
- Provider-bundle derived: NH Health Citations and NH Ownership.
- ACTIVE-input derived: state/national/region/CMI benchmarks and peer distribution.
- Static configuration: region/state mapping.
- Manually versioned reference: MACPAC staffing standards.

Provider Information and nurse staffing use automatic structural validation. Raw
non-nurse staffing stops at ACQUIRED until normalization and full-series validation.
SNF All Owners and SNF Enrollment stop at ACQUIRED until ownership pairing and human review
pass. Derived outputs must record `metadata.upstream_releases`; a changed upstream
ACTIVE release then marks them STALE. MACPAC requires a reviewed publication/version
and is never auto-detected from local filename age.

Network and destination access are adapter boundaries. A scheduled runner later
needs credentials/network policy, durable registry/check-state storage, an object
writer replacing the local-file destination adapter, locking, and alerting. The
candidate/promotion contract does not change.
