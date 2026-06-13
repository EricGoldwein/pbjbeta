# V2/V3 facility deployment runbook

Manual packaging and validation path for superdynamic V2 bundles on clean `main`. Deployment folders under `deployments/` are **generated artifacts** — do not commit them.

## Prerequisites

- Clean `main` worktree synced with `origin/main`
- Vercel CLI installed and logged in (for production deploy only)
- Local `deployments/pbj320-<CCN>/` folder with facility data slices (CSVs/parquet as needed)
- For cold or stale bundles: a known-good **local** V2 reference (default `315461`) available on disk — not in git

## Safe manual path

### 1. Start from clean main

```powershell
git checkout main
git pull origin main
```

### 2. Ensure local deployment folder exists

Use an existing local bundle or create/update facility data only. Do **not** stage or commit `deployments/`, generated CSVs, or provider data.

### 3. Bootstrap or refresh V2 infrastructure (when needed)

When the bundle is new, stale, or missing V2 runtime files:

```powershell
python scripts/bootstrap_superdynamic_v2_facility.py <CCN> --ref 315461
```

Use `--force` only when you intend to overwrite infrastructure (not facility data slices). The reference CCN must exist locally as a complete V2 bundle.

### 4. Run V2 preflight

```powershell
python scripts/preflight_v2_facility_deploy.py <CCN>
```

Optional: `--check-vercel-env` after `vercel link` to confirm `PBJ_DASHBOARD_PASSWORD`.

Preflight runs import, bootstrap, inline JS, evidence layout, and static JS bundle checks. Fix any reported gaps before deploy.

Standalone import check:

```powershell
python scripts/check_v2_deployment_import.py <CCN> --deploy-dir deployments/pbj320-<CCN>
```

### 5. Deploy (safe path)

**Preferred:** deploy without repackaging when the bundle already passed preflight:

```powershell
python scripts/deploy_vercel_facility.py <CCN> --confirm-deploy --no-package
```

**Avoid** `--package` on an existing V2 bundle until deploy-script V2 guards land — `create_vercel_deployment.py` still writes V1 `facility_*_flask_app.py` and can regress the superdynamic entrypoint.

## Current limitations

- **Cold V2 generation** is not fully solved on clean checkout: bootstrap requires a local reference bundle with `facility_*_superdynamic_dashboard.py` (default ref `315461` is not committed).
- **`--package` may regress V2 to V1** until `deploy_vercel_facility.py` gains V2-aware packaging guards.
- **Deployment folders are generated** — keep them local; never commit `deployments/`, facility CSVs, or provider data to the repo.

## Related scripts (on main)

| Script | Role |
|--------|------|
| `scripts/preflight_v2_facility_deploy.py` | Full pre-deploy gate |
| `scripts/check_v2_deployment_import.py` | Lambda-style import smoke |
| `scripts/check_v2_deployment_bootstrap.py` | Data bootstrap + GET / smoke |
| `scripts/bootstrap_superdynamic_v2_facility.py` | Copy V2 runtime from reference |
| `deployment_entrypoint_guard.py` | Detect/preserve V2 entrypoints |
