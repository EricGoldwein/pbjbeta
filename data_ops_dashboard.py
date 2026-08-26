"""Subscription Dashboard Builder service (premium/dynamic facility dashboards).

Does NOT deploy. Does NOT write to pbj-root.
Maps UI actions to proven scripts; refuses unsafe paths.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import cms_data_paths  # noqa: E402
from data_ops_approval import has_acknowledgement, has_approval  # noqa: E402
from data_ops_zweli import ZweliState  # noqa: E402


class BundleState(str, Enum):
    NOT_GENERATED = "NOT_GENERATED"
    GENERATED_LOCALLY = "GENERATED_LOCALLY"
    PREFLIGHT_PASSED = "PREFLIGHT_PASSED"
    DEPLOYED = "DEPLOYED"
    DEPLOYED_BUT_STALE = "DEPLOYED_BUT_STALE"
    UNKNOWN = "UNKNOWN"


class DashboardActionBlocker(str, Enum):
    NONE = "NONE"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    UNPROCESSED = "UNPROCESSED"
    STRUCTURAL_BLOCKED = "STRUCTURAL_BLOCKED"
    ZWELI_BLOCKED = "ZWELI_BLOCKED"
    ZWELI_REQUIRES_ACK = "ZWELI_REQUIRES_ACK"
    NO_V2_REFERENCE = "NO_V2_REFERENCE"
    UNSAFE_PACKAGE_PATH = "UNSAFE_PACKAGE_PATH"
    MISSING_DEPLOY_DIR = "MISSING_DEPLOY_DIR"


@dataclass
class FacilityDashboardStatus:
    ccn: str
    facility_name: Optional[str]
    bundle_state: str
    deploy_dir: Optional[str]
    is_v2: bool
    has_v2_reference: bool
    reference_ccn: Optional[str]
    readiness: dict[str, Any] = field(default_factory=dict)
    preflight: dict[str, Any] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    can_generate_refresh: bool = False
    can_run_preflight: bool = False
    can_deploy: bool = False  # always False in V0 UI
    detail: str = ""
    source_gates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_ccn(ccn: str) -> str:
    s = str(ccn).strip().upper()
    if s.isdigit():
        s = s.zfill(6)
    return s


def _detect_v2(deploy_dir: Path, ccn: str) -> bool:
    return (deploy_dir / f"facility_{ccn}_superdynamic_dashboard.py").is_file()


def _find_v2_reference(root: Path, preferred: tuple[str, ...] = ("315461", "315128", "335513")) -> Optional[str]:
    for ref in preferred:
        d = cms_data_paths.facility_deploy_dir(ref, root)
        if _detect_v2(d, ref):
            return ref
    # any local V2
    dep = cms_data_paths.deployments_dir(root)
    if not dep.is_dir():
        return None
    for child in sorted(dep.glob("pbj320-*")):
        ccn = child.name.replace("pbj320-", "")
        if _detect_v2(child, ccn):
            return ccn
    return None


def _facility_name_from_bundle(deploy_dir: Path, ccn: str) -> Optional[str]:
    for pattern in (
        f"facility_{ccn}_provider_info_data.csv",
        f"facility_{ccn}_complete_data.csv",
    ):
        p = deploy_dir / pattern
        if not p.is_file():
            continue
        try:
            import csv

            with p.open("r", encoding="utf-8-sig", newline="") as f:
                row = next(csv.DictReader(f), None)
            if not row:
                continue
            for k in ("Provider Name", "PROVNAME", "facility_name", "name"):
                if k in row and str(row[k]).strip():
                    return str(row[k]).strip()
        except Exception:  # noqa: BLE001
            continue
    return None


# Source-data blockers that must disable Generate/Refresh (fail closed).
SOURCE_DATA_REFRESH_BLOCKERS: frozenset[str] = frozenset(
    {
        DashboardActionBlocker.SOURCE_UNAVAILABLE.value,
        DashboardActionBlocker.UNPROCESSED.value,
        DashboardActionBlocker.STRUCTURAL_BLOCKED.value,
        DashboardActionBlocker.ZWELI_BLOCKED.value,
        DashboardActionBlocker.ZWELI_REQUIRES_ACK.value,
    }
)


def evaluate_source_gates_for_dashboard(
    *,
    zweli_state: ZweliState | str | None = None,
    zweli_ack: bool = False,
    structural_ok: bool = True,
    provider_processed: bool = True,
    provider_available: bool = True,
) -> list[str]:
    blockers: list[str] = []
    if not provider_available:
        blockers.append(DashboardActionBlocker.SOURCE_UNAVAILABLE.value)
    if not provider_processed:
        blockers.append(DashboardActionBlocker.UNPROCESSED.value)
    if not structural_ok:
        blockers.append(DashboardActionBlocker.STRUCTURAL_BLOCKED.value)
    state = None
    if zweli_state is not None:
        state = ZweliState(zweli_state) if not isinstance(zweli_state, ZweliState) else zweli_state
    if state == ZweliState.BLOCKED:
        blockers.append(DashboardActionBlocker.ZWELI_BLOCKED.value)
    if state == ZweliState.REQUIRES_REVIEW and not zweli_ack:
        blockers.append(DashboardActionBlocker.ZWELI_REQUIRES_ACK.value)
    return blockers


def check_facility_readiness(ccn: str, *, root: Path | None = None) -> dict[str, Any]:
    """Delegate to scripts/check_facility_cms_data_ready.py (read-only)."""
    root = root or cms_data_paths.repo_root()
    ccn = _normalize_ccn(ccn)
    script = root / "scripts" / "check_facility_cms_data_ready.py"
    if not script.is_file():
        return {"ok": False, "error": "check_facility_cms_data_ready.py missing", "ccn": ccn}
    proc = subprocess.run(
        [sys.executable, str(script), ccn],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-2000:],
        "ccn": ccn,
    }


def run_preflight(ccn: str, *, root: Path | None = None) -> dict[str, Any]:
    """Run preflight_v2_facility_deploy.py — validates only, does not deploy."""
    root = root or cms_data_paths.repo_root()
    ccn = _normalize_ccn(ccn)
    script = root / "scripts" / "preflight_v2_facility_deploy.py"
    if not script.is_file():
        return {"ok": False, "error": "preflight_v2_facility_deploy.py missing", "ccn": ccn}
    proc = subprocess.run(
        [sys.executable, str(script), ccn],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-6000:],
        "stderr": proc.stderr[-2000:],
        "ccn": ccn,
    }


def refresh_existing_v2_runtime(ccn: str, *, root: Path | None = None, ref_ccn: str | None = None) -> dict[str, Any]:
    """Proven-ish refresh: bootstrap_superdynamic_v2_facility.run_bootstrap only.

    Does NOT call create_vercel_deployment.py (V1 regression risk on main).
    Does NOT deploy.
    """
    root = root or cms_data_paths.repo_root()
    ccn = _normalize_ccn(ccn)
    deploy_dir = cms_data_paths.facility_deploy_dir(ccn, root)
    if not deploy_dir.is_dir():
        return {
            "ok": False,
            "blocker": DashboardActionBlocker.MISSING_DEPLOY_DIR.value,
            "detail": f"Missing {deploy_dir} — cold generation not proven without reference",
        }
    if not _detect_v2(deploy_dir, ccn):
        return {
            "ok": False,
            "blocker": DashboardActionBlocker.UNSAFE_PACKAGE_PATH.value,
            "detail": (
                "Target is not a V2 superdynamic bundle. Refusing create_vercel_deployment.py "
                "on main (legacy entrypoint / V1 regression risk)."
            ),
        }
    ref = ref_ccn or _find_v2_reference(root)
    if not ref:
        return {
            "ok": False,
            "blocker": DashboardActionBlocker.NO_V2_REFERENCE.value,
            "detail": (
                "CANNOT GENERATE SAFELY — V2 reference bundle unavailable in this runtime"
            ),
        }
    # Import bootstrap
    sys.path.insert(0, str(root / "scripts"))
    try:
        import bootstrap_superdynamic_v2_facility as boot  # type: ignore
    except ImportError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        result = boot.run_bootstrap(root, ccn, ref)
        return {"ok": True, "result": result if not isinstance(result, Path) else str(result), "ref_ccn": ref}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "ref_ccn": ref}


def facility_dashboard_status(
    ccn: str,
    *,
    root: Path | None = None,
    zweli_state: ZweliState | str | None = None,
    provider_release_id: str | None = None,
    structural_ok: bool = True,
    provider_processed: bool = True,
    provider_available: bool = True,
    run_readiness: bool = True,
) -> FacilityDashboardStatus:
    root = root or cms_data_paths.repo_root()
    ccn = _normalize_ccn(ccn)
    deploy_dir = cms_data_paths.facility_deploy_dir(ccn, root)
    exists = deploy_dir.is_dir()
    is_v2 = _detect_v2(deploy_dir, ccn) if exists else False
    ref = _find_v2_reference(root)
    name = _facility_name_from_bundle(deploy_dir, ccn) if exists else None

    zweli_ack = False
    if provider_release_id and zweli_state:
        zweli_ack = has_acknowledgement("cms.provider_info", provider_release_id)

    blockers = evaluate_source_gates_for_dashboard(
        zweli_state=zweli_state,
        zweli_ack=zweli_ack,
        structural_ok=structural_ok,
        provider_processed=provider_processed,
        provider_available=provider_available,
    )
    if not exists:
        blockers.append(DashboardActionBlocker.MISSING_DEPLOY_DIR.value)
    if exists and not is_v2:
        blockers.append(DashboardActionBlocker.UNSAFE_PACKAGE_PATH.value)
    if not ref:
        blockers.append(DashboardActionBlocker.NO_V2_REFERENCE.value)

    if not exists:
        bundle_state = BundleState.NOT_GENERATED.value
        detail = (
            "Cold/new V2 generation is not proven on a clean checkout without a local "
            "V2 reference bundle. Refusing unsafe create_vercel_deployment.py on main."
        )
    elif is_v2:
        bundle_state = BundleState.GENERATED_LOCALLY.value
        detail = "Existing V2 bundle present locally. Safe refresh = bootstrap --ref; never --package on main."
    else:
        bundle_state = BundleState.GENERATED_LOCALLY.value
        detail = "Deploy dir exists but is not V2 superdynamic — treat as unsafe for V2 generate."

    readiness: dict[str, Any] = {}
    if run_readiness and exists:
        readiness = check_facility_readiness(ccn, root=root)

    source_data_blocked = any(b in SOURCE_DATA_REFRESH_BLOCKERS for b in blockers)
    can_refresh = (
        exists
        and is_v2
        and ref is not None
        and not source_data_blocked
    )
    can_preflight = exists and is_v2

    return FacilityDashboardStatus(
        ccn=ccn,
        facility_name=name,
        bundle_state=bundle_state,
        deploy_dir=str(deploy_dir) if exists else None,
        is_v2=is_v2,
        has_v2_reference=ref is not None,
        reference_ccn=ref,
        readiness=readiness,
        blockers=blockers,
        can_generate_refresh=can_refresh,
        can_run_preflight=can_preflight,
        can_deploy=False,
        detail=detail,
        source_gates={
            "zweli_state": str(zweli_state) if zweli_state else None,
            "zweli_ack": zweli_ack,
            "approved": bool(
                provider_release_id and has_approval("cms.provider_info", provider_release_id)
            ),
            "provider_release_id": provider_release_id,
        },
    )


# Proven paths summary for reports / UI copy
EXISTING_V2_REFRESH_PATH = (
    "bootstrap_superdynamic_v2_facility.run_bootstrap(root, ccn, ref_ccn) "
    "then preflight_v2_facility_deploy.py. Deploy separately with "
    "deploy_vercel_facility.py --confirm-deploy --no-package (NOT exposed in Data Ops V0)."
)
COLD_NEW_V2_PATH = (
    "UNRESOLVED on clean main: requires local V2 reference bundle "
    "(default 315461 often uncommitted). create_vercel_deployment.py on main "
    "writes legacy V1 flask entrypoint — unsafe for V2."
)
