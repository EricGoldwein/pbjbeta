"""Detect and preserve V2 superdynamic deployment entrypoints during data repackaging."""

from __future__ import annotations

import json
import os
from pathlib import Path

V2_TEMPLATE_ENV_VALUES = frozenset({"v2", "superdynamic_v2", "superdynamic_dashboard_v2"})
DEFAULT_V2_REFERENCE_CCN = "315461"


def v2_superdynamic_entrypoint_name(ccn: str) -> str:
    ccn = str(ccn).strip().zfill(6)
    return f"facility_{ccn}_superdynamic_dashboard.py"


def _read_vercel_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def vercel_config_is_v2(cfg: dict | None, ccn: str) -> bool:
    if not cfg:
        return False
    ccn = str(ccn).strip().zfill(6)
    env = cfg.get("env") or {}
    tpl = str(env.get("PBJ_SUPERDYNAMIC_TEMPLATE") or "").strip().lower()
    if tpl not in V2_TEMPLATE_ENV_VALUES:
        return False
    app = v2_superdynamic_entrypoint_name(ccn)
    builds = cfg.get("builds") or []
    return any(app in str(b.get("src") or "") for b in builds)


def detect_v2_deployment_bundle(deploy_dir: os.PathLike[str] | str, ccn: str) -> tuple[bool, list[str]]:
    """Return (is_v2_bundle, marker_messages) for an existing deployment folder."""
    deploy = Path(deploy_dir)
    ccn = str(ccn).strip().zfill(6)
    markers: list[str] = []

    entry = deploy / v2_superdynamic_entrypoint_name(ccn)
    if entry.is_file():
        markers.append(f"superdynamic entrypoint: {entry.name}")

    v2_tpl = deploy / "templates" / "superdynamic_dashboard_v2.html"
    if v2_tpl.is_file():
        markers.append("templates/superdynamic_dashboard_v2.html")

    v2_partials = deploy / "templates" / "partials" / "v2" / "guided_nav.html"
    if v2_partials.is_file():
        markers.append("templates/partials/v2/")

    cfg = _read_vercel_json(deploy / "vercel.json")
    if cfg:
        env = cfg.get("env") or {}
        tpl = str(env.get("PBJ_SUPERDYNAMIC_TEMPLATE") or "").strip().lower()
        if tpl in V2_TEMPLATE_ENV_VALUES:
            markers.append("vercel.json PBJ_SUPERDYNAMIC_TEMPLATE=v2")
        src = str((cfg.get("builds") or [{}])[0].get("src") or "")
        if "superdynamic_dashboard" in src:
            markers.append(f"vercel.json entrypoint {src}")

    is_v2 = bool(
        entry.is_file()
        or (v2_tpl.is_file() and v2_partials.is_file())
        or vercel_config_is_v2(cfg, ccn)
    )
    return is_v2, markers


def adapt_vercel_json_from_ref(ref_json: Path, dst_json: Path, ccn: str, ref_ccn: str) -> None:
    raw = ref_json.read_text(encoding="utf-8").replace(ref_ccn, ccn)
    dst_json.write_text(json.dumps(json.loads(raw), indent=2) + "\n", encoding="utf-8")


def patch_vercel_ein_env(vercel_path: Path, ein_mode: str, quarters_csv: str) -> None:
    cfg = _read_vercel_json(vercel_path) or {}
    env = cfg.setdefault("env", {})
    env["EIN_DASHBOARD_MODE"] = ein_mode
    env["EIN_SELECTED_QUARTERS"] = quarters_csv
    vercel_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def restore_v2_vercel_json_from_reference(
    deploy_dir: os.PathLike[str] | str,
    ccn: str,
    project_root: os.PathLike[str] | str,
    ref_ccn: str | None = None,
) -> tuple[bool, str]:
    """Rewrite vercel.json from a reference V2 bundle when legacy config regressed."""
    deploy = Path(deploy_dir)
    ccn = str(ccn).strip().zfill(6)
    ref_ccn = str(ref_ccn or os.getenv("PBJ_V2_REFERENCE_CCN") or DEFAULT_V2_REFERENCE_CCN).strip().zfill(6)
    root = Path(project_root)

    ref_vercel = root / "deployments" / f"pbj320-{ref_ccn}" / "vercel.json"
    if not ref_vercel.is_file():
        return False, f"reference vercel.json missing: {ref_vercel}"

    dst_vercel = deploy / "vercel.json"
    adapt_vercel_json_from_ref(ref_vercel, dst_vercel, ccn, ref_ccn)
    return True, f"restored vercel.json from reference CCN {ref_ccn}"


def ensure_v2_vercel_configuration(
    deploy_dir: os.PathLike[str] | str,
    ccn: str,
    project_root: os.PathLike[str] | str,
    ein_mode: str,
    quarters_csv: str,
    ref_ccn: str | None = None,
) -> tuple[bool, str]:
    """Keep or restore V2 vercel.json; refresh EIN env keys only."""
    deploy = Path(deploy_dir)
    ccn = str(ccn).strip().zfill(6)
    vercel_path = deploy / "vercel.json"
    cfg = _read_vercel_json(vercel_path)

    if vercel_config_is_v2(cfg, ccn):
        patch_vercel_ein_env(vercel_path, ein_mode, quarters_csv)
        return True, "vercel.json preserved (V2); EIN env keys refreshed"

    ok, msg = restore_v2_vercel_json_from_reference(deploy, ccn, project_root, ref_ccn=ref_ccn)
    if not ok:
        return False, msg

    patch_vercel_ein_env(vercel_path, ein_mode, quarters_csv)
    return True, f"{msg}; EIN env keys refreshed"


def ensure_pyarrow_in_requirements(requirements_path: Path) -> bool:
    """Append pyarrow when parquet EIN outputs ship but requirements.txt lacks it."""
    if not requirements_path.is_file():
        return False
    text = requirements_path.read_text(encoding="utf-8")
    if "pyarrow" in text.lower():
        return False
    with requirements_path.open("a", encoding="utf-8") as fh:
        if not text.endswith("\n"):
            fh.write("\n")
        fh.write("pyarrow>=14.0.0\n")
    return True


def legacy_entrypoint_regression_message(deploy_dir: os.PathLike[str] | str, ccn: str) -> str | None:
    """Human-readable failure when a V2 bundle's vercel.json points at legacy flask_app."""
    deploy = Path(deploy_dir)
    ccn = str(ccn).strip().zfill(6)
    is_v2, _ = detect_v2_deployment_bundle(deploy, ccn)
    if not is_v2:
        return None
    cfg = _read_vercel_json(deploy / "vercel.json")
    if vercel_config_is_v2(cfg, ccn):
        return None
    builds = (cfg or {}).get("builds") or []
    src = str((builds[0] if builds else {}).get("src") or "")
    if "flask_app" not in src:
        return None
    return (
        "legacy entrypoint regression: vercel.json points to flask_app but V2 bundle markers "
        "are present. Rerun create_vercel_deployment.py (V2-safe) or "
        "scripts/bootstrap_superdynamic_v2_facility.py <CCN> --ref 315461"
    )
