"""Subscription Dashboard Builder service (premium/dynamic facility dashboards).

Maps operator actions to ``provision_premium_facility.py`` in PBJapp.
Does not write pbj-root. Never logs dashboard passwords.
Deploy is opt-in (typed CCN confirm); local build never passes ``--production``.
"""

from __future__ import annotations

import json
import re
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

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
    ZWELI_NOT_RUN = "ZWELI_NOT_RUN"
    NO_V2_REFERENCE = "NO_V2_REFERENCE"
    UNSAFE_PACKAGE_PATH = "UNSAFE_PACKAGE_PATH"
    MISSING_DEPLOY_DIR = "MISSING_DEPLOY_DIR"
    PROVISION_SCRIPT_MISSING = "PROVISION_SCRIPT_MISSING"
    PASSWORD_REQUIRED = "PASSWORD_REQUIRED"
    DEPLOY_CONFIRM_REQUIRED = "DEPLOY_CONFIRM_REQUIRED"
    INVALID_INTENT = "INVALID_INTENT"


class OperatorIntent(str, Enum):
    PREVIEW = "preview"
    BUILD = "build"
    DEPLOY_STAGING = "deploy_staging"
    DEPLOY_PRODUCTION = "deploy_production"


DASHBOARD_PASSWORD_ENV = "PBJ_DASHBOARD_PASSWORD"
DEPLOY_INTENTS = frozenset(
    {OperatorIntent.DEPLOY_STAGING.value, OperatorIntent.DEPLOY_PRODUCTION.value}
)
VALID_INTENTS = frozenset(item.value for item in OperatorIntent)
VALID_ACCESS_MODES = frozenset({"password_required", "open"})
_SECRET_FORM_KEYS = frozenset(
    {"dashboard_password", "password", "PBJ_DASHBOARD_PASSWORD", DASHBOARD_PASSWORD_ENV}
)


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
    warnings: list[str] = field(default_factory=list)
    can_generate_refresh: bool = False
    can_run_preflight: bool = False
    can_build_local: bool = False
    can_deploy: bool = False
    provision_script: Optional[str] = None
    detail: str = ""
    source_gates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_ccn(ccn: str) -> str:
    s = str(ccn).strip().upper()
    if s.isdigit():
        s = s.zfill(6)
    return s


def _is_six_digit_ccn(ccn: str) -> bool:
    s = _normalize_ccn(ccn)
    return s.isdigit() and len(s) == 6


def provision_script_path(root: Path | None = None) -> Path:
    root = pbjapp_root(root)
    return root / "scripts" / "provision_premium_facility.py"


def redact_secret_text(text: str, secret: str) -> str:
    if not secret or not text:
        return text
    return text.replace(secret, "<redacted>")


def sanitize_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip password-like keys so session/flash never retain secrets."""
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if str(key) in _SECRET_FORM_KEYS:
            continue
        if isinstance(value, dict):
            out[key] = sanitize_action_payload(value)
        elif isinstance(value, str) and key in {"stdout", "stderr", "detail", "command"}:
            out[key] = value
        else:
            out[key] = value
    return out


BUNDLE_STATE_LABELS = {
    "NOT_GENERATED": "Not built yet",
    "GENERATED_LOCALLY": "Available locally",
    "PREFLIGHT_PASSED": "Available locally",
    "DEPLOYED": "Available locally",
    "DEPLOYED_BUT_STALE": "Available locally",
    "UNKNOWN": "Unknown",
}

NOTE_LABELS = {
    "ZWELI_NOT_RUN": None,
    "ZWELI_REQUIRES_ACK": "Provider Information needs review before publish",
    "ZWELI_BLOCKED": "Provider Information is blocked",
    "SOURCE_UNAVAILABLE": "Provider Information is not on disk",
    "UNPROCESSED": "Provider Information is not processed",
    "STRUCTURAL_BLOCKED": "Provider Information failed structural checks",
    "NO_V2_REFERENCE": "No V2 reference bundle in this runtime",
    "UNSAFE_PACKAGE_PATH": "Existing folder is not a V2 bundle",
    "PROVISION_SCRIPT_MISSING": "Provision script is missing (set PBJ_REPO_ROOT)",
    "MISSING_DEPLOY_DIR": None,
    "PASSWORD_REQUIRED": "Enter the dashboard password",
    "INVALID_INTENT": "Choose Preview plan or Build locally",
    "DEPLOY_CONFIRM_REQUIRED": "Check publish and type the CCN to confirm",
}

INTENT_LABELS = {
    "preview": "Preview plan",
    "build": "Update local dashboard",
    "deploy_staging": "Publish to staging",
    "deploy_production": "Publish to production",
    "readiness": "Readiness check",
    "preflight": "Preflight",
    "run": "Update local dashboard",
}


def local_view_port(ccn: str) -> int | None:
    """Last four digits of the CCN (PBJapp local viewing port). None if unusable."""
    n = _normalize_ccn(ccn)
    if len(n) < 4 or not n[-4:].isdigit():
        return None
    port = int(n[-4:])
    if port < 1024:
        return None
    return port


def local_view_url(ccn: str) -> str | None:
    port = local_view_port(ccn)
    if port is None:
        return None
    return f"http://127.0.0.1:{port}/"


def local_view_listening(ccn: str, timeout_s: float = 0.4) -> bool:
    port = local_view_port(ccn)
    if port is None:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_local_v3_script(root: Path | None = None) -> Path:
    root = pbjapp_root(root)
    return root / "scripts" / "start_local_v3_facility.py"


def ensure_local_viewer(
    ccn: str,
    *,
    root: Path | None = None,
    wait_s: float = 20.0,
    restart: bool = False,
) -> dict[str, Any]:
    """Start the on-disk V2 dashboard on the last-four port.

    After a local build, pass restart=True so Flask reloads the new CSVs.
    Reusing a running process keeps the previous Provider Info / case-mix in memory.
    """
    ccn_n = _normalize_ccn(ccn)
    url = local_view_url(ccn_n)
    port = local_view_port(ccn_n)
    if not url or port is None:
        return {"ok": False, "url": None, "started": False, "detail": "That CCN has no local view port."}
    # Deploy bundles live under PBJ_DATA_ROOT; never pass raw PBJapp root as data root.
    deploy = cms_data_paths.facility_deploy_dir(
        ccn_n, cms_data_paths.optional_repo_root(root)
    )
    pbj_root = pbjapp_root(root)
    if not _detect_v2(deploy, ccn_n):
        return {
            "ok": False,
            "url": url,
            "started": False,
            "detail": "No local V2 dashboard on disk yet. Build locally first.",
        }
    if local_view_listening(ccn_n) and not restart:
        return {"ok": True, "url": url, "started": False, "running": True, "detail": "Already running."}
    script = start_local_v3_script(pbj_root)
    if not script.is_file():
        return {
            "ok": False,
            "url": url,
            "started": False,
            "detail": f"Start script missing: {script.name}",
        }
    log_dir = _ROOT / "_scratch" / "local_viewers"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{ccn_n}.log"
    env = os.environ.copy()
    env["PBJ_FACILITY_CCN"] = ccn_n
    env["PBJ_QA_MODE"] = "replace" if restart else "reuse"
    env.pop("PORT", None)
    env.pop("PBJ_DATA_OPS_PORT", None)
    if not (env.get("PBJ_DATA_ROOT") or "").strip():
        try:
            dep = cms_data_paths.deployments_dir()
            if dep is not None and dep.parent:
                env["PBJ_DATA_ROOT"] = str(dep.parent)
        except Exception:
            pass
    log_f = log_path.open("a", encoding="utf-8")
    kwargs: dict[str, Any] = {
        "cwd": str(pbj_root),
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": log_f,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        detached = getattr(subprocess, "DETACHED_PROCESS", 0)
        kwargs["creationflags"] = flags | detached
    else:
        kwargs["start_new_session"] = True
        kwargs["close_fds"] = True
    subprocess.Popen([sys.executable, str(script)], **kwargs)
    try:
        log_f.close()
    except OSError:
        pass
    if restart:
        drop_deadline = time.monotonic() + 8.0
        while time.monotonic() < drop_deadline:
            if not local_view_listening(ccn_n):
                break
            time.sleep(0.2)
    deadline = time.monotonic() + max(wait_s, 1.0)
    while time.monotonic() < deadline:
        if local_view_listening(ccn_n):
            return {
                "ok": True,
                "url": url,
                "started": True,
                "running": True,
                "restarted": bool(restart),
                "detail": "Local dashboard reloaded with the latest files." if restart else "Local dashboard is ready.",
            }
        time.sleep(0.4)
    return {
        "ok": False,
        "url": url,
        "started": True,
        "running": False,
        "detail": f"Started the viewer but {url} is not answering yet. Wait a few seconds and open it again.",
    }


def _pbjapp_bundle_root() -> Path:
    try:
        from provider_quarter_mapping import _pbjapp_root_for_bundles

        return _pbjapp_root_for_bundles(None)
    except Exception:
        return cms_data_paths.repo_root()


def _provider_month_count(deploy_dir: Path, ccn: str) -> int | None:
    path = deploy_dir / f"facility_{ccn}_provider_info_data.csv"
    if not path.is_file():
        return None
    try:
        from provider_quarter_mapping import processing_months_from_provider_csv

        return len(processing_months_from_provider_csv(path))
    except Exception:
        return None


def _provider_slice_history_note(deploy_dir: Path, ccn: str) -> str | None:
    months = _provider_month_count(deploy_dir, ccn)
    if months is None:
        return None
    if months >= 6:
        return None
    return (
        f"Provider Information: {months} month{'s' if months != 1 else ''} packaged · "
        "rebuild to add full history"
    )


def _last_built_label(deploy_dir: Path | None) -> str:
    if not deploy_dir or not deploy_dir.is_dir():
        return ""
    try:
        from datetime import datetime

        return datetime.fromtimestamp(deploy_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return ""


def _data_status_rows(
    *,
    ccn: str,
    deploy_dir: Path | None,
    local_available: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        from provider_quarter_mapping import audit_active_provider_quarter_mapping

        q_audit = audit_active_provider_quarter_mapping()
    except Exception:
        q_audit = {}
    q_ok = not q_audit.get("needs_attention")
    q_text = (q_audit.get("resolved_quarter") or "Current") if q_ok else (q_audit.get("detail") or "Needs attention")
    rows.append(
        {
            "id": "quarter_map",
            "label": "interval_quarter_mapping.json",
            "ok": q_ok,
            "text": q_text,
        }
    )
    if deploy_dir and deploy_dir.is_dir():
        months = _provider_month_count(deploy_dir, ccn)
        if months is None:
            pi_ok, pi_text = False, "missing"
        elif months < 6:
            pi_ok, pi_text = False, f"{months} month{'s' if months != 1 else ''} · update needed"
        else:
            pi_ok, pi_text = True, f"{months} months"
        staff_ok = (deploy_dir / f"facility_{ccn}_complete_data.csv").is_file()
        own_ok = (deploy_dir / "ownership" / f"NH_Ownership_facility_{ccn}.csv").is_file() or (
            deploy_dir / "ownership" / f"SNF_All_Owners_facility_{ccn}.csv"
        ).is_file()
    else:
        pi_ok, pi_text = False, "not built"
        staff_ok = False
        own_ok = False
    rows.append(
        {
            "id": "provider_info",
            "label": f"facility_{ccn}_provider_info_data.csv",
            "ok": pi_ok,
            "text": pi_text,
        }
    )
    rows.append(
        {
            "id": "staffing",
            "label": f"facility_{ccn}_complete_data.csv",
            "ok": staff_ok,
            "text": "on disk" if staff_ok else ("missing" if local_available else "build first"),
        }
    )
    rows.append(
        {
            "id": "ownership",
            "label": "ownership/*.csv",
            "ok": own_ok,
            "text": "on disk" if own_ok else ("missing" if local_available else "build first"),
        }
    )
    return rows


def builder_view_model(status: dict[str, Any]) -> dict[str, Any]:
    """Operator-facing copy. No JSON dumps, no internal script names."""
    state = str(status.get("bundle_state") or "")
    notes: list[str] = []
    for code in status.get("blockers") or []:
        if state == "NOT_GENERATED" and code == "MISSING_DEPLOY_DIR":
            continue
        label = NOTE_LABELS.get(code, code.replace("_", " ").capitalize())
        if label:
            notes.append(label)
    ccn = str(status.get("ccn") or "")
    # Canonical deploy path (PBJ_DATA_ROOT / env / json). Never PBJapp repo stub.
    deploy_dir = cms_data_paths.facility_deploy_dir(ccn) if ccn else None
    local_available = bool(status.get("is_v2")) and bool(deploy_dir and deploy_dir.is_dir())
    if deploy_dir and deploy_dir.is_dir():
        slice_note = _provider_slice_history_note(deploy_dir, ccn)
        if slice_note:
            notes.append(slice_note)
    data_status = _data_status_rows(ccn=ccn, deploy_dir=deploy_dir, local_available=local_available)
    needs_update = any(not row["ok"] for row in data_status if row["id"] in {"provider_info", "quarter_map"})
    if local_available:
        primary_build_label = "Update local dashboard" if needs_update else "Rebuild"
        primary_build_kind = "update" if needs_update else "rebuild"
    else:
        primary_build_label = "Build local dashboard"
        primary_build_kind = "build"
    publish_reasons: list[str] = []
    if not local_available:
        publish_reasons.append("Build a local dashboard before publishing.")
    for code in status.get("blockers") or []:
        if code in SOURCE_DATA_REFRESH_BLOCKERS:
            label = NOTE_LABELS.get(code)
            if label:
                publish_reasons.append(label)
    if needs_update and local_available:
        publish_reasons.append("Update local data (Provider Information history or quarter map) before publishing.")
    readiness = status.get("readiness") or {}
    if local_available and readiness and readiness.get("ok") is False:
        publish_reasons.append("Readiness check has not passed for this facility.")
    can_publish = bool(status.get("can_deploy")) and local_available and not publish_reasons
    name = status.get("facility_name") or "This facility"
    view_url = local_view_url(ccn) if local_available else None
    last_built = _last_built_label(deploy_dir) if local_available else ""
    return {
        **status,
        "headline": name,
        "state_label": BUNDLE_STATE_LABELS.get(state, state.replace("_", " ").title()),
        "state_kind": "empty" if not local_available else "ok",
        "local_available": local_available,
        "last_built": last_built,
        "template_label": "V2" if local_available else ("Ready to create" if state == "NOT_GENERATED" else "Not V2"),
        "summary": (
            "Open it from this page. It is not live on the web until you publish."
            if local_available
            else "No local dashboard on disk yet."
        ),
        "notes": notes,
        "data_status": data_status,
        "needs_update": needs_update,
        "primary_build_label": primary_build_label,
        "primary_build_kind": primary_build_kind,
        "prove_command": "python scripts/check_provider_quarter_flow.py --prove --ccn " + ccn if ccn else "python scripts/check_provider_quarter_flow.py --prove",
        "status_command": f"python scripts/check_provider_quarter_flow.py --ccn {ccn}" if ccn else "python scripts/check_provider_quarter_flow.py",
        "can_build_local": bool(status.get("can_build_local")),
        "can_deploy": bool(status.get("can_deploy")),
        "can_publish": can_publish,
        "publish_block_reason": publish_reasons[0] if publish_reasons else "",
        "can_run_preflight": bool(status.get("can_run_preflight")),
        "local_view_url": view_url,
        "local_view_port": local_view_port(ccn) if local_available else None,
        "local_view_running": local_view_listening(ccn) if local_available else False,
    }


DIRTY_CRITICAL_OPERATOR_MSG = (
    "Production/staging is blocked: PBJapp has uncommitted dashboard runtime files. "
    "Local viewing is fine. Commit or restore those files in PBJapp, then publish again."
)

_PROVENANCE_REASON_LABELS = {
    "artifact changed since provenance was recorded": (
        "the file was rewritten after packaging recorded it"
    ),
    "artifact provenance missing": "the provenance sidecar is missing",
    "artifact source release is STALE": "the bundled slice is not the active CMS release",
    "artifact missing": "the expected file is missing from the bundle",
}


def _preflight_operator_message(blob: str) -> str | None:
    """Turn PACKAGE PREFLIGHT FAIL log lines into a one-line operator reason."""
    if not blob:
        return None
    fails: list[tuple[str, str]] = []
    lines = blob.splitlines()
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if not stripped.startswith("FAIL "):
            continue
        rest = stripped[5:].strip()
        extra = ""
        if i + 1 < len(lines) and lines[i + 1].startswith("     "):
            extra = lines[i + 1].strip()
        fails.append((rest, extra))
    if not fails and "PACKAGE PREFLIGHT FAIL" not in blob:
        return None
    if fails:
        rest, extra = fails[0]
        cap = rest
        if cap.endswith(" ACTIVE-release provenance"):
            cap = cap[: -len(" ACTIVE-release provenance")].strip()
        why = _PROVENANCE_REASON_LABELS.get(extra, extra)
        if why:
            return f"Checks failed: {cap} — {why}."[:500]
        return f"Checks failed: {cap}."[:500]
    return "Bundle checks failed."


def _dirty_critical_names_from_blob(blob: str) -> list[str]:
    keys = (
        "templates/",
        "static/js/pbj_v3_",
        "scripts/provision_premium",
        "scripts/deploy_vercel_facility.py",
        "create_vercel_deployment.py",
        "packaging_refresh_gates.py",
        "factory_deploy_gates_lib.py",
        "release_source/superdynamic_v2/",
        "pbj_premium_access.py",
    )
    names: list[str] = []
    for ln in blob.splitlines():
        s = ln.strip().replace("\\", "/")
        if not s or s.startswith("PRECONDITION") or "allow-dirty" in s or "git working tree" in s:
            continue
        token = s.split()[-1]
        if any(k in token for k in keys) and token not in names:
            names.append(token)
        if len(names) >= 6:
            break
    return names


_TRACEBACK_EXC_RE = re.compile(
    r"^(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception))\s*:\s*(?P<msg>.+)$"
)


def _traceback_exception_operator_message(blob: str) -> str | None:
    """Final concrete traceback exception line when no ERROR:/Unexpected error: match."""
    if not blob:
        return None
    for ln in reversed(blob.splitlines()):
        stripped = ln.strip()
        if not stripped:
            continue
        if _TRACEBACK_EXC_RE.match(stripped):
            return stripped[:500]
    return None



def _failed_capability_operator_message(blob: str) -> str | None:
    """Surface artifact-contract FAILED CAPABILITY blocks in the UI note."""
    lines = blob.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped.startswith("FAILED CAPABILITY:"):
            continue
        parts = [stripped]
        for cont in lines[i + 1 :]:
            if cont.startswith("  ") or cont.startswith("\t"):
                parts.append(cont.strip())
            elif not cont.strip():
                break
            else:
                break
        return " ".join(parts)[:500]
    return None


def operator_failure_message(result: dict[str, Any] | None) -> str:
    if not result:
        return "Build failed."
    errors = result.get("errors") or []
    if errors:
        return "; ".join(NOTE_LABELS.get(str(e), str(e).replace("_", " ")) for e in errors)
    blob = f"{result.get('stdout') or ''}\n{result.get('stderr') or ''}"
    if "password-gated premium production deploy requires" in blob:
        return (
            "Production stopped before Vercel: this facility is password-gated and "
            "needs both API and browser acceptance. Retry Publish."
        )
    if "PRIVATE_DATA_ROOT is required for --confirm-deploy" in blob:
        return (
            "Publish stopped before Vercel: PBJapp has no private-data folder configured, "
            "so the upload could not start. Local dashboard is unchanged."
        )
    for ln in reversed(blob.splitlines()):
        stripped = ln.strip()
        if stripped.startswith("Unexpected error:"):
            text = stripped.replace("Unexpected error:", "", 1).strip()
            if "extraction manifests require PRIVATE_DATA_ROOT" in text:
                return (
                    "Packaging stopped while writing the extraction manifest. "
                    "Local slices may already be on disk."
                )
            return text[:500]
        if stripped.startswith("ERROR:"):
            return stripped[6:].strip()[:500]
    cap_msg = _failed_capability_operator_message(blob)
    if cap_msg:
        return cap_msg
    traceback_msg = _traceback_exception_operator_message(blob)
    if traceback_msg:
        return traceback_msg
    preflight_msg = _preflight_operator_message(blob)
    if preflight_msg:
        return preflight_msg
    if "dirty CRITICAL" in blob or "allow-dirty-critical-files" in blob:
        names = _dirty_critical_names_from_blob(blob)
        if names:
            return DIRTY_CRITICAL_OPERATOR_MSG + " Blocking: " + ", ".join(names) + "."
        return DIRTY_CRITICAL_OPERATOR_MSG
    stderr = (result.get("stderr") or "").strip()
    if stderr:
        lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
        uniq = []
        for ln in lines:
            if ln.startswith("PRECONDITION BLOCK:") and "CRITICAL" not in ln:
                continue
            if ln not in uniq:
                uniq.append(ln)
        if uniq:
            return " ".join(uniq[:3])[:500]
    return result.get("detail") or "Build failed."


def last_action_summary(last_action: dict[str, Any] | None) -> str:
    if not last_action:
        return ""
    result = last_action.get("result") if isinstance(last_action.get("result"), dict) else last_action
    intent = (result or {}).get("intent") or last_action.get("action") or ""
    label = INTENT_LABELS.get(str(intent), str(intent).replace("_", " "))
    ok = (result or {}).get("ok")
    if ok is True:
        if (result or {}).get("dry_run"):
            return f"{label} finished — no files written, no deploy."
        if (result or {}).get("deployed"):
            return f"{label} finished."
        return f"{label} finished — local only."
    if ok is False:
        return f"{label} blocked: {operator_failure_message(result)}"
    return label


def validate_dashboard_run(
    *,
    ccn: str,
    intent: str,
    access_mode: str,
    password: str,
    confirm_ccn: str,
    confirm_publish: bool = False,
) -> list[str]:
    """Fail-closed checks before invoking provision. Never includes the password."""
    errors: list[str] = []
    ccn_n = _normalize_ccn(ccn)
    if not _is_six_digit_ccn(ccn_n):
        errors.append("CCN must be six digits")
    intent_n = (intent or "").strip()
    if intent_n not in VALID_INTENTS:
        errors.append(DashboardActionBlocker.INVALID_INTENT.value)
    mode = (access_mode or "").strip()
    if mode not in VALID_ACCESS_MODES:
        errors.append("access mode must be password_required or open")
    if mode == "password_required" and not (password or "").strip():
        errors.append(DashboardActionBlocker.PASSWORD_REQUIRED.value)
    if intent_n in DEPLOY_INTENTS:
        typed = str(confirm_ccn or "").strip().zfill(6) if str(confirm_ccn or "").strip().isdigit() else str(confirm_ccn or "").strip()
        if typed != ccn_n or not confirm_publish:
            errors.append(DashboardActionBlocker.DEPLOY_CONFIRM_REQUIRED.value)
    script = provision_script_path()
    if not script.is_file():
        errors.append(DashboardActionBlocker.PROVISION_SCRIPT_MISSING.value)
    return errors


def pbjapp_root(root: Path | None = None) -> Path:
    """PBJapp tree that contains provision_premium_facility.py — not the Data Ops repo."""
    if root is not None:
        return root.resolve()
    via = cms_data_paths.repo_root()
    if (via / "scripts" / "provision_premium_facility.py").is_file():
        return via.resolve()
    sibling = (_ROOT.parent / "PBJapp").resolve()
    if (sibling / "scripts" / "provision_premium_facility.py").is_file():
        return sibling
    return via.resolve()


def _private_data_root_from_json(cfg_root: Path) -> Path | None:
    cfg_path = cfg_root / "data_paths.local.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    val = cfg.get("private_data_root") if isinstance(cfg, dict) else None
    if not isinstance(val, str) or not val.strip():
        return None
    p = Path(val.strip()).expanduser()
    if not p.is_absolute():
        p = cfg_root / p
    return p.resolve()


_PRIVATE_DATA_FALLBACKS = (Path("D:/Data/PBJ320_Private"),)


def apply_pbjapp_provision_env(env: dict[str, str], *, root: Path | None = None) -> dict[str, str]:
    """Run provision as PBJapp. Never log resolved private paths."""
    pbj_root = pbjapp_root(root)
    out = dict(env)
    out["PBJ_REPO_ROOT"] = str(pbj_root)
    if (out.get("PRIVATE_DATA_ROOT") or "").strip():
        return out
    for cfg_root in (pbj_root, _ROOT):
        resolved = _private_data_root_from_json(cfg_root)
        if resolved is not None and resolved.exists():
            out["PRIVATE_DATA_ROOT"] = str(resolved)
            return out
    for candidate in _PRIVATE_DATA_FALLBACKS:
        if candidate.is_dir():
            out["PRIVATE_DATA_ROOT"] = str(candidate.resolve())
            return out
    return out



def vercel_public_url_for_intent(ccn: str, intent: str) -> str | None:
    """Known Vercel alias: staging uses -v2 project; production does not."""
    ccn_n = _normalize_ccn(ccn)
    if not ccn_n:
        return None
    intent_n = (intent or "").strip()
    if intent_n == OperatorIntent.DEPLOY_STAGING.value:
        return f"https://pbj320-{ccn_n}-v2.vercel.app"
    if intent_n == OperatorIntent.DEPLOY_PRODUCTION.value:
        return f"https://pbj320-{ccn_n}.vercel.app"
    return None


def build_provision_argv(
    *,
    ccn: str,
    intent: str,
    access_mode: str,
    cold: bool,
    scaffold_config: bool,
    refresh_data: bool,
    allow_dirty_tree: bool,
    allow_dirty_critical_files: bool = False,
) -> list[str]:
    ccn_n = _normalize_ccn(ccn)
    cmd = [
        sys.executable,
        str(provision_script_path()),
        "--ccn",
        ccn_n,
        "--access-mode",
        access_mode,
        "--verbose",
    ]
    if intent == OperatorIntent.PREVIEW.value:
        cmd.append("--dry-run")
    if cold:
        cmd.append("--cold")
    if scaffold_config:
        cmd.append("--scaffold-config")
    if refresh_data:
        cmd.append("--refresh-data")
    deploy = intent in DEPLOY_INTENTS
    if allow_dirty_tree:
        cmd.append("--allow-dirty-tree")
    if allow_dirty_critical_files:
        cmd.append("--allow-dirty-critical-files")
    if intent == OperatorIntent.DEPLOY_STAGING.value:
        cmd.append("--staging")
    if intent == OperatorIntent.DEPLOY_PRODUCTION.value:
        cmd.extend(
            [
                "--production",
                "--run-api-acceptance",
                "--run-browser-acceptance",
            ]
        )
    if intent in DEPLOY_INTENTS and "--production" in cmd and "--staging" in cmd:
        raise ValueError("staging and production cannot both be set")
    return cmd


def _infer_cold_and_scaffold(ccn: str, *, root: Path | None = None) -> tuple[bool, bool]:
    ccn_n = _normalize_ccn(ccn)
    deploy_dir = cms_data_paths.facility_deploy_dir(
        ccn_n, cms_data_paths.optional_repo_root(root)
    )
    pbj_root = pbjapp_root(root)
    config_path = pbj_root / "facility_config" / f"{ccn_n}.json"
    missing_bundle = not deploy_dir.is_dir()
    return missing_bundle, missing_bundle or not config_path.is_file()


def run_dashboard_provision(
    *,
    ccn: str,
    intent: str,
    access_mode: str,
    password: str,
    confirm_ccn: str = "",
    confirm_publish: bool = False,
    refresh_data: bool = False,
    allow_dirty_tree: bool = False,
    allow_dirty_critical_files: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Run provision. Password is injected only as child env; never returned."""
    secret = (password or "").strip()
    errors = validate_dashboard_run(
        ccn=ccn,
        intent=intent,
        access_mode=access_mode,
        password=secret,
        confirm_ccn=confirm_ccn,
        confirm_publish=confirm_publish,
    )
    ccn_n = _normalize_ccn(ccn)
    if errors:
        return sanitize_action_payload(
            {
                "ok": False,
                "ccn": ccn_n,
                "intent": intent,
                "access_mode": access_mode,
                "errors": errors,
                "deployed": False,
            }
        )
    cold, scaffold = _infer_cold_and_scaffold(ccn_n)
    argv = build_provision_argv(
        ccn=ccn_n,
        intent=intent,
        access_mode=access_mode,
        cold=cold,
        scaffold_config=scaffold,
        refresh_data=refresh_data,
        allow_dirty_tree=allow_dirty_tree,
        allow_dirty_critical_files=allow_dirty_critical_files,
    )
    env = os.environ.copy()
    if access_mode == "password_required":
        env[DASHBOARD_PASSWORD_ENV] = secret
    else:
        env.pop(DASHBOARD_PASSWORD_ENV, None)
    env = apply_pbjapp_provision_env(env, root=pbjapp_root())
    run = runner or (
        lambda cmd, env, cwd: subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
        )
    )
    proc = run(argv, env, pbjapp_root())
    stdout = redact_secret_text(proc.stdout or "", secret)
    stderr = redact_secret_text(proc.stderr or "", secret)
    deployed = intent in DEPLOY_INTENTS and proc.returncode == 0 and intent != OperatorIntent.PREVIEW.value
    return sanitize_action_payload(
        {
            "ok": proc.returncode == 0,
            "ccn": ccn_n,
            "intent": intent,
            "access_mode": access_mode,
            "exit_code": int(proc.returncode),
            "stdout": stdout[-8000:],
            "stderr": stderr[-4000:],
            "command": argv,
            "cold": cold,
            "scaffold_config": scaffold,
            "deployed": deployed,
            "public_url": (
                vercel_public_url_for_intent(ccn_n, intent) if deployed else None
            ),
            "dry_run": intent == OperatorIntent.PREVIEW.value,
            "operator_message": (
                None
                if proc.returncode == 0
                else operator_failure_message(
                    {"stdout": stdout, "stderr": stderr, "ok": False, "errors": []}
                )
            ),
        }
    )


def _detect_v2(deploy_dir: Path, ccn: str) -> bool:
    return (deploy_dir / f"facility_{ccn}_superdynamic_dashboard.py").is_file()


def _find_v2_reference(root: Path | None = None, preferred: tuple[str, ...] = ("315461", "315128", "335513")) -> Optional[str]:
    data_root = cms_data_paths.optional_repo_root(root)
    for ref in preferred:
        d = cms_data_paths.facility_deploy_dir(ref, data_root)
        if _detect_v2(d, ref):
            return ref
    # any local V2 under PBJ_DATA_ROOT (or explicit test root)
    dep = cms_data_paths.deployments_dir(data_root)
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
        DashboardActionBlocker.ZWELI_NOT_RUN.value,
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
    if state == ZweliState.NOT_RUN:
        blockers.append(DashboardActionBlocker.ZWELI_NOT_RUN.value)
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
    # Always pass canonical facility_deploy_dir so Checks do not hit the repo stub.
    deploy_dir = cms_data_paths.facility_deploy_dir(ccn)
    proc = subprocess.run(
        [sys.executable, str(script), ccn, "--deploy-dir", str(deploy_dir)],
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
    pbj_root = pbjapp_root(root)
    ccn = _normalize_ccn(ccn)
    deploy_dir = cms_data_paths.facility_deploy_dir(
        ccn, cms_data_paths.optional_repo_root(root)
    )
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
    # Import bootstrap from PBJapp scripts
    sys.path.insert(0, str(pbj_root / "scripts"))
    try:
        import bootstrap_superdynamic_v2_facility as boot  # type: ignore
    except ImportError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        result = boot.run_bootstrap(pbj_root, ccn, ref)
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
    # Keep optional root for tests; never treat PBJapp/repo_root as the data root.
    ccn = _normalize_ccn(ccn)
    deploy_dir = cms_data_paths.facility_deploy_dir(
        ccn, cms_data_paths.optional_repo_root(root)
    )
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
            "No local bundle yet. Look up the CCN, set a dashboard password, "
            "then Build locally (or Preview plan). Publish is a separate confirm."
        )
    elif is_v2:
        bundle_state = BundleState.GENERATED_LOCALLY.value
        detail = (
            "A local V2 dashboard is already on disk. Open it from this page. "
            "It is not live on the web until you confirm publish."
        )
    else:
        bundle_state = BundleState.GENERATED_LOCALLY.value
        detail = (
            "Deploy dir exists but is not V2 superdynamic — Build locally will "
            "follow the provision cold/scaffold path if needed."
        )

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
    script = provision_script_path(root)
    has_script = script.is_file()
    source_warnings = [b for b in blockers if b in SOURCE_DATA_REFRESH_BLOCKERS]
    if not has_script:
        blockers.append(DashboardActionBlocker.PROVISION_SCRIPT_MISSING.value)

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
        warnings=source_warnings,
        can_generate_refresh=can_refresh,
        can_run_preflight=can_preflight,
        can_build_local=has_script,
        can_deploy=has_script,
        provision_script=str(script) if has_script else None,
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
    "Existing V2: provision_premium_facility.py without --production. "
    "Optional --refresh-data. Deploy is a separate intent with typed CCN confirm."
)
COLD_NEW_V2_PATH = (
    "New CCN: provision_premium_facility.py --cold --scaffold-config "
    "--access-mode password_required (no --production). "
    "Deploy only via explicit Build and deploy after typing the CCN."
)
