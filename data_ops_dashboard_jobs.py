"""Background dashboard provision jobs with phase-based progress.

Percent is completed-step weight / planned-step weight, from real
``--- Phase: name ---`` lines. No elapsed-time animation.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_ops_dashboard import (
    DASHBOARD_PASSWORD_ENV,
    DEPLOY_INTENTS,
    apply_pbjapp_provision_env,
    build_provision_argv,
    operator_failure_message,
    pbjapp_root,
    redact_secret_text,
    sanitize_action_payload,
    validate_dashboard_run,
    _infer_cold_and_scaffold,
    _normalize_ccn,
)
import cms_data_paths

_ROOT = Path(__file__).resolve().parent
JOBS_DIR = _ROOT / "_scratch" / "dashboard_jobs"
PHASE_LINE = re.compile(r"^--- Phase: ([a-z_]+) ---")
# Local --package often runs create_vercel_deployment inside the factory "package" phase.
EXTRACT_HINT = re.compile(
    r"\[nonnurse\]|create_vercel_deployment\.py|Creating Vercel Deployment Package|"
    r"dailynursestaffing|facility_.*_nonnurse",
    re.I,
)
NONNURSE_FRAC = re.compile(r"\[nonnurse\] process .+?\((\d+)/(\d+)\)")
EXTRACT_DONE_PHASES = frozenset(
    {"identity_reconcile", "bootstrap", "known_empty_validate", "preflight"}
)

# Operator steps and relative wall-clock share for a typical cold local build.
# Extract dominates; weights are not a timer — they only apply after a step finishes.
OPERATOR_STEPS: tuple[dict[str, Any], ...] = (
    {
        "id": "ready",
        "label": "Prepare sources",
        "phases": ("scaffold_config", "facility_config", "cms_data_ready", "pre_extract"),
        "weight": 10,
        "local_only": True,
    },
    {
        "id": "staffing",
        "label": "Staffing extract",
        "phases": ("extract",),
        "weight": 55,
        "local_only": True,
    },
    {
        "id": "identity",
        "label": "Identity",
        "phases": ("identity_reconcile",),
        "weight": 5,
        "local_only": True,
    },
    {
        "id": "ownership",
        "label": "Ownership",
        "phases": ("bootstrap",),
        "weight": 10,
        "local_only": True,
    },
    {
        "id": "package",
        "label": "Package bundle",
        "phases": ("known_empty_validate", "package"),
        "weight": 15,
    },
    {
        "id": "checks",
        "label": "Checks",
        "phases": ("preflight",),
        "weight": 5,
    },
    {
        "id": "publish",
        "label": "Ship to Vercel",
        "phases": ("vercel_project", "vercel_link", "deploy", "api_acceptance", "browser_acceptance"),
        "weight": 20,
        "deploy_only": True,
    },
)


def jobs_dir() -> Path:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return JOBS_DIR


def planned_operator_steps(intent: str) -> list[dict[str, Any]]:
    deploy = intent in DEPLOY_INTENTS
    out: list[dict[str, Any]] = []
    for step in OPERATOR_STEPS:
        if step.get("deploy_only") and not deploy:
            continue
        if deploy and step.get("local_only"):
            continue
        label = str(step["label"])
        if deploy and step["id"] == "package":
            label = "Confirm local dashboard"
        out.append(
            {
                "id": step["id"],
                "label": label,
                "weight": int(step["weight"]),
                "phases": list(step["phases"]),
                "state": "pending",
                "note": "Waiting",
            }
        )
    return out


def parse_started_phases(log_text: str) -> list[str]:
    seen: list[str] = []
    for raw in (log_text or "").splitlines():
        m = PHASE_LINE.match(raw.strip())
        if not m:
            continue
        name = m.group(1)
        if name not in seen:
            seen.append(name)
    return seen


ASSESS_LINE = re.compile(
    r"^\s+(nurse_pbj|nonnurse_pbj|citations|provider_info|ein_employee_detail|ownership|runtime_config)\s+(reuse|REFRESH)\b",
    re.I | re.M,
)
SNAPSHOT_FRAC = re.compile(r"\[SNAPSHOT(?:-CORE)?\]\s+(\d+)\s*/\s*(\d+)\s+dates")
PKG_PHASE = re.compile(r"^Phase (\d+)/(\d+):\s*(.+)$", re.M)
PKG_PHASE_SHORT = {
    "Smart data refresh assessment": "Checking data",
    "Incremental code sync": "Copying UI",
    "Roster snapshot reuse-or-fail": "Roster snapshot",
    "Package manifest validation": "Checking package",
    "Fast bundle checks": "Checking bundle",
}
SKIP_ROW = re.compile(
    r"^(facility_config|cms_data_ready|pre_extract|extract|identity_reconcile|bootstrap|package|known_empty_validate)\s+skipped",
    re.M,
)


def _last_nonnurse_fraction(log_text: str) -> tuple[int, int] | None:
    last: tuple[int, int] | None = None
    for m in NONNURSE_FRAC.finditer(log_text or ""):
        n, d = int(m.group(1)), int(m.group(2))
        if d > 0:
            last = (n, d)
    return last


def _last_snapshot_fraction(log_text: str) -> tuple[int, int] | None:
    last: tuple[int, int] | None = None
    for m in SNAPSHOT_FRAC.finditer(log_text or ""):
        n, d = int(m.group(1)), int(m.group(2))
        if d > 0:
            last = (n, d)
    return last


def _log_evidence(log_text: str) -> dict[str, Any]:
    text = log_text or ""
    assess = {m.group(1).lower(): m.group(2).lower() for m in ASSESS_LINE.finditer(text)}
    skipped = {m.group(1) for m in SKIP_ROW.finditer(text)}
    nurse_done = assess.get("nurse_pbj") == "reuse" or "complete_data.csv: validated" in text or (
        "Extraction Summary:" in text and "complete_data.csv" in text
    )
    nn_done = (
        assess.get("nonnurse_pbj") == "reuse"
        or "nonnurse_daily.csv: validated" in text
        or "[nonnurse] Wrote" in text
    )
    pkg_phase = None
    for m in PKG_PHASE.finditer(text):
        title = (m.group(3) or "").strip()
        pkg_phase = (int(m.group(1)), int(m.group(2)), title)
    return {
        "assess": assess,
        "skipped": skipped,
        "saw_assess": "Smart data refresh assessment" in text,
        "nurse_done": bool(nurse_done),
        "nn_done": bool(nn_done),
        "staffing_done": bool(nurse_done and nn_done) or "extract" in skipped,
        "staffing_reused": assess.get("nurse_pbj") == "reuse" and assess.get("nonnurse_pbj") == "reuse",
        "identity_skipped": "identity_reconcile" in skipped,
        "ownership_skipped": "bootstrap" in skipped,
        "ownership_refresh": assess.get("ownership") == "refresh",
        "ownership_reuse": assess.get("ownership") == "reuse",
        "snapshot": _last_snapshot_fraction(text),
        "pkg_phase": pkg_phase,
    }


def _package_step_note(ev: dict[str, Any]) -> str:
    if ev.get("snapshot"):
        n, d = ev["snapshot"]
        return f"Roster {n}/{d}"
    ph = ev.get("pkg_phase")
    if ph:
        n, total = int(ph[0]), int(ph[1])
        title = str(ph[2] or "").strip() if len(ph) > 2 else ""
        short = PKG_PHASE_SHORT.get(title, title)
        if short:
            return f"{short} · {n}/{total}"
        return f"{n}/{total}"
    return "Now"


def _extract_still_running(log_text: str, started: list[str]) -> bool:
    if any(ph in EXTRACT_DONE_PHASES for ph in started):
        return False
    ev = _log_evidence(log_text)
    if ev["staffing_done"]:
        return False
    return bool(EXTRACT_HINT.search(log_text or ""))


def progress_from_log(
    log_text: str,
    intent: str,
    *,
    finished: bool = False,
    ok: bool = False,
    fail_note: str | None = None,
) -> dict[str, Any]:
    steps = planned_operator_steps(intent)
    started = parse_started_phases(log_text)
    ev = _log_evidence(log_text)
    phase_to_step = {}
    for step in steps:
        for ph in step["phases"]:
            phase_to_step[ph] = step["id"]

    started_for_steps = list(started)
    extract_running = (not finished) and _extract_still_running(log_text, started)
    if extract_running:
        started_for_steps = [p for p in started_for_steps if p not in ("package", "known_empty_validate")]
        if "extract" not in started_for_steps:
            started_for_steps.append("extract")

    started_step_ids: list[str] = []
    for ph in started_for_steps:
        sid = phase_to_step.get(ph)
        if sid and sid not in started_step_ids:
            started_step_ids.append(sid)

    for step in steps:
        step["note"] = "Waiting"
        if finished and ok:
            step["state"] = "done"
            step["note"] = "Done"
            continue
        if step["id"] in started_step_ids:
            idx = started_step_ids.index(step["id"])
            later = started_step_ids[idx + 1 :]
            if later:
                step["state"] = "done"
                step["note"] = "Done"
            elif finished and not ok:
                step["state"] = "failed"
                step["note"] = "Failed"
            else:
                step["state"] = "active"
                step["note"] = "Now"
        else:
            step["state"] = "pending"

    by_id = {s["id"]: s for s in steps}

    def mark_done(step_id: str, note: str) -> None:
        step = by_id.get(step_id)
        if not step or step["state"] == "failed":
            return
        step["state"] = "done"
        step["note"] = note

    if ev["saw_assess"] or "pre_extract" in ev["skipped"] or "cms_data_ready" in ev["skipped"]:
        mark_done("ready", "Already set up")
    if ev["staffing_done"]:
        mark_done("staffing", "Reused on disk" if ev["staffing_reused"] else "Done")
    if ev["identity_skipped"]:
        mark_done("identity", "Not this run")
    if "package" in ev["skipped"]:
        mark_done("package", "Already packaged")
    if ev["ownership_reuse"] or (ev["ownership_skipped"] and not ev["ownership_refresh"]):
        mark_done("ownership", "Already in bundle" if ev["ownership_reuse"] or ev["ownership_skipped"] else "Done")
    elif ev["ownership_refresh"] and by_id.get("ownership", {}).get("state") != "failed":
        own = by_id.get("ownership")
        if own and own["state"] != "active":
            own["note"] = "Needs refresh"

    if extract_running and by_id.get("staffing"):
        by_id["staffing"]["state"] = "active"
        nn = _last_nonnurse_fraction(log_text)
        by_id["staffing"]["note"] = f"Non-nurse {nn[0]}/{nn[1]}" if nn else "Extracting"
        pkg = by_id.get("package")
        if pkg and pkg["state"] == "active":
            pkg["state"] = "pending"
            pkg["note"] = "Waiting"
    elif extract_running and by_id.get("package"):
        pkg = by_id["package"]
        if pkg["state"] != "failed":
            pkg["state"] = "active"
            pkg["note"] = "Refreshing local files"

    if (not extract_running) and ev["staffing_done"] and by_id.get("package") and by_id["package"]["state"] in (
        "pending",
        "active",
        "failed",
    ):
        if "identity" in by_id and by_id["identity"]["state"] == "pending":
            mark_done("identity", "Not this run")
        pkg = by_id["package"]
        if pkg["state"] != "failed":
            pkg["state"] = "active"
            pkg["note"] = _package_step_note(ev)

    active_idx = next((i for i, s in enumerate(steps) if s["state"] == "active"), None)
    if active_idx is not None:
        for i, step in enumerate(steps):
            if i < active_idx and step["state"] == "pending":
                if step["id"] == "ownership" and ev["ownership_refresh"]:
                    continue
                step["state"] = "done"
                if step["note"] == "Waiting":
                    step["note"] = "Skipped"

    if finished and ok:
        for step in steps:
            step["state"] = "done"
            if step["note"] in ("Waiting", "Now"):
                step["note"] = "Done"

    preflight_failed = "PACKAGE PREFLIGHT FAIL" in (log_text or "") or re.search(
        r"^preflight\s+failed", log_text or "", re.M | re.I
    )
    if finished and not ok and preflight_failed:
        pkg = by_id.get("package")
        if pkg and pkg["state"] == "failed":
            pkg["state"] = "done"
            pkg["note"] = "Done"
        chk = by_id.get("checks")
        if chk:
            chk["state"] = "failed"
            why = (fail_note or "").strip()
            if not why or why == "Build failed.":
                why = operator_failure_message(
                    {"ok": False, "errors": [], "stdout": log_text or "", "stderr": ""}
                )
            chk["note"] = why if why and why != "Build failed." else "Failed"

    total = sum(int(s["weight"]) for s in steps) or 1
    done_w = sum(int(s["weight"]) for s in steps if s["state"] == "done")
    frac_w = 0.0
    nn = _last_nonnurse_fraction(log_text) if extract_running else None
    current = next((s for s in steps if s["state"] == "active"), None)
    if current and current["id"] == "staffing" and nn:
        frac_w = int(current["weight"]) * (nn[0] / nn[1])
    elif current and current["id"] == "package":
        if ev["snapshot"]:
            frac_w = int(current["weight"]) * (ev["snapshot"][0] / ev["snapshot"][1])
        elif ev["pkg_phase"]:
            frac_w = int(current["weight"]) * (ev["pkg_phase"][0] / max(ev["pkg_phase"][1], 1))

    percent = 100 if (finished and ok) else int(round(100 * (done_w + frac_w) / total))
    percent = max(0, min(99 if not (finished and ok) else 100, percent))
    failed = next((s for s in steps if s["state"] == "failed"), None)
    if finished and ok:
        label = "Finished"
    elif failed:
        label = f"{failed['label']} failed"
    elif current:
        label = f"{current['label']} · {current.get('note') or 'Now'}"
    else:
        label = "Starting"
    return {
        "percent": percent,
        "label": label,
        "steps": steps,
        "started_phases": started,
    }


def _job_path(job_id: str) -> Path:
    safe = "".join(ch for ch in job_id if ch.isalnum())
    return jobs_dir() / f"{safe}.json"


def _log_path(job_id: str) -> Path:
    safe = "".join(ch for ch in job_id if ch.isalnum())
    return jobs_dir() / f"{safe}.log"


def read_job(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return sanitize_action_payload(data) if isinstance(data, dict) else None


def _write_job(payload: dict[str, Any]) -> None:
    """Persist job JSON. Prefer atomic replace; retry/fallback on Windows locks."""
    path = _job_path(str(payload["id"]))
    text = json.dumps(payload, indent=2)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    last_err: BaseException | None = None
    for attempt in range(6):
        try:
            os.replace(str(tmp), str(path))
            return
        except PermissionError as exc:
            last_err = exc
            time.sleep(0.05 * (attempt + 1))
        except OSError as exc:
            # WinError 5 Access is denied (and similar) while a poller holds the file.
            if getattr(exc, "winerror", None) == 5:
                last_err = exc
                time.sleep(0.05 * (attempt + 1))
            else:
                raise
    # Fallback: non-atomic direct write so the job thread can still persist state.
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise PermissionError(
            f"Could not write job file {path} after retries (last={last_err})"
        ) from exc
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def start_dashboard_job(
    *,
    ccn: str,
    intent: str,
    access_mode: str,
    password: str,
    confirm_ccn: str = "",
    confirm_publish: bool = False,
    refresh_data: bool = False,
) -> dict[str, Any]:
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
        return {
            "ok": False,
            "errors": errors,
            "detail": operator_failure_message({"errors": errors}),
            "ccn": ccn_n,
            "intent": intent,
        }
    cold, scaffold = _infer_cold_and_scaffold(ccn_n)
    allow_dirty = True
    argv = build_provision_argv(
        ccn=ccn_n,
        intent=intent,
        access_mode=access_mode,
        cold=cold,
        scaffold_config=scaffold,
        refresh_data=refresh_data,
        allow_dirty_tree=allow_dirty,
        allow_dirty_critical_files=allow_dirty,
    )
    job_id = secrets.token_hex(8)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    prog = progress_from_log("", intent)
    payload = {
        "id": job_id,
        "ok": True,
        "ccn": ccn_n,
        "intent": intent,
        "state": "running",
        "percent": 0,
        "label": "Starting publish" if intent in DEPLOY_INTENTS else "Starting",
        "steps": prog["steps"],
        "detail": "Publishing to Vercel…" if intent in DEPLOY_INTENTS else "Local build started.",
        "started_at": now,
        "updated_at": now,
        "exit_code": None,
    }
    _write_job(payload)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if access_mode == "password_required":
        env[DASHBOARD_PASSWORD_ENV] = secret
    else:
        env.pop(DASHBOARD_PASSWORD_ENV, None)
    env = apply_pbjapp_provision_env(env, root=pbjapp_root())
    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "job_id": job_id,
            "argv": argv,
            "env": env,
            "cwd": str(pbjapp_root()),
            "intent": intent,
            "secret": secret,
        },
        daemon=True,
    )
    thread.start()
    return {
        "ok": True,
        "job_id": job_id,
        "ccn": ccn_n,
        "intent": intent,
        "percent": 0,
        "state": "running",
        "label": "Starting",
        "steps": prog["steps"],
    }


def _run_job(*, job_id: str, argv: list[str], env: dict[str, str], cwd: str, intent: str, secret: str) -> None:
    log_path = _log_path(job_id)
    chunks: list[str] = []
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        job = read_job(job_id) or {"id": job_id}
        job.update({"state": "error", "ok": False, "detail": str(exc), "percent": 0, "exit_code": 1})
        try:
            _write_job(job)
        except OSError:
            pass
        return
    assert proc.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_f:
        for line in proc.stdout:
            chunks.append(line)
            safe = redact_secret_text(line, secret)
            log_f.write(safe)
            log_f.flush()
            blob = redact_secret_text("".join(chunks), secret)
            prog = progress_from_log(blob, intent, finished=False)
            job = read_job(job_id) or {"id": job_id}
            job.update(
                {
                    "state": "running",
                    "percent": prog["percent"],
                    "label": prog["label"],
                    "steps": prog["steps"],
                    "detail": safe.strip()[:240] or job.get("detail"),
                    "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "started_phases": prog["started_phases"],
                }
            )
            try:
                _write_job(job)
            except OSError:
                # Transient Windows lock (e.g. poller reading JSON); keep streaming.
                pass
    code = int(proc.wait())
    blob = redact_secret_text("".join(chunks), secret)
    fail_msg = None
    if code != 0:
        fail_msg = operator_failure_message(
            {"ok": False, "errors": [], "stdout": blob, "stderr": ""}
        )
    prog = progress_from_log(
        blob, intent, finished=True, ok=code == 0, fail_note=fail_msg
    )
    job = read_job(job_id) or {"id": job_id}
    job.update(
        {
            "state": "ok" if code == 0 else "error",
            "ok": code == 0,
            "percent": 100 if code == 0 else prog["percent"],
            "label": "Finished" if code == 0 else prog["label"],
            "steps": [] if code != 0 and not prog.get("started_phases") else prog["steps"],
            "detail": "Finished." if code == 0 else (fail_msg or "Build failed."),
            "exit_code": code,
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "started_phases": prog["started_phases"],
        }
    )
    ccn_n = str(job.get("ccn") or "")
    if ccn_n:
        from data_ops_dashboard import local_view_url

        job["local_view_url"] = local_view_url(ccn_n)
    intent_n = str(job.get("intent") or "")
    if intent_n in DEPLOY_INTENTS and ccn_n:
        try:
            log_blob = _log_path(str(job.get("id") or "")).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            log_blob = ""
        parsed = parse_deployment_url_from_log(log_blob, ccn=ccn_n, intent=intent_n)
        if code == 0:
            job["public_url"] = vercel_public_url_for_intent(ccn_n, intent_n)
            if parsed:
                job["deployment_url"] = parsed
                if not job.get("public_url"):
                    job["public_url"] = parsed
        elif parsed:
            # Soft failure / partial deploy: expose known deployment URL only.
            job["deployment_url"] = parsed
    try:
        _write_job(job)
    except OSError as write_exc:
        job.update(
            {
                "state": "error",
                "ok": False,
                "detail": f"Job status file update failed after retries: {write_exc}",
                "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
        )
        try:
            _write_job(job)
        except OSError:
            pass


def enrich_running_job_progress(job: dict[str, Any]) -> dict[str, Any]:
    """Re-parse the on-disk log so polls stay accurate even if the worker used an older parser."""
    if (job.get("state") or "") != "running":
        return job
    job_id = str(job.get("id") or "")
    intent = str(job.get("intent") or "build")
    log_path = _log_path(job_id)
    if not job_id or not log_path.is_file():
        return job
    try:
        blob = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return job
    prog = progress_from_log(blob, intent, finished=False)
    out = dict(job)
    out["percent"] = prog["percent"]
    out["label"] = prog["label"]
    out["steps"] = prog["steps"]
    out["started_phases"] = prog["started_phases"]
    return out



def vercel_public_url_for_intent(ccn: str, intent: str) -> str | None:
    """Known Vercel alias pattern for staging (-v2) / production."""
    ccn_n = _normalize_ccn(ccn)
    if not ccn_n:
        return None
    intent_n = (intent or "").strip()
    if intent_n == "deploy_staging":
        return f"https://pbj320-{ccn_n}-v2.vercel.app"
    if intent_n == "deploy_production":
        return f"https://pbj320-{ccn_n}.vercel.app"
    return None


_DEPLOYMENT_URL_RE = re.compile(
    r"(?:deployment_url|Production URL|Staging URL|URL \(if alias unchanged\))[:\s]+"
    r"(https://[a-zA-Z0-9.-]+\.vercel\.app\S*)",
    re.IGNORECASE,
)
_HTTPS_VERCEL_RE = re.compile(r"https://[a-zA-Z0-9][a-zA-Z0-9.-]*\.vercel\.app")


def parse_deployment_url_from_log(log_text: str, *, ccn: str = "", intent: str = "") -> str | None:
    """Prefer explicit receipt/print lines; fall back to known alias pattern."""
    blob = log_text or ""
    for m in _DEPLOYMENT_URL_RE.finditer(blob):
        url = m.group(1).rstrip(").,]\"'")
        if url:
            return url
    ccn_n = _normalize_ccn(ccn) if ccn else ""
    found = _HTTPS_VERCEL_RE.findall(blob)
    if ccn_n:
        for url in reversed(found):
            if f"pbj320-{ccn_n}" in url:
                return url
    if found:
        return found[-1]
    return vercel_public_url_for_intent(ccn, intent)


def job_public_view(job: dict[str, Any]) -> dict[str, Any]:
    job = enrich_running_job_progress(job)
    intent = str(job.get("intent") or "")
    public_url = job.get("public_url")
    deployment_url = job.get("deployment_url")
    # Backfill alias for successful publishes (covers jobs finished before URL fields existed).
    if (
        not public_url
        and (job.get("state") == "ok" or job.get("ok") is True)
        and intent in DEPLOY_INTENTS
    ):
        public_url = vercel_public_url_for_intent(str(job.get("ccn") or ""), intent)
    return sanitize_action_payload(
        {
            "id": job.get("id"),
            "ccn": job.get("ccn"),
            "intent": job.get("intent"),
            "state": job.get("state"),
            "percent": int(job.get("percent") or 0),
            "label": job.get("label"),
            "steps": job.get("steps") or [],
            "detail": job.get("detail"),
            "ok": job.get("ok"),
            "exit_code": job.get("exit_code"),
            "local_view_url": job.get("local_view_url"),
            "local_view_ok": job.get("local_view_ok"),
            "local_view_detail": job.get("local_view_detail"),
            "public_url": public_url,
            "deployment_url": deployment_url,
        }
    )
