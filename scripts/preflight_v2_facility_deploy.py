#!/usr/bin/env python3
"""
Pre-deploy validation for V2 superdynamic facility bundles.

Catches cold-deploy gaps seen on CCN 315128 before Vercel production deploy.

Usage:
  python scripts/preflight_v2_facility_deploy.py 315128
  python scripts/preflight_v2_facility_deploy.py 315128 --password Evans320
  python scripts/preflight_v2_facility_deploy.py 315128 --check-vercel-env
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REQUIRED_PARTIALS = (
    "templates/partials/superdynamic_url_macros.html",
    "templates/partials/v2/guided_nav.html",
)

REQUIRED_CONFIG = (
    "config/citation_severity_rank.json",
    "config/nonnurse_staff_groups.json",
)

REPO_CANONICAL_RUNTIME = (
    "config/nonnurse_staff_groups.json",
    "config/citation_severity_rank.json",
    "config/citation_pbj_bridge.json",
    "config/citation_topic_registry.json",
    "config/citation_narrative_role_map.json",
    "superdynamic/security_http.py",
    "static/data/interval_quarter_mapping.json",
    "templates/superdynamic_methodology.html",
    ".vercelignore",
)

REQUIRED_PY = (
    "cms_data_paths.py",
    "data_path_resolver.py",
    "citation_lib.py",
    "citation_pbj_bridge.py",
    "nonnurse_staffing_lib.py",
    f"facility_{{ccn}}_superdynamic_dashboard.py",
)

REQUIRED_STATIC_JS = (
    "static/js/pbj_v2_dashboard_extras.js",
    "static/js/superdynamic_utils.js",
    "static/js/pbj320_snapshot_report.js",
    "static/js/pbj_rb3_ai_bridge.js",
)


def _vercel_cmd() -> list[str]:
    resolved = shutil.which("vercel")
    if resolved:
        return [resolved]
    npx = shutil.which("npx")
    if npx:
        return [npx, "vercel"]
    return ["vercel"]


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return int(p.returncode), out.strip()


def _check_file(deploy: Path, rel: str, ccn: str) -> tuple[bool, str]:
    rel = rel.replace("{ccn}", ccn)
    p = deploy / rel
    if p.is_file():
        return True, f"OK   {rel} ({p.stat().st_size} B)"
    return False, f"FAIL {rel} missing"


def _check_v2_partials_layout(deploy: Path) -> tuple[bool, str]:
    wrong = deploy / "templates" / "partials" / "guided_nav.html"
    right = deploy / "templates" / "partials" / "v2" / "guided_nav.html"
    if wrong.is_file() and not right.is_file():
        return False, "FAIL v2 partials copied to templates/partials/ instead of partials/v2/"
    if not right.is_file():
        return False, "FAIL templates/partials/v2/guided_nav.html missing"
    return True, "OK   v2 partials under templates/partials/v2/"


def _check_legacy_entrypoint_regression(deploy: Path, ccn: str) -> tuple[bool, str]:
    from deployment_entrypoint_guard import legacy_entrypoint_regression_message

    msg = legacy_entrypoint_regression_message(deploy, ccn)
    if msg:
        return False, f"FAIL {msg}"
    return True, "OK   no legacy entrypoint regression detected"


def _check_vercel_json(deploy: Path, ccn: str) -> tuple[bool, str]:
    path = deploy / "vercel.json"
    if not path.is_file():
        return False, "FAIL vercel.json missing"
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"FAIL vercel.json invalid JSON: {exc}"
    env = cfg.get("env") or {}
    tpl = str(env.get("PBJ_SUPERDYNAMIC_TEMPLATE") or "").strip().lower()
    if tpl not in ("v2", "superdynamic_v2", "superdynamic_dashboard_v2"):
        return False, f"FAIL PBJ_SUPERDYNAMIC_TEMPLATE={tpl!r} (expected v2)"
    builds = cfg.get("builds") or []
    app = f"facility_{ccn}_superdynamic_dashboard.py"
    if not any(app in str(b.get("src") or "") for b in builds):
        return False, f"FAIL vercel.json builds.src must reference {app}"
    return True, f"OK   vercel.json v2 entrypoint {app}"


def _check_vercel_password(ccn: str, expected: str) -> tuple[bool, str]:
    deploy = _ROOT / "deployments" / f"pbj320-{ccn}"
    code, out = _run(_vercel_cmd() + ["env", "ls"], cwd=deploy)
    if code != 0:
        return False, f"WARN vercel env ls failed (not linked?): {out[:120]}"
    if "PBJ_DASHBOARD_PASSWORD" not in out:
        return False, "FAIL PBJ_DASHBOARD_PASSWORD not set on Vercel project"
    if expected:
        # Cannot read encrypted value; verify via local hash hint only in docs.
        return True, f"OK   PBJ_DASHBOARD_PASSWORD present (expected {expected!r} — confirm via login smoke)"
    return True, "OK   PBJ_DASHBOARD_PASSWORD present on Vercel"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ccn")
    parser.add_argument("--password", default="", help="Expected dashboard password (for checklist note)")
    parser.add_argument(
        "--check-vercel-env",
        action="store_true",
        help="Run vercel env ls from deployment folder",
    )
    parser.add_argument(
        "--skip-repo-template-js",
        action="store_true",
        help="Skip repo-root inline JS gate (not recommended)",
    )
    args = parser.parse_args()
    ccn = str(args.ccn).strip().zfill(6)
    deploy = _ROOT / "deployments" / f"pbj320-{ccn}"
    if not deploy.is_dir():
        print(f"ERROR: {deploy} missing — run create_vercel_deployment.py then bootstrap", file=sys.stderr)
        return 1

    print(f"\n=== V2 preflight: CCN {ccn} ===\n")
    fails = 0

    for rel in REQUIRED_PARTIALS + REQUIRED_CONFIG:
        ok, msg = _check_file(deploy, rel, ccn)
        print(msg)
        fails += 0 if ok else 1

    for rel in REPO_CANONICAL_RUNTIME:
        p = _ROOT / rel
        if p.is_file():
            print(f"OK   repo {rel} ({p.stat().st_size} B)")
        else:
            print(f"FAIL repo {rel} missing (bootstrap will require --ref fallback)")
            fails += 1

    ok, msg = _check_v2_partials_layout(deploy)
    print(msg)
    fails += 0 if ok else 1

    ok, msg = _check_legacy_entrypoint_regression(deploy, ccn)
    print(msg)
    fails += 0 if ok else 1

    for rel in REQUIRED_PY + REQUIRED_STATIC_JS:
        ok, msg = _check_file(deploy, rel, ccn)
        print(msg)
        fails += 0 if ok else 1

    ok, msg = _check_vercel_json(deploy, ccn)
    print(msg)
    fails += 0 if ok else 1

    tpl = deploy / "templates" / "superdynamic_dashboard_v2.html"
    if not tpl.is_file():
        print("FAIL templates/superdynamic_dashboard_v2.html missing")
        fails += 1
    else:
        for name, script in (
            ("inline JS", "scripts/check_v2_inline_js.py"),
            ("evidence layout", "scripts/check_v2_evidence_layout.py"),
        ):
            code, out = _run(
                [sys.executable, str(_ROOT / script), "--template", str(tpl)],
                cwd=_ROOT,
            )
            status = "OK  " if code == 0 else "FAIL"
            print(f"{status} {name} gate on deployment template")
            if code != 0:
                print(out[:400])
                fails += 1

        code, out = _run(
            [
                sys.executable,
                str(_ROOT / "scripts/check_v2_static_js_bundle.py"),
                "--template",
                str(tpl),
                "--deploy-dir",
                str(deploy),
            ],
            cwd=_ROOT,
        )
        print("OK   static JS bundle" if code == 0 else f"FAIL static JS bundle\n{out[:400]}")
        fails += 0 if code == 0 else 1

    for script, label in (
        (["scripts/check_v2_deployment_import.py", ccn, "--deploy-dir", str(deploy)], "import"),
        (["scripts/check_v2_deployment_bootstrap.py", ccn, "--deploy-dir", str(deploy)], "bootstrap"),
    ):
        code, out = _run([sys.executable, str(_ROOT / script[0])] + script[1:], cwd=_ROOT)
        print(f"{'OK  ' if code == 0 else 'FAIL'} deployment {label} check")
        if code != 0:
            print(out[:400])
            fails += 1

    repo_tpl = _ROOT / "templates" / "superdynamic_dashboard_v2.html"
    if repo_tpl.is_file() and not args.skip_repo_template_js:
        for name, script in (
            ("repo inline JS", "scripts/check_v2_inline_js.py"),
            ("repo evidence layout", "scripts/check_v2_evidence_layout.py"),
        ):
            code, out = _run(
                [sys.executable, str(_ROOT / script), "--template", str(repo_tpl)],
                cwd=_ROOT,
            )
            status = "OK  " if code == 0 else "FAIL"
            print(f"{status} {name} gate on repo canonical template")
            if code != 0:
                print(out[:400])
                fails += 1
    elif not repo_tpl.is_file():
        print("FAIL templates/superdynamic_dashboard_v2.html missing at repo root")
        fails += 1

    if args.check_vercel_env:
        ok, msg = _check_vercel_password(ccn, args.password.strip())
        print(msg)
        fails += 0 if ok else 1
    elif args.password:
        print(f"NOTE password expected: {args.password!r} (pass --check-vercel-env to verify Vercel env)")

    print(f"\n{'PASS' if fails == 0 else 'FAIL'}: {fails} blocking issue(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
