#!/usr/bin/env python3
"""
Repeatable Vercel deploy for facility dashboards under deployments/pbj320-<CCN>/.

Safety:
  - Production deploy (vercel --prod) only runs with explicit --confirm-deploy.
  - Never prints .env.local or other secret files.
  - Assumes Vercel CLI is installed and you are already logged in (vercel login).

Typical flow:
  1. Build/refresh the bundle (no Vercel):
       python scripts/deploy_vercel_facility.py 335513 --package
  2. Deploy production when you explicitly choose to (after review):
       python scripts/deploy_vercel_facility.py 335513 --confirm-deploy
     (Or package + deploy in one step: add --package before --confirm-deploy.)

  Deploy without repackaging (folder already fresh):
       python scripts/deploy_vercel_facility.py 335513 --confirm-deploy --no-package
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _vercel_argv_prefix() -> list[str]:
    """
    Return argv prefix to invoke the Vercel CLI.

    On Windows, subprocess cannot reliably start bare ``vercel`` when only
    ``vercel.cmd`` exists in npm's global folder; ``shutil.which`` resolves
    the full path (including ``.cmd``). If the CLI is missing from PATH
    (e.g. GUI-launched Python), callers still get a clear error.
    """
    resolved = shutil.which("vercel")
    if resolved:
        return [resolved]
    npx = shutil.which("npx")
    if npx:
        return [npx, "vercel"]
    print(
        "ERROR: Vercel CLI not found. Install it (npm i -g vercel) and ensure "
        "the npm global bin directory is on PATH, or run this script from the "
        "same shell where `vercel` works.",
        file=sys.stderr,
    )
    raise SystemExit(127)


def _run(cmd: list[str], cwd: Path) -> int:
    """Run command; stream to stdout/stderr. Returns process return code."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=str(cwd))
    return int(p.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package and/or deploy a facility dashboard to Vercel."
    )
    parser.add_argument(
        "provnum",
        help="Six-digit CCN (e.g. 335513)",
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help="Run create_vercel_deployment.py for this CCN (refreshes app, template, CSVs, vercel.json).",
    )
    parser.add_argument(
        "--no-package",
        action="store_true",
        help="Skip packaging; only run Vercel CLI steps (requires --confirm-deploy).",
    )
    parser.add_argument(
        "--confirm-deploy",
        action="store_true",
        help="Required to run `vercel link` and `vercel --prod`. Without this flag, no deploy is attempted.",
    )
    args = parser.parse_args()
    prov = str(args.provnum).strip().zfill(6)
    if not prov.isdigit() or len(prov) != 6:
        print("ERROR: provnum must be a 6-digit CCN.", file=sys.stderr)
        return 2

    root = _project_root()
    deploy_dir = root / "deployments" / f"pbj320-{prov}"
    if not deploy_dir.is_dir():
        print(f"ERROR: Missing {deploy_dir}", file=sys.stderr)
        return 1

    if not args.package and not args.confirm_deploy:
        print(
            "ERROR: pass at least one of:\n"
            "  --package          refresh deployments/pbj320-<CCN>/ (create_vercel_deployment.py)\n"
            "  --confirm-deploy   run vercel link + vercel --prod (requires prior package unless --no-package)\n"
            "  or both for package-then-deploy.",
            file=sys.stderr,
        )
        return 2

    if args.no_package and not args.confirm_deploy:
        print("ERROR: --no-package only applies with --confirm-deploy (deploy without repackaging).", file=sys.stderr)
        return 2

    if args.package and args.no_package:
        print("ERROR: cannot use --package and --no-package together.", file=sys.stderr)
        return 2

    do_package = bool(args.package)

    if do_package:
        create_script = root / "create_vercel_deployment.py"
        if not create_script.is_file():
            print(f"ERROR: {create_script} not found.", file=sys.stderr)
            return 1
        code = _run([sys.executable, str(create_script), prov], cwd=root)
        if code != 0:
            print("ERROR: Packaging failed.", file=sys.stderr)
            return code

    if not args.confirm_deploy:
        print(
            f"\nPackaging finished for {prov}. No Vercel deploy (no --confirm-deploy).\n"
            f"When you approve production deploy:\n"
            f"  python scripts/deploy_vercel_facility.py {prov} --confirm-deploy\n"
            f"Or refresh bundle and deploy:\n"
            f"  python scripts/deploy_vercel_facility.py {prov} --package --confirm-deploy"
        )
        return 0

    # Deploy: do not read or print .env.local
    import_check = root / "scripts" / "check_v2_deployment_import.py"
    if import_check.is_file():
        code = _run(
            [sys.executable, str(import_check), prov, "--deploy-dir", str(deploy_dir)],
            cwd=root,
        )
        if code != 0:
            print(
                "ERROR: deployment import check failed — fix the bundle before --confirm-deploy.",
                file=sys.stderr,
            )
            return code

    vercel = _vercel_argv_prefix()
    code = _run(
        vercel + ["link", f"--project=pbj320-{prov}", "--yes"],
        cwd=deploy_dir,
    )
    if code != 0:
        return code
    code = _run(vercel + ["--prod", "--yes"], cwd=deploy_dir)
    if code != 0:
        return code
    print(f"\nProduction URL (if alias unchanged): https://pbj320-{prov}.vercel.app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
