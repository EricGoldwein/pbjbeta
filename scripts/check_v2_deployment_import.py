#!/usr/bin/env python3
"""Smoke-test that a Vercel deployment bundle can import its Flask entrypoint.

Simulates Lambda layout: only the deployment folder is on sys.path (no repo root).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _entry_module(deploy_dir: Path) -> str:
    for py in sorted(deploy_dir.glob("facility_*_superdynamic_dashboard.py")):
        return py.stem
    for py in sorted(deploy_dir.glob("facility_*_flask_app.py")):
        return py.stem
    raise FileNotFoundError(f"No facility entrypoint in {deploy_dir}")


def run_check(deploy_dir: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    deploy_dir = deploy_dir.resolve()
    if not deploy_dir.is_dir():
        return False, [f"Missing deployment dir: {deploy_dir}"]

    required = [
        deploy_dir / "pbj_fea_loader.py",
        deploy_dir / "_pbj_canonical_facility_ein_employee_analytics.py",
        deploy_dir / "facility_ein_employee_analytics.py",
    ]
    entry_py = next(deploy_dir.glob("facility_*_superdynamic_dashboard.py"), None)
    entry_text = entry_py.read_text(encoding="utf-8", errors="replace") if entry_py else ""
    if "api/report_builder_v3/preview" in entry_text:
        required.extend(
            [
                deploy_dir / "report_builder_v3.py",
                deploy_dir / "report_item_registry.py",
                deploy_dir / "report_findings_engine.py",
            ]
        )
    for path in required:
        if not path.is_file():
            errors.append(f"Missing bundled module: {path.name}")

    if errors:
        return False, errors

    entry = _entry_module(deploy_dir)
    old_path = list(sys.path)
    old_cwd = os.getcwd()
    deploy_str = str(deploy_dir)
    try:
        os.chdir(deploy_dir)
        if deploy_str in sys.path:
            sys.path.remove(deploy_str)
        sys.path.insert(0, deploy_str)
        importlib.import_module("facility_ein_employee_analytics")
        importlib.import_module(entry)
    except Exception as exc:
        errors.append(f"import {entry}: {exc}")
    finally:
        os.chdir(old_cwd)
        if sys.path and sys.path[0] == deploy_str:
            sys.path.pop(0)
        elif deploy_str in sys.path:
            sys.path.remove(deploy_str)
        for name in list(sys.modules):
            if name == entry or name == "facility_ein_employee_analytics":
                sys.modules.pop(name, None)
            elif name.startswith("facility_") and name.endswith("_superdynamic_dashboard"):
                sys.modules.pop(name, None)
            elif name.startswith("_pbj_canonical_"):
                sys.modules.pop(name, None)

    return (not errors, errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "provnum",
        nargs="?",
        help="Six-digit CCN (default: infer from --deploy-dir)",
    )
    parser.add_argument(
        "--deploy-dir",
        type=Path,
        help="Path to deployments/pbj320-<CCN>/",
    )
    args = parser.parse_args()
    if args.deploy_dir:
        deploy_dir = args.deploy_dir
    elif args.provnum:
        prov = str(args.provnum).strip().zfill(6)
        deploy_dir = _ROOT / "deployments" / f"pbj320-{prov}"
    else:
        parser.error("pass provnum or --deploy-dir")
        return 2

    ok, errors = run_check(deploy_dir)
    if ok:
        print(f"PASS: deployment import OK ({deploy_dir.name})")
        return 0
    print(f"FAIL: deployment import check ({deploy_dir.name})", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
