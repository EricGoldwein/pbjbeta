#!/usr/bin/env python3
"""Pre-deploy smoke: Vercel bundle can bootstrap facility PBJ data on a read-only filesystem.

Catches failures that import-only checks miss (e.g. stray deployments/ subfolder causing
get_facility_folder() mkdir on Lambda, empty global_df, wrong CCN).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_CSV_SUFFIXES = (
    "_complete_data.csv",
    "_provider_info_data.csv",
)


def _entry_module(deploy_dir: Path) -> str:
    for py in sorted(deploy_dir.glob("facility_*_superdynamic_dashboard.py")):
        return py.stem
    for py in sorted(deploy_dir.glob("facility_*_flask_app.py")):
        return py.stem
    raise FileNotFoundError(f"No facility entrypoint in {deploy_dir}")


def _infer_provnum(deploy_dir: Path, provnum: str | None) -> str:
    if provnum:
        return str(provnum).strip().zfill(6)
    name = deploy_dir.name
    if name.startswith("pbj320-") and name[7:].isdigit():
        return name[7:].zfill(6)
    raise ValueError(f"Cannot infer CCN from {deploy_dir}; pass provnum explicitly.")


_REQUIRED_URL_MACRO_EXPORTS = (
    "pbj_facility_page_href",
    "pbj_premium_page_href",
    "pbj_api_href",
)


def _apply_vercel_json_env(deploy_dir: Path) -> None:
    """Mirror Vercel runtime env from bundle vercel.json (template + V3 panes flags)."""
    path = deploy_dir / "vercel.json"
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    env = data.get("env")
    if not isinstance(env, dict):
        return
    for key, val in env.items():
        if val is not None and str(key).strip():
            os.environ[str(key)] = str(val)


def _check_url_macros(deploy_dir: Path) -> list[str]:
    """guided_nav.html imports shared URL macros — stale bundles 500 at render time."""
    errors: list[str] = []
    path = deploy_dir / "templates" / "partials" / "superdynamic_url_macros.html"
    if not path.is_file():
        errors.append(f"Missing {path.relative_to(deploy_dir)}")
        return errors
    text = path.read_text(encoding="utf-8", errors="replace")
    for name in _REQUIRED_URL_MACRO_EXPORTS:
        needle = f"macro {name}"
        if needle not in text:
            errors.append(
                f"superdynamic_url_macros.html missing {needle!r} — "
                "sync repo templates/partials/superdynamic_url_macros.html before deploy."
            )
    return errors


def _check_index_render(mod, prov: str) -> list[str]:
    """Import/bootstrap checks miss Jinja render failures (UndefinedError on /)."""
    errors: list[str] = []
    app = getattr(mod, "app", None)
    if app is None:
        errors.append("entry module has no Flask app")
        return errors
    try:
        with app.test_client() as client:
            resp = client.get("/")
    except Exception as exc:
        errors.append(f"GET / template render failed: {exc}")
        return errors
    if resp.status_code != 200:
        body = (resp.get_data(as_text=True) or "")[:240].replace("\n", " ")
        errors.append(f"GET / returned HTTP {resp.status_code} (expected 200): {body}")
        return errors
    html = resp.get_data(as_text=True) or ""
    if "Internal Server Error" in html:
        errors.append("GET / body contains Internal Server Error")
    if prov and prov not in html and "Unknown Facility" in html:
        errors.append(f"GET / rendered Unknown Facility for CCN {prov}")
    return errors


def _check_bundle_hygiene(deploy_dir: Path, prov: str) -> list[str]:
    errors: list[str] = []
    nested = deploy_dir / "deployments"
    if nested.is_dir():
        errors.append(
            "Forbidden nested deployments/ inside bundle — triggers repo-path logic on "
            "Vercel read-only filesystem. Remove it and add deployments/ to .vercelignore."
        )
    for suffix in _REQUIRED_CSV_SUFFIXES:
        path = deploy_dir / f"facility_{prov}{suffix}"
        if not path.is_file():
            errors.append(f"Missing bundle CSV: {path.name}")
    vercel_ignore = deploy_dir / ".vercelignore"
    if vercel_ignore.is_file():
        text = vercel_ignore.read_text(encoding="utf-8", errors="replace")
        if "deployments/" not in text:
            errors.append(
                ".vercelignore should list deployments/ so nested repo folders never ship."
            )
    return errors


def _readonly_deployments_mkdir(self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False):
    """Simulate Lambda: mkdir under bundle deployments/ is read-only."""
    parts = self.parts
    if "deployments" in parts:
        deploy_idx = None
        for i, part in enumerate(parts):
            if part == "deployments" and i > 0:
                deploy_idx = i
                break
        if deploy_idx is not None:
            raise OSError(30, "Read-only file system (simulated Vercel)", str(self))
    return _ORIG_PATH_MKDIR(self, mode, parents=parents, exist_ok=exist_ok)


_ORIG_PATH_MKDIR = Path.mkdir


def run_check(deploy_dir: Path, provnum: str | None = None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    deploy_dir = deploy_dir.resolve()
    if not deploy_dir.is_dir():
        return False, [f"Missing deployment dir: {deploy_dir}"]

    try:
        prov = _infer_provnum(deploy_dir, provnum)
    except ValueError as exc:
        return False, [str(exc)]

    errors.extend(_check_bundle_hygiene(deploy_dir, prov))
    errors.extend(_check_url_macros(deploy_dir))
    if errors:
        return False, errors

    entry = _entry_module(deploy_dir)
    deploy_str = str(deploy_dir)
    old_path = list(sys.path)
    old_cwd = os.getcwd()
    old_env = dict(os.environ)
    mod = None

    try:
        os.chdir(deploy_dir)
        if deploy_str in sys.path:
            sys.path.remove(deploy_str)
        sys.path.insert(0, deploy_str)

        os.environ["PBJ_FACILITY_CCN"] = prov
        os.environ["PBJ_PROVNUM"] = prov
        os.environ.setdefault("VERCEL", "1")
        _apply_vercel_json_env(deploy_dir)
        # Template render smoke must not depend on Vercel password env.
        os.environ.pop("PBJ_DASHBOARD_PASSWORD", None)
        os.environ.pop("PBJ_DASHBOARD_PASSWORD_ALIASES", None)

        with patch.object(Path, "mkdir", _readonly_deployments_mkdir):
            mod = importlib.import_module(entry)
            if not hasattr(mod, "ensure_data_loaded"):
                errors.append(f"{entry} has no ensure_data_loaded()")
                return False, errors

            mod.global_df = None
            mod._data_initialized = False
            mod.ensure_data_loaded()

            gdf = getattr(mod, "global_df", None)
            if gdf is None or getattr(gdf, "empty", True):
                errors.append(
                    "ensure_data_loaded() left global_df empty under simulated Vercel layout "
                    "(read-only deployments/ mkdir). Dashboard would show Unknown Facility and no charts."
                )
                return False, errors

            row_count = len(gdf)
            if row_count < 100:
                errors.append(f"global_df suspiciously small ({row_count} rows)")

            if "PROVNUM" in gdf.columns:
                seen = {str(v).strip().zfill(6) for v in gdf["PROVNUM"].dropna().unique()}
                if prov not in seen:
                    errors.append(f"global_df PROVNUM mismatch: expected {prov}, saw {sorted(seen)[:5]}")

            if "CY_Qtr" in gdf.columns:
                quarters = gdf["CY_Qtr"].dropna().unique()
                if len(quarters) < 4:
                    errors.append(f"global_df has too few quarters ({len(quarters)})")

            errors.extend(_check_index_render(mod, prov))

    except Exception as exc:
        errors.append(f"bootstrap smoke failed: {exc}")
    finally:
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(old_env)
        if sys.path and sys.path[0] == deploy_str:
            sys.path.pop(0)
        elif deploy_str in sys.path:
            sys.path.remove(deploy_str)
        sys.path[:] = old_path
        if mod is not None:
            name = mod.__name__
            sys.modules.pop(name, None)

    return (not errors, errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provnum", nargs="?", help="Six-digit CCN")
    parser.add_argument("--deploy-dir", type=Path, help="Path to deployments/pbj320-<CCN>/")
    args = parser.parse_args()

    if args.deploy_dir:
        deploy_dir = args.deploy_dir
    elif args.provnum:
        prov = str(args.provnum).strip().zfill(6)
        deploy_dir = _ROOT / "deployments" / f"pbj320-{prov}"
    else:
        parser.error("pass provnum or --deploy-dir")
        return 2

    ok, errors = run_check(deploy_dir, args.provnum)
    if ok:
        print(f"PASS: deployment bootstrap OK ({deploy_dir.name})")
        return 0
    print(f"FAIL: deployment bootstrap check ({deploy_dir.name})", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
