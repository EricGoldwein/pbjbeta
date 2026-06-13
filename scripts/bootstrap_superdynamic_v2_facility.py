#!/usr/bin/env python3
"""
Bootstrap a cold V2 superdynamic facility bundle from a reference deployment.

Usage:
  python scripts/bootstrap_superdynamic_v2_facility.py 315128 --ref 315461

Preserves facility-specific data slices (CSV/parquet/EIN) already in the target
deploy folder unless --force is passed for infrastructure overwrites.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# Infrastructure copied from reference (not facility_* data slices).
COPY_REL_PATHS: tuple[str, ...] = (
    "superdynamic",
    "admin_turnover_review.py",
    "facility_risk_signals.py",
    "facility_deploy_meta.py",
    "pbj_case_mix_cmi.py",
    "pbj_chow_facility.py",
    "pbj_premium_demo_section.py",
    "pbj_ai_config.py",
    "geo_distribution_lib.py",
    "report_builder_v3.py",
    "report_findings_engine.py",
    "report_item_registry.py",
    "pbj_fea_loader.py",
    "entity_longitudinal_metrics.py",
    "geo_nursing_cmi_quarterly.csv",
    "geo_nursing_cmi_quarterly.meta.json",
    "facility_quarterly_metrics.parquet",
    "cms_region_quarterly_metrics.csv",
    "cms_data_paths.py",
    "data_path_resolver.py",
    "citation_lib.py",
    "citation_pbj_bridge.py",
    "citation_taxonomy.py",
    "citation_date_confidence.py",
    "nonnurse_staffing_lib.py",
    "pbj_staffing_normalize.py",
    "file_path_utils.py",
    "prov_info.py",
    "prov_info_quarter_map.py",
    "facility_ein_lib.py",
    "facility_ein_employee_analytics.py",
    "_pbj_canonical_facility_ein_employee_analytics.py",
    "facility_report_lib.py",
    ".vercelignore",
    "requirements.txt",
    "ownership/chow_index.json",
    "ownership/entity_lookup.csv",
    "static/data/interval_quarter_mapping.json",
    "templates/report_builder_v3_standalone.html",
    "templates/superdynamic_methodology.html",
    "templates/partials/superdynamic_url_macros.html",
)

CONFIG_GLOBS: tuple[str, ...] = ("config/*.json",)

# Written at deploy time; do not copy from reference bundle during bootstrap.
CONFIG_SKIP_NAMES: frozenset[str] = frozenset({"facility_deployed_at.json"})

STATIC_JS_FILES: tuple[str, ...] = (
    "superdynamic_utils.js",
    "pbj_facility_display_name.js",
    "pbj_red_flag_quarter_resolver.js",
    "pbj_v2_dashboard_extras.js",
    "pbj_v2_benchmark_workspace.js",
    "pbj_v2_peer_comparison.js",
    "pbj_v2_perf_debug.js",
    "pbj_v2_geo_distribution.js",
    "pbj_work_date_bridge.js",
    "pbj_facility_events.js",
    "pbj_ein_roster_lifecycle.js",
    "pbj_sustained_work_flags.js",
    "pbj_v2_chart_scope_toolbar.js",
    "pbj_sparkline.js",
    "pbj_report_builder_v3.js",
    "pbj_report_item_registry.js",
    "pbj320_snapshot_report.js",
    "pbj_rb3_ai_bridge.js",
)

REQUIRED_V2_RUNTIME: tuple[str, ...] = (
    "facility_{ccn}_superdynamic_dashboard.py",
    "vercel.json",
    "templates/superdynamic_dashboard_v2.html",
    "templates/partials/superdynamic_url_macros.html",
    "templates/partials/v2/guided_nav.html",
    "config/nonnurse_staff_groups.json",
    "config/citation_severity_rank.json",
    "cms_data_paths.py",
    "data_path_resolver.py",
    "nonnurse_staffing_lib.py",
    "citation_lib.py",
    "static/js/pbj_v2_dashboard_extras.js",
    "static/js/superdynamic_utils.js",
    "ownership/chow_index.json",
    "superdynamic/security_http.py",
)

FACILITY_DATA_GLOB = re.compile(
    r"^facility_\d{6}_(complete_data|nonnurse_daily|citations|provider_info_data|ein_.*)\."
)


def _normalize_ccn(raw: str) -> str:
    s = str(raw).strip().zfill(6)
    if not s.isdigit() or len(s) != 6:
        raise ValueError(f"CCN must be 6 digits, got {raw!r}")
    return s


def _deploy_dir(root: Path, ccn: str) -> Path:
    return root / "deployments" / f"pbj320-{ccn}"


def _is_facility_data(path: Path) -> bool:
    return bool(FACILITY_DATA_GLOB.match(path.name))


def _resolve_source(root: Path, ref: Path, rel: str) -> Path | None:
    """Prefer repo-root sources; fall back to reference deployment bundle."""
    for base in (root, ref):
        p = base / rel
        if p.exists():
            return p
    return None


def _copy_canonical_v2_main_template(root: Path, ref: Path, target: Path, *, force: bool) -> bool:
    """Repo ``templates/superdynamic_dashboard_v2.html`` is canonical; ref is fallback only."""
    src = _resolve_source(root, ref, "templates/superdynamic_dashboard_v2.html")
    if src is None:
        print("  [WARN] templates/superdynamic_dashboard_v2.html missing in repo and ref")
        return False
    dst = target / "templates/superdynamic_dashboard_v2.html"
    origin = "repo" if str(src.resolve()).startswith(str(root.resolve())) else "ref"
    _copy_tree(src, dst, force=force)
    print(f"  [OK] templates/superdynamic_dashboard_v2.html (from {origin})")
    return True


def _copy_tree(src: Path, dst: Path, *, force: bool) -> None:
    if src.is_dir():
        if dst.exists() and not force:
            print(f"  [SKIP dir] {dst.relative_to(dst.parents[2])} (exists; use --force)")
            return
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        if dst.exists() and not force:
            print(f"  [SKIP file] {dst.name} (exists; use --force)")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _adapt_superdynamic_entrypoint(ref_app: Path, dst_app: Path, ccn: str, ref_ccn: str) -> None:
    text = ref_app.read_text(encoding="utf-8")
    text = text.replace(ref_ccn, ccn)
    text = re.sub(
        r"_REPORT_BUILDER_V3_ALLOWED_CCNS = frozenset\(\{[^}]+\}\)",
        f'_REPORT_BUILDER_V3_ALLOWED_CCNS = frozenset({{"{ccn}"}})',
        text,
    )
    dst_app.write_text(text, encoding="utf-8")


def _adapt_vercel_json(ref_json: Path, dst_json: Path, ccn: str, ref_ccn: str) -> None:
    raw = ref_json.read_text(encoding="utf-8").replace(ref_ccn, ccn)
    dst_json.write_text(json.dumps(json.loads(raw), indent=2) + "\n", encoding="utf-8")


def _manifest_diff(root: Path, target: Path, ref: Path, ccn: str) -> list[str]:
    lines: list[str] = []
    for rel in REQUIRED_V2_RUNTIME:
        rel = rel.replace("{ccn}", ccn)
        tp = target / rel
        rp = ref / rel
        if tp.is_file():
            ts, rs = tp.stat().st_size, (rp.stat().st_size if rp.is_file() else 0)
            lines.append(f"  OK   {rel} ({ts / 1024:.1f} KB; ref {rs / 1024:.1f} KB)")
        else:
            lines.append(f"  MISS {rel}")
    return lines


def run_bootstrap(
    root: Path,
    ccn: str,
    ref_ccn: str,
    *,
    force: bool = False,
) -> int:
    target = _deploy_dir(root, ccn)
    ref = _deploy_dir(root, ref_ccn)
    if not ref.is_dir():
        print(f"ERROR: reference bundle missing: {ref}", file=sys.stderr)
        return 1
    target.mkdir(parents=True, exist_ok=True)

    repo_tpl = root / "templates" / "superdynamic_dashboard_v2.html"
    if repo_tpl.is_file():
        for gate in ("scripts/check_v2_inline_js.py", "scripts/check_v2_evidence_layout.py"):
            import subprocess

            r = subprocess.run(
                [sys.executable, str(root / gate), "--template", str(repo_tpl)],
                cwd=str(root),
            )
            if r.returncode != 0:
                print(f"ERROR: repo canonical template failed {gate}", file=sys.stderr)
                return 1

    print(f"\n=== Bootstrap V2 superdynamic: {ccn} from ref {ref_ccn} ===")
    print(f"Target: {target}")
    print(f"Reference: {ref}\n")

    _copy_canonical_v2_main_template(root, ref, target, force=force)

    for rel in COPY_REL_PATHS:
        src = _resolve_source(root, ref, rel)
        if src is None:
            print(f"  [WARN] missing ref/repo source: {rel}")
            continue
        dst = target / rel
        _copy_tree(src, dst, force=force)
        print(f"  [OK] {rel}")

    # partials/v2 tree (must land under partials/v2/, not partials/)
    v2_src = root / "templates" / "partials" / "v2"
    if not v2_src.is_dir():
        v2_src = ref / "templates" / "partials" / "v2"
    v2_dst = target / "templates" / "partials" / "v2"
    if v2_src.is_dir():
        if v2_dst.exists() and force:
            shutil.rmtree(v2_dst)
        if not v2_dst.exists() or force:
            if v2_dst.exists():
                shutil.rmtree(v2_dst)
            shutil.copytree(v2_src, v2_dst)
            print("  [OK] templates/partials/v2/**")

    for pattern in CONFIG_GLOBS:
        copied: set[str] = set()
        for base in (root, ref):
            for src in sorted(base.glob(pattern)):
                if src.name in CONFIG_SKIP_NAMES:
                    continue
                if src.name in copied and not force:
                    continue
                dst = target / src.relative_to(base)
                _copy_tree(src, dst, force=force)
                copied.add(src.name)
                origin = "repo" if base is root else "ref"
                print(f"  [OK] {src.relative_to(base).as_posix()} (from {origin})")

    for name in STATIC_JS_FILES:
        for base in (root / "static" / "js", ref / "static" / "js"):
            src = base / name
            if src.is_file():
                dst = target / "static" / "js" / name
                _copy_tree(src, dst, force=force)
                print(f"  [OK] static/js/{name}")
                break
        else:
            print(f"  [WARN] static/js/{name} not found in ref or repo")

    ref_app = ref / f"facility_{ref_ccn}_superdynamic_dashboard.py"
    if not ref_app.is_file():
        print(f"ERROR: reference entrypoint missing: {ref_app}", file=sys.stderr)
        return 1
    dst_app = target / f"facility_{ccn}_superdynamic_dashboard.py"
    if dst_app.exists() and not force:
        print(f"  [SKIP] {dst_app.name} (exists; use --force to overwrite)")
    else:
        _adapt_superdynamic_entrypoint(ref_app, dst_app, ccn, ref_ccn)
        print(f"  [OK] {dst_app.name}")

    ref_vercel = ref / "vercel.json"
    if ref_vercel.is_file():
        dst_vercel = target / "vercel.json"
        if dst_vercel.exists() and not force:
            print("  [SKIP] vercel.json (exists; use --force)")
        else:
            _adapt_vercel_json(ref_vercel, dst_vercel, ccn, ref_ccn)
            print("  [OK] vercel.json (v2)")

    print("\n--- Manifest diff (required V2 runtime) ---")
    missing = []
    for line in _manifest_diff(root, target, ref, ccn):
        print(line)
        if line.startswith("  MISS"):
            missing.append(line.strip())
    if missing:
        print(f"\nERROR: {len(missing)} required file(s) missing after bootstrap.", file=sys.stderr)
        return 1

    print("\nBootstrap complete. Facility data slices preserved unless --force overwrote them.")
    print(f"Next: python scripts/preflight_v2_facility_deploy.py {ccn}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ccn", help="Target facility CCN")
    parser.add_argument("--ref", default="315461", help="Reference CCN (default 315461)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing infrastructure files in target bundle",
    )
    args = parser.parse_args()
    try:
        ccn = _normalize_ccn(args.ccn)
        ref_ccn = _normalize_ccn(args.ref)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return run_bootstrap(_ROOT, ccn, ref_ccn, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
