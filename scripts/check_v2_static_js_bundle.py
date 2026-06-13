#!/usr/bin/env python3
"""Verify v2 template script tags are present in a deployment static/js bundle."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TEMPLATE = _ROOT / "templates" / "superdynamic_dashboard_v2.html"
_DEFAULT_STANDALONE_TEMPLATE = _ROOT / "templates" / "report_builder_v3_standalone.html"
_SCRIPT_SRC = re.compile(
    r"""static/js/([A-Za-z0-9_\-]+\.js)""",
    re.IGNORECASE,
)
_DEPLOY_SYNC_REQUIRED = {
    "superdynamic_utils.js",
    "pbj_facility_display_name.js",
    "pbj_sparkline.js",
    "pbj_work_date_bridge.js",
    "pbj_red_flag_quarter_resolver.js",
    "pbj_v2_dashboard_extras.js",
    "pbj_v2_benchmark_workspace.js",
    "pbj_facility_events.js",
    "pbj_ein_roster_lifecycle.js",
    "pbj_sustained_work_flags.js",
    "pbj_daily_staffing_flags.js",
    "pbj_v2_chart_scope_toolbar.js",
    "pbj_v2_scope_config.js",
    "pbj_v2_daily_table_scope.js",
    "pbj_v2_geo_distribution.js",
    "pbj_v2_peer_comparison.js",
    "pbj_v3_panes.js",
    "pbj_v3_risk_timeline.js",
    "pbj_v3_handoffs.js",
    "pbj_v2_perf_debug.js",
    "pbj_report_builder_v3.js",
    "pbj_report_item_registry.js",
    "pbj320_snapshot_report.js",
    "pbj_rb3_ai_bridge.js",
    "pbj_ownership_disclosures.js",
}


def _scripts_referenced(template: Path) -> set[str]:
    text = template.read_text(encoding="utf-8")
    found = set(_SCRIPT_SRC.findall(text))
    return found


def run_check(
    *,
    template: Path,
    deploy_dir: Path,
    extra_templates: list[Path] | None = None,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not template.is_file():
        return False, [f"Missing template: {template}"]
    refs = _scripts_referenced(template)
    for extra in extra_templates or []:
        if extra.is_file():
            refs |= _scripts_referenced(extra)
    missing_sync = sorted(refs - _DEPLOY_SYNC_REQUIRED)
    if missing_sync:
        errors.append(
            "Template references JS not in deploy sync list: "
            + ", ".join(missing_sync)
            + " — update deploy_vercel_facility.py _sync_superdynamic_v2_ui"
        )
    if deploy_dir.is_dir():
        static_dir = deploy_dir / "static" / "js"
        for name in sorted(refs):
            if not (static_dir / name).is_file():
                errors.append(f"Missing in deployment bundle: static/js/{name}")
    return (not errors, errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=_DEFAULT_TEMPLATE)
    parser.add_argument(
        "--extra-template",
        type=Path,
        action="append",
        default=None,
        help="Additional templates whose script refs must exist in the deployment bundle",
    )
    parser.add_argument("--deploy-dir", type=Path, required=True)
    args = parser.parse_args()
    extras = list(args.extra_template or [])
    if _DEFAULT_STANDALONE_TEMPLATE.is_file() and _DEFAULT_STANDALONE_TEMPLATE not in extras:
        extras.append(_DEFAULT_STANDALONE_TEMPLATE)
    ok, errors = run_check(
        template=args.template.resolve(),
        deploy_dir=args.deploy_dir.resolve(),
        extra_templates=[p.resolve() for p in extras],
    )
    if ok:
        print(f"PASS: static JS bundle OK ({args.deploy_dir.name})")
        return 0
    print(f"FAIL: static JS bundle check ({args.deploy_dir.name})", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
