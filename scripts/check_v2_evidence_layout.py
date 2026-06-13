#!/usr/bin/env python3
"""Guard expandable evidence rows in v2 dashboards against layout regressions.

Expandable chart evidence (Supporting facility data, Case-Mix, Harrington, etc.)
must use the single-column ``pbj-chart-evidence-list`` panel — not
``pbj-chart-evidence-grid--cols-*`` side-by-side grids introduced by accident.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TEMPLATE = _ROOT / "templates" / "superdynamic_dashboard_v2.html"
_PARTIALS_DIR = _ROOT / "templates" / "partials" / "v2"
_MACRO = _PARTIALS_DIR / "chart_rollup_macros.html"

_STYLE_BLOCK = re.compile(r"<style\b[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)
_SCRIPT_BLOCK = re.compile(r"<script\b[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
_FORBIDDEN_MARKUP = re.compile(
    r"""class\s*=\s*["'][^"']*\bpbj-chart-evidence-grid(?:--cols-[23])?\b""",
    re.IGNORECASE,
)
_FORBIDDEN_HOST = re.compile(
    r"""class\s*=\s*["'][^"']*\bpbj-chart-evidence-grid-host\b""",
    re.IGNORECASE,
)
_REQUIRED_LIST_CSS = (
    ".pbj-chart-evidence-list {",
    ".pbj-chart-evidence-list > *:not(:last-child)",
)


def _strip_non_markup(html: str) -> str:
    html = _STYLE_BLOCK.sub("", html)
    html = _SCRIPT_BLOCK.sub("", html)
    return html


def _check_file(path: Path, *, label: str) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"{label}: missing file {path}"]
    markup = _strip_non_markup(path.read_text(encoding="utf-8"))
    for pattern, msg in (
        (_FORBIDDEN_MARKUP, "uses pbj-chart-evidence-grid* on markup — use pbj-chart-evidence-list"),
        (_FORBIDDEN_HOST, "uses pbj-chart-evidence-grid-host — remove grid wrapper from evidence rows"),
    ):
        if pattern.search(markup):
            errors.append(f"{label}: {msg}")
    return errors


def _check_macro() -> list[str]:
    errors: list[str] = []
    if not _MACRO.is_file():
        return ["chart_rollup_macros.html: missing"]
    text = _MACRO.read_text(encoding="utf-8")
    if 'macro pbj_chart_evidence_group' not in text:
        errors.append("chart_rollup_macros.html: pbj_chart_evidence_group macro missing")
    elif 'pbj-chart-evidence-list' not in text:
        errors.append(
            "chart_rollup_macros.html: pbj_chart_evidence_group must wrap rows in pbj-chart-evidence-list"
        )
    return errors


def _check_list_css(template: Path) -> list[str]:
    if not template.is_file():
        return [f"{template}: missing template for list CSS check"]
    css_blocks = _STYLE_BLOCK.findall(template.read_text(encoding="utf-8"))
    css = "\n".join(css_blocks)
    missing = [needle for needle in _REQUIRED_LIST_CSS if needle not in css]
    if missing:
        return [
            f"{template.name}: missing pbj-chart-evidence-list panel CSS ({missing[0]} …)"
        ]
    return []


def run_checks(*, template: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    errors.extend(_check_macro())
    errors.extend(_check_list_css(template))
    errors.extend(_check_file(template, label=template.name))
    if _PARTIALS_DIR.is_dir():
        for partial in sorted(_PARTIALS_DIR.rglob("*.html")):
            errors.extend(_check_file(partial, label=partial.relative_to(_ROOT).as_posix()))
    return (not errors, errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=_DEFAULT_TEMPLATE,
        help="Path to superdynamic_dashboard_v2.html",
    )
    args = parser.parse_args()
    ok, errors = run_checks(template=args.template.resolve())
    if ok:
        print("PASS: v2 evidence layout checks OK")
        return 0
    print("FAIL: v2 evidence layout regression detected:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
