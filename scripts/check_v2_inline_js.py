#!/usr/bin/env python3
"""Syntax-check inline <script> blocks in superdynamic_dashboard_v2.html via node --check.

Catches duplicate const/let, stray tokens, and other parse errors in the large v2 inline
bundle before package/deploy. Jinja placeholders are stripped/replaced so Node can parse.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TEMPLATE = _ROOT / "templates" / "superdynamic_dashboard_v2.html"

_SCRIPT_OPEN = re.compile(
    r"<script\b(?=[^>]*>)(?![^>]*\bsrc\s*=)(?![^>]*type\s*=\s*['\"]application/json['\"])[^>]*>",
    re.IGNORECASE,
)
_SCRIPT_CLOSE = re.compile(r"</script>", re.IGNORECASE)
_JINJA_BLOCK = re.compile(r"\{%.*?%\}", re.DOTALL)
_JINJA_TOJSON = re.compile(r"\{\{[^}]*\|\s*tojson[^}]*\}\}", re.IGNORECASE | re.DOTALL)
_JINJA_EXPR = re.compile(r"\{\{.*?\}\}", re.DOTALL)


def _extract_inline_script_blocks(html: str) -> list[str]:
    blocks: list[str] = []
    pos = 0
    while True:
        open_m = _SCRIPT_OPEN.search(html, pos)
        if not open_m:
            break
        start = open_m.end()
        close_m = _SCRIPT_CLOSE.search(html, start)
        if not close_m:
            raise ValueError("Unclosed <script> block in template")
        body = html[start : close_m.start()]
        if body.strip():
            blocks.append(body)
        pos = close_m.end()
    return blocks


def _sanitize_jinja(js: str) -> str:
    js = _JINJA_BLOCK.sub("", js)
    js = _JINJA_TOJSON.sub("null", js)
    js = _JINJA_EXPR.sub("__JINJA__", js)
    return js


def _node_check(js: str, label: str) -> tuple[bool, str]:
    node = shutil.which("node")
    if not node:
        return False, "node executable not found on PATH"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".js",
        delete=False,
        encoding="utf-8",
        newline="\n",
    ) as tmp:
        tmp.write(js)
        tmp_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [node, "--check", str(tmp_path)],
            capture_output=True,
            text=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    if proc.returncode == 0:
        return True, f"{label}: OK"
    detail = (proc.stderr or proc.stdout or "syntax error").strip()
    return False, f"{label}: {detail}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run node --check on inline JS extracted from superdynamic_dashboard_v2.html.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=_DEFAULT_TEMPLATE,
        help="Path to v2 template (default: repo templates/superdynamic_dashboard_v2.html).",
    )
    args = parser.parse_args()
    tpl = args.template.resolve()
    if not tpl.is_file():
        print(f"ERROR: template not found: {tpl}", file=sys.stderr)
        return 2

    html = tpl.read_text(encoding="utf-8")
    blocks = _extract_inline_script_blocks(html)
    if not blocks:
        print("ERROR: no inline script blocks found", file=sys.stderr)
        return 2

    failures: list[str] = []
    for idx, raw in enumerate(blocks, start=1):
        sanitized = _sanitize_jinja(raw)
        ok, msg = _node_check(sanitized, f"inline script block {idx}/{len(blocks)}")
        print(msg)
        if not ok:
            failures.append(msg)

    if failures:
        print(f"\nFAIL: {len(failures)} inline script block(s) failed node --check", file=sys.stderr)
        return 1

    print(f"\nPASS: {len(blocks)} inline script block(s) syntax OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
