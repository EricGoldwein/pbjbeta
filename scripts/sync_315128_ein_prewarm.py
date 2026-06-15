#!/usr/bin/env python3
"""Sync EIN prewarm + favicon helpers from canonical dashboard into 315128 deploy entry."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
canonical = (ROOT / "dynamic_facility_dashboard.py").read_text(encoding="utf-8")
deploy_path = ROOT / "deployments" / "pbj320-315128" / "facility_315128_superdynamic_dashboard.py"
text = deploy_path.read_text(encoding="utf-8")

if "_EIN_ROSTER_PREWARM_STARTED" not in text:
    old = "_EIN_NURSING_SUMMARIES_ROWS: list[dict[str, Any]] | None = None"
    new = (
        old
        + "\n_EIN_ROSTER_PREWARM_STARTED = False\n_EIN_ROSTER_PREWARM_LOCK = threading.Lock()"
    )
    if old in text:
        text = text.replace(old, new, 1)
    else:
        anchor = "ein_nursing_summaries_df = None"
        if anchor not in text:
            raise SystemExit("EIN globals anchor missing")
        text = text.replace(
            anchor,
            anchor
            + "\n_EIN_ROSTER_PREWARM_STARTED = False\n_EIN_ROSTER_PREWARM_LOCK = threading.Lock()",
            1,
        )

if "_EIN_NURSING_SUMMARIES_ROWS_CACHE" not in text:
    anchor = "ein_nursing_summaries_df = None"
    if anchor not in text:
        raise SystemExit("ein_nursing_summaries_df global missing")
    text = text.replace(
        anchor,
        anchor
        + "\n_EIN_NURSING_SUMMARIES_ROWS_CACHE: tuple[Any, ...] | None = None"
        + "\n_EIN_NURSING_SUMMARIES_ROWS: list[dict[str, Any]] | None = None",
        1,
    )

if "def _schedule_ein_roster_prewarm" not in text:
    m = re.search(
        r"def _ein_nursing_summaries_rows_all\(\) -> list\[dict\[str, Any\]\]:.*?"
        r"def _warm_ein_sustained_work_cache\(prov: str\) -> None:.*?"
        r"def _schedule_ein_roster_prewarm\(prov: str\) -> None:.*?"
        r"daemon=True,\n    \)\.start\(\)\n",
        canonical,
        re.S,
    )
    if not m:
        raise SystemExit("ein prewarm block missing in canonical")
    block = m.group(0) + "\n"
    anchor = "def _ensure_ein_employee_detail_loaded"
    if anchor not in text:
        raise SystemExit("insert anchor missing")
    text = text.replace(anchor, block + anchor, 1)
else:
    if "def _ein_nursing_summaries_rows_all" not in text:
        m = re.search(
            r"def _ein_nursing_summaries_rows_all\(\) -> list\[dict\[str, Any\]\]:.*?\n\n",
            canonical,
            re.S,
        )
        if not m:
            raise SystemExit("_ein_nursing_summaries_rows_all missing in canonical")
        anchor = "def _warm_ein_sustained_work_cache"
        if anchor not in text:
            raise SystemExit("_warm anchor missing")
        text = text.replace(anchor, m.group(0) + anchor, 1)

needle = 'print(f"[EIN] Loaded tables for {prov} (detail={detail_base})")'
if needle in text and "_schedule_ein_roster_prewarm(prov)" not in text.split(needle, 1)[1][:120]:
    text = text.replace(
        needle,
        needle + "\n            _schedule_ein_roster_prewarm(prov)",
        1,
    )

if "_pbj_favicon_directory_and_name" not in text:
    old_path_fn = '''def _pbj_favicon_path() -> Optional[str]:
    """Project-root favicon path when ``pbj_favicon.png`` is shipped with the app."""
    p = os.path.join(_app_root, "pbj_favicon.png")
    return p if os.path.isfile(p) else None'''
    new_path_fn = '''def _pbj_favicon_path() -> Optional[str]:
    """Resolve favicon PNG under app root or ``pbj_images/``."""
    for rel in ("pbj_favicon.png", os.path.join("pbj_images", "pbj_favicon.png")):
        p = os.path.join(_app_root, rel)
        if os.path.isfile(p):
            return p
    return None


def _pbj_favicon_directory_and_name() -> tuple[str, str] | None:
    path = _pbj_favicon_path()
    if not path:
        return None
    return os.path.dirname(path), os.path.basename(path)'''
    if old_path_fn in text:
        text = text.replace(old_path_fn, new_path_fn, 1)
    old_png_route = '''@app.route("/pbj_favicon.png")
def pbj_favicon_png():
    """Serve ``pbj_favicon.png`` from the application directory (repo root for facility apps)."""
    if not _pbj_favicon_path():
        return ("", 404)
    return send_from_directory(_app_root, "pbj_favicon.png", mimetype="image/png")'''
    new_png_route = '''@app.route("/pbj_favicon.png")
def pbj_favicon_png():
    """Serve ``pbj_favicon.png`` from the application directory (repo root for facility apps)."""
    fav = _pbj_favicon_directory_and_name()
    if not fav:
        return ("", 404)
    directory, filename = fav
    return send_from_directory(directory, filename, mimetype="image/png")'''
    if old_png_route in text:
        text = text.replace(old_png_route, new_png_route, 1)
    old_ico_route = '''@app.route("/favicon.ico")
def favicon_ico():
    """Browsers request ``/favicon.ico`` by default; reuse the PNG asset when present."""
    if not _pbj_favicon_path():
        return ("", 204)
    return send_from_directory(_app_root, "pbj_favicon.png", mimetype="image/png")'''
    new_ico_route = '''@app.route("/favicon.ico")
def favicon_ico():
    """Browsers request ``/favicon.ico`` by default; reuse the PNG asset when present."""
    fav = _pbj_favicon_directory_and_name()
    if not fav:
        return ("", 204)
    directory, filename = fav
    return send_from_directory(directory, filename, mimetype="image/png")'''
    if old_ico_route in text:
        text = text.replace(old_ico_route, new_ico_route, 1)

deploy_path.write_text(text, encoding="utf-8")
print(f"synced -> {deploy_path}")
