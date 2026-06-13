"""
Canonical report item model for Report Builder v3.

Report Builder owns this schema. The V2 dashboard must NOT import this module;
future dashboard features send structured items via API/events only.

Layers:
  - scope: report period, events (not reorderable body items)
  - finding: system-detected staffing patterns for the selected period/windows
  - user_added: charts, tables, flags, notes queued from dashboard / AI / user
  - supporting_context: limited outside-period context when materially relevant
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict

ReportItemType = Literal["chart", "table", "finding", "event_window", "methodology", "note"]
ReportItemSource = Literal["report_builder", "dashboard", "pbj320_control_center", "ai", "user"]
ReportItemCategory = Literal["scope", "finding", "user_added", "supporting_context"]

SOURCE_REPORT_BUILDER: ReportItemSource = "report_builder"
SOURCE_DASHBOARD: ReportItemSource = "dashboard"
SOURCE_CONTROL_CENTER: ReportItemSource = "pbj320_control_center"
SOURCE_AI: ReportItemSource = "ai"
SOURCE_USER: ReportItemSource = "user"


class DateRangeRef(TypedDict, total=False):
    start: str
    end: str


class ReportItemDef(TypedDict, total=False):
    id: str
    type: ReportItemType
    category: ReportItemCategory
    title: str
    subtitle: str
    source: ReportItemSource
    enabled: bool
    order: int
    include_key: str
    section_marker_id: str
    data_ref: str
    date_range: DateRangeRef
    related_events: List[str]
    removable: bool


# Built-in memo body items (all source=report_builder today).
BUILTIN_REPORT_ITEMS: List[ReportItemDef] = [
    {
        "id": "key_staffing_findings",
        "type": "finding",
        "category": "finding",
        "title": "Key Staffing Findings",
        "subtitle": "Auto-detected staffing patterns for the report period and event windows",
        "source": SOURCE_REPORT_BUILDER,
        "enabled": True,
        "include_key": "key_staffing_findings",
        "section_marker_id": "key_staffing_findings",
        "removable": False,
    },
    {
        "id": "daily_staffing_table",
        "type": "table",
        "category": "finding",
        "title": "Daily Key-Date Staffing",
        "subtitle": "Staffing table for each case event date",
        "source": SOURCE_REPORT_BUILDER,
        "enabled": True,
        "include_key": "daily_staffing_table",
        "section_marker_id": "daily_staffing_table",
        "removable": False,
    },
    {
        "id": "state_compliance",
        "type": "finding",
        "category": "finding",
        "title": "Days Below State Minimum",
        "subtitle": "Compliance summary for the report period",
        "source": SOURCE_REPORT_BUILDER,
        "enabled": True,
        "include_key": "state_compliance",
        "section_marker_id": "state_compliance",
        "removable": False,
    },
    {
        "id": "quarterly_staffing",
        "type": "chart",
        "category": "finding",
        "title": "Longitudinal Staffing Analysis",
        "subtitle": "Charts and period/quarter tables",
        "source": SOURCE_REPORT_BUILDER,
        "enabled": True,
        "include_key": "quarterly_staffing",
        "section_marker_id": "quarterly_staffing",
        "removable": False,
    },
    {
        "id": "case_mix",
        "type": "table",
        "category": "finding",
        "title": "Case-Mix Analysis",
        "subtitle": "Case-mix and Harrington-adjusted metrics",
        "source": SOURCE_REPORT_BUILDER,
        "enabled": True,
        "include_key": "case_mix",
        "section_marker_id": "case_mix",
        "removable": False,
    },
    {
        "id": "red_flags",
        "type": "finding",
        "category": "finding",
        "title": "CMS Red Flags",
        "subtitle": "Provider info flags during the report period",
        "source": SOURCE_REPORT_BUILDER,
        "enabled": True,
        "include_key": "red_flags",
        "section_marker_id": "red_flags",
        "removable": False,
    },
    {
        "id": "event_windows",
        "type": "event_window",
        "category": "finding",
        "title": "Staffing Around Key Events",
        "subtitle": "Supplemental windows around case events",
        "source": SOURCE_REPORT_BUILDER,
        "enabled": True,
        "include_key": "event_windows",
        "section_marker_id": "event_windows",
        "removable": False,
    },
    {
        "id": "supporting_context",
        "type": "note",
        "category": "supporting_context",
        "title": "Relevant Context Outside Selected Period",
        "subtitle": "Limited context when patterns extend beyond the report period",
        "source": SOURCE_REPORT_BUILDER,
        "enabled": True,
        "include_key": "supporting_context",
        "section_marker_id": "supporting_context",
        "removable": False,
    },
    {
        "id": "appendix",
        "type": "methodology",
        "category": "finding",
        "title": "Appendix",
        "subtitle": "Methods, sources, and reference tables",
        "source": SOURCE_REPORT_BUILDER,
        "enabled": True,
        "include_key": "appendix",
        "section_marker_id": "appendix",
        "removable": False,
    },
]

DEFAULT_ITEM_ORDER: List[str] = [item["id"] for item in BUILTIN_REPORT_ITEMS if item.get("id")]

BUILTIN_BY_ID: Dict[str, ReportItemDef] = {str(i["id"]): i for i in BUILTIN_REPORT_ITEMS if i.get("id")}

EXECUTIVE_INCLUDE_KEYS = ("key_dates", "date_ranges_of_interest", "period_summary")


def default_builtin_items() -> List[Dict[str, Any]]:
    """Serializable defaults for API/UI."""
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(BUILTIN_REPORT_ITEMS):
        row = deepcopy(item)
        row["order"] = idx
        out.append(row)
    return out


def default_include_sections() -> Dict[str, bool]:
    inc = {k: True for k in EXECUTIVE_INCLUDE_KEYS}
    for item in BUILTIN_REPORT_ITEMS:
        key = str(item.get("include_key") or item.get("id") or "")
        if key:
            inc[key] = bool(item.get("enabled", True))
    return inc


def _coerce_item_row(raw: Any, fallback_order: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    item_id = str(raw.get("id") or "").strip()
    if not item_id:
        return None
    builtin = BUILTIN_BY_ID.get(item_id)
    row: Dict[str, Any] = deepcopy(builtin) if builtin else {
        "id": item_id,
        "type": str(raw.get("type") or "note"),
        "category": str(raw.get("category") or "user_added"),
        "title": str(raw.get("title") or item_id),
        "subtitle": str(raw.get("subtitle") or ""),
        "source": str(raw.get("source") or SOURCE_USER),
        "enabled": True,
        "include_key": item_id,
        "section_marker_id": item_id,
        "removable": True,
    }
    if raw.get("title"):
        row["title"] = str(raw["title"])
    if raw.get("subtitle") is not None:
        row["subtitle"] = str(raw.get("subtitle") or "")
    if raw.get("type"):
        row["type"] = str(raw["type"])
    if raw.get("category"):
        row["category"] = str(raw["category"])
    if raw.get("source"):
        row["source"] = str(raw["source"])
    if "enabled" in raw:
        row["enabled"] = bool(raw.get("enabled"))
    if raw.get("data_ref"):
        row["data_ref"] = str(raw["data_ref"])
    if isinstance(raw.get("date_range"), dict):
        row["date_range"] = {
            "start": str(raw["date_range"].get("start") or ""),
            "end": str(raw["date_range"].get("end") or ""),
        }
    if isinstance(raw.get("related_events"), list):
        row["related_events"] = [str(x) for x in raw["related_events"] if x]
    row["order"] = int(raw.get("order")) if raw.get("order") is not None else fallback_order
    return row


def parse_report_items_payload(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return default_builtin_items()
    parsed: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw):
        row = _coerce_item_row(item, idx)
        if row:
            parsed.append(row)
    if not parsed:
        return default_builtin_items()
    return parsed


def normalize_item_order(items: Sequence[Dict[str, Any]]) -> List[str]:
    """Stable section marker order from report items."""
    sorted_items = sorted(items, key=lambda r: (int(r.get("order", 9999)), str(r.get("id") or "")))
    out: List[str] = []
    seen: set[str] = set()
    for row in sorted_items:
        sid = str(row.get("section_marker_id") or row.get("id") or "").strip()
        if sid in BUILTIN_BY_ID or sid in {i["id"] for i in BUILTIN_REPORT_ITEMS}:
            if sid not in seen:
                out.append(sid)
                seen.add(sid)
    for default_id in DEFAULT_ITEM_ORDER:
        if default_id not in seen:
            out.append(default_id)
    return out


def include_sections_from_items(items: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
    inc = default_include_sections()
    for row in items:
        key = str(row.get("include_key") or row.get("id") or "").strip()
        if key in inc:
            inc[key] = bool(row.get("enabled", True))
    return inc


def _legacy_normalize_section_order(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return list(DEFAULT_ITEM_ORDER)
    out: List[str] = []
    seen: set[str] = set()
    for item in raw:
        sid = str(item or "").strip()
        if sid in BUILTIN_BY_ID and sid not in seen:
            out.append(sid)
            seen.add(sid)
    for sid in DEFAULT_ITEM_ORDER:
        if sid not in seen:
            out.append(sid)
    return out


def _legacy_merge_include_sections(raw: Any) -> Dict[str, bool]:
    inc = default_include_sections()
    if isinstance(raw, dict):
        for k, v in raw.items():
            key = str(k or "").strip()
            if key in inc:
                inc[key] = bool(v)
    return inc


def resolve_report_plan(payload: Dict[str, Any]) -> Tuple[List[str], Dict[str, bool], List[Dict[str, Any]]]:
    """
    Unified plan for preview/export.
    Accepts report_items (preferred) or legacy section_order + include_sections.
    """
    external = payload.get("user_report_items") or payload.get("queued_items") or []
    if isinstance(payload.get("report_items"), list) and payload.get("report_items"):
        items = merge_external_items(parse_report_items_payload(payload["report_items"]), external)
        order = normalize_item_order(items)
        inc = include_sections_from_items(items)
        return order, inc, items

    order = _legacy_normalize_section_order(payload.get("section_order"))
    inc = _legacy_merge_include_sections(payload.get("include_sections"))
    items = merge_external_items(default_builtin_items(), external)
    for row in items:
        iid = str(row.get("id") or "")
        row["enabled"] = bool(inc.get(str(row.get("include_key") or iid), True))
        if iid in order:
            row["order"] = order.index(iid)
    if external:
        order = normalize_item_order(items)
        inc = include_sections_from_items(items)
    return order, inc, items


def merge_external_items(
    builtin_items: Sequence[Dict[str, Any]],
    external_items: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Append user/dashboard/AI items; built-ins keep their slots unless same id."""
    merged = [deepcopy(i) for i in builtin_items]
    by_id = {str(i["id"]): i for i in merged if i.get("id")}
    next_order = max((int(i.get("order", 0)) for i in merged), default=-1) + 1
    for raw in external_items or []:
        row = _coerce_item_row(raw, next_order)
        if not row:
            continue
        iid = str(row["id"])
        if iid in by_id and str(row.get("source")) == SOURCE_REPORT_BUILDER:
            continue
        if iid in by_id:
            by_id[iid].update(row)
        else:
            merged.append(row)
            by_id[iid] = row
            next_order += 1
    return merged


class SupportingContextHint(TypedDict, total=False):
    code: str
    message: str
    related_events: List[str]


def collect_supporting_context_hints(
    *,
    report_start_iso: str,
    report_end_iso: str,
    data_lo_iso: str,
    data_hi_iso: str,
    key_date_isos: Sequence[str],
    event_window_ranges: Sequence[Dict[str, Any]],
    resident_stay_start: Optional[str] = None,
    resident_stay_end: Optional[str] = None,
) -> List[SupportingContextHint]:
    """Detect when limited outside-period context may help explain the memo."""
    hints: List[SupportingContextHint] = []

    def _parse(iso: str) -> Optional[str]:
        t = str(iso or "").strip()
        return t if len(t) == 10 else None

    rs, re_ = _parse(report_start_iso), _parse(report_end_iso)
    lo, hi = _parse(data_lo_iso), _parse(data_hi_iso)

    if rs and re_ and lo and hi and (rs < lo or re_ > hi):
        hints.append({
            "code": "period_clipped_to_data",
            "message": (
                "The selected report period was clipped to available PBJ data "
                f"({lo} – {hi}). Patterns at the edges of your selection may reflect partial coverage."
            ),
        })

    for ev_iso in key_date_isos:
        ev = _parse(ev_iso)
        if ev and rs and re_ and (ev < rs or ev > re_):
            hints.append({
                "code": "event_outside_period",
                "message": (
                    f"Case event on {ev} falls outside the selected report period ({rs} – {re_}). "
                    "It is listed in the memo but is not included in report-period charts unless you expand the period."
                ),
                "related_events": [ev],
            })

    for win in event_window_ranges:
        ws = _parse(str(win.get("window_start") or ""))
        we = _parse(str(win.get("window_end") or ""))
        if ws and lo and ws < lo:
            hints.append({
                "code": "window_before_data",
                "message": (
                    f"Event window \"{win.get('window_label', 'Window')}\" begins before available PBJ data ({lo}). "
                    "Staffing figures reflect only days with data."
                ),
                "related_events": [str(win.get("event_date") or "")] if win.get("event_date") else [],
            })
        if we and hi and we > hi:
            hints.append({
                "code": "window_after_data",
                "message": (
                    f"Event window \"{win.get('window_label', 'Window')}\" extends after available PBJ data ({hi})."
                ),
                "related_events": [str(win.get("event_date") or "")] if win.get("event_date") else [],
            })
        if rs and re_ and ws and we and (ws < rs or we > re_):
            hints.append({
                "code": "window_outside_period",
                "message": (
                    f"Event window \"{win.get('window_label', 'Window')}\" ({ws} – {we}) extends outside the "
                    f"main report period ({rs} – {re_}). This window is supplemental, not a substitute for the report period."
                ),
                "related_events": [str(win.get("event_date") or "")] if win.get("event_date") else [],
            })

    rss, rse = _parse(resident_stay_start or ""), _parse(resident_stay_end or "")
    if rss and rse and rs and re_ and (rss > rs or rse < re_):
        hints.append({
            "code": "stay_narrower_than_report",
            "message": (
                f"Resident stay period ({rss} – {rse}) is narrower than the selected report period ({rs} – {re_}). "
                "Main findings use the report period; stay-specific context may differ."
            ),
        })

    # De-dupe by code+message
    seen: set[str] = set()
    unique: List[SupportingContextHint] = []
    for h in hints:
        key = f"{h.get('code')}|{h.get('message')}"
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique


def generate_supporting_context_html(hints: Sequence[SupportingContextHint]) -> str:
    if not hints:
        return ""
    items = "".join(f"<li>{_xml_escape(str(h.get('message') or ''))}</li>" for h in hints)
    return (
        '<div class="page-break"></div>'
        '<h2 style="color:#2c3e50;margin-bottom:12px;">Relevant Context Outside Selected Period</h2>'
        '<p style="font-size:10pt;color:#475569;">The following notes describe limited context beyond the main '
        "report period. They are provided for orientation only and do not expand the primary analysis window.</p>"
        f'<ul style="font-size:10pt;line-height:1.55;">{items}</ul>'
    )


def _xml_escape(s: str) -> str:
    return (
        str(s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
