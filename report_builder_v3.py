"""
Report Builder v3 helpers: section ordering, event windows, validation, report summary.
Used by /api/report_builder_v3/* endpoints; v2 report builder remains unchanged.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from report_item_registry import (
    BUILTIN_REPORT_ITEMS,
    DEFAULT_ITEM_ORDER,
    collect_supporting_context_hints,
    generate_supporting_context_html,
    resolve_report_plan,
)

# Back-compat aliases
DEFAULT_SECTION_ORDER: List[str] = list(DEFAULT_ITEM_ORDER)
REPORT_SECTION_CATALOG: List[Dict[str, Any]] = list(BUILTIN_REPORT_ITEMS)

EVENT_WINDOW_PRESETS: Dict[str, Dict[str, Any]] = {
    "before_30": {"label": "30 days before event", "kind": "before", "days": 30},
    "before_60": {"label": "60 days before event", "kind": "before", "days": 60},
    "before_90": {"label": "90 days before event", "kind": "before", "days": 90},
    "after_14": {"label": "14 days after event", "kind": "after", "days": 14},
    "after_30": {"label": "30 days after event", "kind": "after", "days": 30},
    "event_quarter": {"label": "Quarter containing event", "kind": "quarter"},
    "custom": {"label": "Custom window", "kind": "custom"},
}

KEY_DATE_TYPES = (
    "incident",
    "admission",
    "discharge",
    "survey",
    "complaint",
    "other",
)


def default_include_sections() -> Dict[str, bool]:
    from report_item_registry import default_include_sections as _default_include_sections

    return _default_include_sections()


def normalize_section_order(raw: Any) -> List[str]:
    from report_item_registry import _legacy_normalize_section_order

    return _legacy_normalize_section_order(raw)


def merge_include_sections(raw: Any) -> Dict[str, bool]:
    from report_item_registry import _legacy_merge_include_sections

    return _legacy_merge_include_sections(raw)


def parse_key_date_events(raw: Any) -> Tuple[List[datetime], Dict[str, str], Dict[str, str]]:
    """Return sorted dates, notes by iso, event type by iso."""
    if raw is None:
        return [], {}, {}
    entries: List[Tuple[str, str, str]] = []

    def _push(date_part: str, label: str, event_type: str) -> None:
        dp = str(date_part or "").strip()
        if dp:
            entries.append((dp, str(label or "").strip(), str(event_type or "other").strip().lower()))

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                d_raw = str(item.get("date") or item.get("key_date") or "").strip()
                note = str(item.get("note") or item.get("label") or "").strip()
                et = str(item.get("type") or item.get("event_type") or "other").strip().lower()
                if et not in KEY_DATE_TYPES:
                    et = "other"
                _push(d_raw, note, et)
            else:
                _push(str(item or "").strip(), "", "other")
    else:
        for token in re.split(r"[\n,;]+", str(raw)):
            t = str(token or "").strip()
            if t:
                _push(t, "", "other")

    out: List[datetime] = []
    notes: Dict[str, str] = {}
    types: Dict[str, str] = {}
    seen: set[str] = set()

    for date_token, note, et in entries:
        dt = _parse_flexible_date(date_token)
        if dt is None:
            continue
        iso = dt.strftime("%Y-%m-%d")
        if iso in seen:
            if note:
                notes[iso] = note
            if et and et != "other":
                types[iso] = et
            continue
        seen.add(iso)
        out.append(dt)
        if note:
            notes[iso] = note
        types[iso] = et if et in KEY_DATE_TYPES else "other"

    return sorted(out), notes, types


def _parse_flexible_date(token: str) -> Optional[datetime]:
    t = str(token or "").strip()
    if not t:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        try:
            return datetime.strptime(t, "%Y-%m-%d")
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{2,4})", t)
    if not m:
        return None
    mm, dd = int(m.group(1)), int(m.group(2))
    y_raw = m.group(3)
    yyyy = int(y_raw) + (2000 if len(y_raw) == 2 else 0)
    try:
        return datetime(year=yyyy, month=mm, day=dd)
    except ValueError:
        return None


def parse_event_window_selection(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        preset = str(item.get("preset") or item.get("id") or "").strip()
        if preset not in EVENT_WINDOW_PRESETS and preset != "custom":
            continue
        row: Dict[str, Any] = {"preset": preset}
        if preset == "custom":
            row["start_date"] = str(item.get("start_date") or "").strip()
            row["end_date"] = str(item.get("end_date") or "").strip()
            row["label"] = str(item.get("label") or "Custom window").strip() or "Custom window"
        if item.get("event_date"):
            row["event_date"] = str(item.get("event_date") or "").strip()
        out.append(row)
    return out


def _quarter_bounds_from_date(dt: datetime) -> Tuple[datetime, datetime]:
    q = ((dt.month - 1) // 3) + 1
    start_month = (q - 1) * 3 + 1
    end_month = start_month + 2
    start = datetime(dt.year, start_month, 1)
    if end_month == 12:
        end = datetime(dt.year, 12, 31)
    else:
        end = datetime(dt.year, end_month + 1, 1) - timedelta(days=1)
    return start, end


def resolve_event_window_range(
    preset: str,
    event_dt: datetime,
    custom_start: Optional[str] = None,
    custom_end: Optional[str] = None,
) -> Tuple[Optional[datetime], Optional[datetime], str]:
    meta = EVENT_WINDOW_PRESETS.get(preset)
    if not meta:
        return None, None, ""
    if preset == "custom":
        sd = _parse_flexible_date(str(custom_start or ""))
        ed = _parse_flexible_date(str(custom_end or ""))
        if not sd or not ed:
            return None, None, "Custom window"
        if ed < sd:
            sd, ed = ed, sd
        return sd, ed, str(meta.get("label") or "Custom window")
    if preset == "event_quarter":
        lo, hi = _quarter_bounds_from_date(event_dt)
        return lo, hi, meta["label"]
    days = int(meta.get("days") or 0)
    if meta.get("kind") == "before":
        end = event_dt - timedelta(days=1)
        start = event_dt - timedelta(days=days)
        return start, end, meta["label"]
    if meta.get("kind") == "after":
        start = event_dt + timedelta(days=1)
        end = event_dt + timedelta(days=days)
        return start, end, meta["label"]
    return None, None, meta.get("label", "")


def collect_v3_warnings(
    *,
    report_start: datetime,
    report_end: datetime,
    data_lo: datetime,
    data_hi: datetime,
    key_dates: Sequence[datetime],
    event_windows: Sequence[Dict[str, Any]],
    has_period_data: bool,
) -> List[str]:
    warnings: List[str] = []
    if not has_period_data:
        warnings.append(
            "The selected report period has no PBJ daily data. Expand the period or choose dates within dataset coverage."
        )
    for kd in key_dates:
        if kd.date() < report_start.date() or kd.date() > report_end.date():
            warnings.append(
                f"Event on {kd.strftime('%b %d, %Y')} is outside the selected report period. "
                "It can still be listed, but it will not appear in report-period charts unless you expand the period."
            )
    for win in event_windows:
        ws = win.get("window_start")
        we = win.get("window_end")
        if isinstance(ws, datetime) and isinstance(we, datetime):
            if we.date() < data_lo.date() or ws.date() > data_hi.date():
                warnings.append(
                    f"Window \"{win.get('window_label', 'Event window')}\" ({ws.strftime('%b %d, %Y')} – "
                    f"{we.strftime('%b %d, %Y')}) extends outside available PBJ data."
                )
    return warnings


def build_v3_report_summary_html(meta: Dict[str, Any]) -> str:
    facility = _xml_escape(str(meta.get("facility_name") or ""))
    ccn = _xml_escape(str(meta.get("ccn") or ""))
    period = _xml_escape(str(meta.get("report_period_label") or ""))
    generated = _xml_escape(str(meta.get("generated_date") or ""))
    events_html = meta.get("events_html") or "<li><em>No case events entered</em></li>"
    windows_html = meta.get("windows_html") or "<li><em>No event windows selected</em></li>"
    return f"""
    <div class="rb-v3-report-summary" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:14px 16px;margin:16px 0 20px;">
        <h2 style="color:#2c3e50;margin:0 0 10px;font-size:13pt;">Report Summary</h2>
        <table style="width:100%;font-size:10pt;border-collapse:collapse;">
            <tr><td style="padding:4px 8px 4px 0;font-weight:600;width:34%;vertical-align:top;">Facility</td><td style="padding:4px 0;">{facility} (CCN {ccn})</td></tr>
            <tr><td style="padding:4px 8px 4px 0;font-weight:600;vertical-align:top;">Report period</td><td style="padding:4px 0;">{period}</td></tr>
            <tr><td style="padding:4px 8px 4px 0;font-weight:600;vertical-align:top;">Case events</td><td style="padding:4px 0;"><ul style="margin:0;padding-left:18px;">{events_html}</ul></td></tr>
            <tr><td style="padding:4px 8px 4px 0;font-weight:600;vertical-align:top;">Event windows</td><td style="padding:4px 0;"><ul style="margin:0;padding-left:18px;">{windows_html}</ul></td></tr>
            <tr><td style="padding:4px 8px 4px 0;font-weight:600;vertical-align:top;">Generated</td><td style="padding:4px 0;">{generated}</td></tr>
        </table>
        <p style="font-size:9pt;color:#64748b;margin:10px 0 0;">Event windows are supplemental to the main report period. Staffing levels are described using neutral language and do not imply causation.</p>
    </div>
    """


def _xml_escape(s: str) -> str:
    return (
        str(s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def format_event_list_html(
    key_dates: Sequence[datetime],
    notes: Dict[str, str],
    types: Dict[str, str],
) -> str:
    if not key_dates:
        return ""
    parts: List[str] = []
    for dt in key_dates:
        iso = dt.strftime("%Y-%m-%d")
        label = notes.get(iso) or types.get(iso, "other").replace("_", " ").title()
        et = types.get(iso, "other")
        type_label = et.replace("_", " ").title() if et != "other" else ""
        suffix = f" ({type_label})" if type_label else ""
        parts.append(f"<li><strong>{_xml_escape(label)}</strong>{suffix} — {dt.strftime('%b %d, %Y')}</li>")
    return "".join(parts)


def format_windows_list_html(windows: Sequence[Dict[str, Any]]) -> str:
    if not windows:
        return ""
    parts: List[str] = []
    for w in windows:
        ev = w.get("event_label") or "Event"
        wl = w.get("window_label") or "Window"
        ws = w.get("window_start")
        we = w.get("window_end")
        if isinstance(ws, datetime) and isinstance(we, datetime):
            parts.append(
                f"<li>{_xml_escape(str(ev))}: {_xml_escape(wl)} — "
                f"{ws.strftime('%b %d, %Y')} – {we.strftime('%b %d, %Y')}</li>"
            )
    return "".join(parts)


def generate_event_windows_section_html(
    windows: Sequence[Dict[str, Any]],
    period_metrics_fn,
    min_staffing: float,
    include_total: bool,
) -> str:
    """Build supplemental event-window section. period_metrics_fn(start, end) -> dict."""
    if not windows:
        return ""
    blocks: List[str] = []
    blocks.append(
        '<div class="page-break"></div>'
        '<h2 style="color:#2c3e50;margin-bottom:16px;">Staffing Around Key Events</h2>'
        '<p style="font-size:10pt;color:#475569;">The following summaries describe staffing levels during supplemental '
        "windows around case events. These windows do not replace the main report period.</p>"
    )
    for w in windows:
        ws = w.get("window_start")
        we = w.get("window_end")
        if not isinstance(ws, datetime) or not isinstance(we, datetime):
            continue
        pm = period_metrics_fn(ws, we) or {}
        ev_label = _xml_escape(str(w.get("event_label") or "Event"))
        wl = _xml_escape(str(w.get("window_label") or "Window"))
        avg_total = pm.get("total_hprd")
        avg_direct = pm.get("direct_care_hprd")
        days_under_d = int(pm.get("days_under_minimum_direct") or 0)
        days_under_t = int(pm.get("days_under_minimum_total") or 0)
        total_days = int(pm.get("total_days") or 0)
        below_line = ""
        if min_staffing > 0 and total_days > 0:
            if include_total and days_under_t:
                below_line = (
                    f"<li>Days below threshold (total HPRD): <strong>{days_under_t}</strong> of {total_days}</li>"
                )
            elif days_under_d:
                below_line = (
                    f"<li>Days below threshold (direct care HPRD): <strong>{days_under_d}</strong> of {total_days}</li>"
                )
            else:
                below_line = f"<li>Days below threshold: 0 of {total_days}</li>"
        hprd_bits = []
        if include_total and avg_total is not None:
            hprd_bits.append(f"Average total HPRD: <strong>{float(avg_total):.2f}</strong>")
        if avg_direct is not None:
            hprd_bits.append(f"Average direct care HPRD: <strong>{float(avg_direct):.2f}</strong>")
        hprd_line = "<li>" + "; ".join(hprd_bits) + "</li>" if hprd_bits else ""
        blocks.append(
            f'<div style="margin:18px 0;padding:12px 14px;border:1px solid #e2e8f0;border-radius:6px;background:#fff;">'
            f"<h3 style=\"margin:0 0 8px;font-size:12pt;color:#334155;\">{ev_label} — {ws.strftime('%b %d, %Y') if w.get('event_date') else ''}</h3>"
            f"<p style=\"margin:0 0 8px;font-size:10pt;\"><strong>{wl}:</strong> "
            f"{ws.strftime('%b %d, %Y')} – {we.strftime('%b %d, %Y')}</p>"
            f"<ul style=\"font-size:10pt;margin:0;padding-left:18px;\">{hprd_line}{below_line}</ul>"
            f"</div>"
        )
    return "\n".join(blocks)


_SECTION_MARKER_RE = re.compile(
    r"<!--\s*PBJ-RB-SECTION-BEGIN:([a-z_]+)\s*-->(.*?)<!--\s*PBJ-RB-SECTION-END:\1\s*-->",
    re.DOTALL | re.IGNORECASE,
)


def inject_section_marker(section_id: str, html: str) -> str:
    if not html or not str(html).strip():
        return ""
    return f"<!-- PBJ-RB-SECTION-BEGIN:{section_id} -->\n{html}\n<!-- PBJ-RB-SECTION-END:{section_id} -->"


def reorder_report_sections(
    html: str,
    section_order: Sequence[str],
    include_sections: Dict[str, bool],
) -> str:
    """Reorder marked body sections; unmarked content (header, executive summary) is preserved."""
    text = str(html or "")
    matches = list(_SECTION_MARKER_RE.finditer(text))
    if not matches:
        return text
    blocks: Dict[str, str] = {m.group(1): m.group(2) for m in matches}
    first_begin = matches[0].start()
    last_end = matches[-1].end()
    prefix = text[:first_begin]
    suffix = text[last_end:]
    order = normalize_section_order(list(section_order))
    parts: List[str] = [prefix]
    for sid in order:
        if not include_sections.get(sid, True):
            continue
        chunk = blocks.get(sid, "")
        if chunk and str(chunk).strip():
            parts.append(f"<!-- PBJ-RB-SECTION-BEGIN:{sid} -->{chunk}<!-- PBJ-RB-SECTION-END:{sid} -->")
    parts.append(suffix)
    return "".join(parts)


def inject_v3_summary(html: str, summary_html: str) -> str:
    if not summary_html or not str(summary_html).strip():
        return html
    marker = "<div class=\"summary-box\">"
    if marker in html:
        return html.replace(marker, summary_html + "\n    " + marker, 1)
    marker2 = "<h1 style=\"color: #2c3e50"
    if marker2 in html:
        idx = html.find(marker2)
        end = html.find("</p>", idx)
        if end > 0:
            insert_at = end + 4
            return html[:insert_at] + "\n    " + summary_html + html[insert_at:]
    return summary_html + html


def build_resolved_event_windows(
    key_dates: Sequence[datetime],
    notes: Dict[str, str],
    types: Dict[str, str],
    selections: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Expand preset selections into concrete windows per event."""
    if not selections or not key_dates:
        return []
    resolved: List[Dict[str, Any]] = []
    for ev in key_dates:
        iso = ev.strftime("%Y-%m-%d")
        ev_label = notes.get(iso) or types.get(iso, "Event").replace("_", " ").title()
        for sel in selections:
            preset = str(sel.get("preset") or "").strip()
            if sel.get("event_date") and str(sel.get("event_date")).strip() not in ("", iso):
                continue
            ws, we, wlabel = resolve_event_window_range(
                preset,
                ev,
                sel.get("start_date"),
                sel.get("end_date"),
            )
            if not ws or not we:
                continue
            resolved.append(
                {
                    "event_date": iso,
                    "event_label": ev_label,
                    "window_label": wlabel,
                    "window_start": ws,
                    "window_end": we,
                    "preset": preset,
                }
            )
    return resolved


def format_period_label(start_dt: datetime, end_dt: datetime) -> str:
    return f"{start_dt.strftime('%b %d, %Y')} – {end_dt.strftime('%b %d, %Y')}"


_SECTION_TAIL: Tuple[str, ...] = ("supporting_context", "appendix")
_SECTION_HEAD: Tuple[str, ...] = ("key_staffing_findings",)


def prioritize_section_order_from_findings(
    base_order: Sequence[str],
    findings: Sequence[Dict[str, Any]],
    *,
    has_case_events: bool = False,
    enabled: bool = True,
) -> List[str]:
    """
    Re-rank memo body sections so the report leads with evidence tied to detected issues.
    Executive summary (unmarked) stays first; appendix stays last.
    """
    if not enabled:
        return normalize_section_order(list(base_order))
    order = normalize_section_order(list(base_order))
    boost: Dict[str, float] = {sid: 0.0 for sid in order}
    for f in findings or []:
        sev = str(f.get("severity") or "context")
        sev_boost = {"high": 30.0, "medium": 18.0, "low": 8.0, "context": 3.0}.get(sev, 0.0)
        src = str(f.get("source_section") or "").strip()
        ftype = str(f.get("type") or "").strip()
        if src in boost:
            boost[src] += sev_boost
        if ftype == "compliance":
            boost["state_compliance"] = boost.get("state_compliance", 0.0) + sev_boost + 5.0
        if ftype in ("rn_coverage", "event_window"):
            boost["daily_staffing_table"] = boost.get("daily_staffing_table", 0.0) + sev_boost + 4.0
        if ftype == "acuity_gap":
            boost["case_mix"] = boost.get("case_mix", 0.0) + sev_boost + 6.0
        if ftype in ("staffing_volatility", "weekend_pattern"):
            boost["quarterly_staffing"] = boost.get("quarterly_staffing", 0.0) + sev_boost + 3.0
        if ftype == "employee_anomaly":
            boost["appendix"] = boost.get("appendix", 0.0) + min(sev_boost, 12.0)
    if has_case_events:
        boost["event_windows"] = boost.get("event_windows", 0.0) + 25.0
        boost["daily_staffing_table"] = boost.get("daily_staffing_table", 0.0) + 10.0

    def _sort_key(sid: str) -> Tuple[float, int]:
        if sid in _SECTION_HEAD:
            return (-1000.0, order.index(sid) if sid in order else 999)
        if sid in _SECTION_TAIL:
            return (1000.0 + order.index(sid), order.index(sid) if sid in order else 999)
        return (-boost.get(sid, 0.0), order.index(sid) if sid in order else 999)

    ranked = sorted(order, key=_sort_key)
    return ranked


def _parse_resident_stay_period(payload: Dict[str, Any]) -> Optional[Tuple[datetime, datetime]]:
    rs_start = str(payload.get("resident_stay_start") or "").strip()
    rs_end = str(payload.get("resident_stay_end") or "").strip()
    scope = payload.get("report_scope")
    if isinstance(scope, dict):
        stay = scope.get("resident_stay_period")
        if isinstance(stay, dict):
            rs_start = rs_start or str(stay.get("start") or "").strip()
            rs_end = rs_end or str(stay.get("end") or "").strip()
    if not rs_start or not rs_end:
        return None
    try:
        sd = datetime.strptime(rs_start, "%Y-%m-%d")
        ed = datetime.strptime(rs_end, "%Y-%m-%d")
        if ed < sd:
            sd, ed = ed, sd
        return sd, ed
    except ValueError:
        return None


def build_v3_report_extras(
    *,
    payload: Dict[str, Any],
    df: Any,
    start_dt: datetime,
    end_dt: datetime,
    bounds_lo: datetime,
    bounds_hi: datetime,
    key_dates: Sequence[datetime],
    key_date_notes: Dict[str, str],
    key_date_types: Dict[str, str],
    include_total_staffing: bool,
    min_staffing: float,
    facility_info: Dict[str, Any],
    facility_report_lib: Any,
    ein_detail_df: Any = None,
    state_code: str = "",
    macpac_standards: Optional[Dict[str, Any]] = None,
    case_mix_data: Optional[List[Dict[str, Any]]] = None,
    quarters_in_range: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute v3-only report kwargs and warnings from API payload."""
    section_order, include_sections, report_items = resolve_report_plan(payload)
    event_window_selections = parse_event_window_selection(payload.get("event_windows"))
    resolved_windows = build_resolved_event_windows(
        key_dates, key_date_notes, key_date_types, event_window_selections
    )

    def _period_metrics_fn(ws: datetime, we: datetime) -> Dict[str, Any]:
        pm = facility_report_lib.calculate_period_metrics(df, ws, we) or {}
        if min_staffing > 0:
            days_under = facility_report_lib.calculate_days_under_state_minimum(
                df, ws, we, min_staffing
            )
            if isinstance(days_under, dict):
                pm.update(days_under)
        return pm

    event_html = ""
    if include_sections.get("event_windows", True) and resolved_windows:
        event_html = generate_event_windows_section_html(
            resolved_windows,
            _period_metrics_fn,
            min_staffing,
            include_total_staffing,
        )

    meta = {
        "facility_name": facility_info.get("name") or "",
        "ccn": facility_info.get("provnum") or "",
        "report_period_label": format_period_label(start_dt, end_dt),
        "generated_date": datetime.now().strftime("%B %d, %Y"),
        "events_html": format_event_list_html(key_dates, key_date_notes, key_date_types)
        or "<li><em>No case events entered</em></li>",
        "windows_html": format_windows_list_html(resolved_windows)
        or "<li><em>No event windows selected</em></li>",
    }
    summary_html = build_v3_report_summary_html(meta)

    has_period_data = not getattr(df, "empty", True)
    warnings = collect_v3_warnings(
        report_start=start_dt,
        report_end=end_dt,
        data_lo=bounds_lo,
        data_hi=bounds_hi,
        key_dates=key_dates,
        event_windows=resolved_windows,
        has_period_data=has_period_data,
    )

    context_hints = collect_supporting_context_hints(
        report_start_iso=start_dt.strftime("%Y-%m-%d"),
        report_end_iso=end_dt.strftime("%Y-%m-%d"),
        data_lo_iso=bounds_lo.strftime("%Y-%m-%d"),
        data_hi_iso=bounds_hi.strftime("%Y-%m-%d"),
        key_date_isos=[d.strftime("%Y-%m-%d") for d in key_dates],
        event_window_ranges=[
            {
                "window_start": w.get("window_start").strftime("%Y-%m-%d")
                if hasattr(w.get("window_start"), "strftime")
                else w.get("window_start"),
                "window_end": w.get("window_end").strftime("%Y-%m-%d")
                if hasattr(w.get("window_end"), "strftime")
                else w.get("window_end"),
                "window_label": w.get("window_label"),
                "event_date": w.get("event_date"),
            }
            for w in resolved_windows
        ],
        resident_stay_start=str(payload.get("resident_stay_start") or "").strip() or None,
        resident_stay_end=str(payload.get("resident_stay_end") or "").strip() or None,
    )
    supporting_html = ""
    if include_sections.get("supporting_context", True) and context_hints:
        supporting_html = generate_supporting_context_html(context_hints)
        for hint in context_hints:
            msg = str(hint.get("message") or "").strip()
            if msg and msg not in warnings:
                warnings.append(msg)

    findings: List[Dict[str, Any]] = []
    findings_html = ""
    auto_findings = payload.get("auto_detect_findings", True) is not False
    if auto_findings and include_sections.get("key_staffing_findings", True):
        try:
            from report_findings_engine import (
                analyze_staffing_findings,
                findings_to_json,
                parse_category_filters,
                render_findings_section_html,
            )

            case_events = []
            for dt in key_dates:
                iso = dt.strftime("%Y-%m-%d")
                case_events.append(
                    {
                        "date": dt,
                        "note": key_date_notes.get(iso, ""),
                        "type": key_date_types.get(iso, "other"),
                    }
                )
            stay_period = _parse_resident_stay_period(payload)
            st = str(state_code or facility_info.get("state") or "").strip()
            thresholds = macpac_standards if isinstance(macpac_standards, dict) else {"min_staffing": min_staffing}
            from report_findings_engine import cap_findings_for_memo

            raw_findings = analyze_staffing_findings(
                facility_df=df,
                report_period=(start_dt, end_dt),
                case_events=case_events,
                event_windows=resolved_windows,
                resident_stay_period=stay_period,
                state_code=st,
                state_thresholds=thresholds,
                available_pbj_bounds=(bounds_lo, bounds_hi),
                ein_detail_df=ein_detail_df,
                include_total_staffing=include_total_staffing,
                category_filters=parse_category_filters(payload.get("finding_categories")),
                case_mix_data=case_mix_data,
                quarters_in_range=quarters_in_range,
            )
            memo_findings, total_n = cap_findings_for_memo(raw_findings)
            findings = findings_to_json(memo_findings)
            findings_html = render_findings_section_html(
                raw_findings, total_count=total_n
            )
            if payload.get("smart_section_flow", True) is not False and raw_findings:
                section_order = prioritize_section_order_from_findings(
                    section_order,
                    list(raw_findings),
                    has_case_events=bool(key_dates),
                    enabled=True,
                )
        except Exception:
            findings = []
            findings_html = ""

    return {
        "section_order": section_order,
        "include_sections": include_sections,
        "report_items": report_items,
        "v3_report_summary_html": summary_html,
        "event_windows_section_html": event_html,
        "supporting_context_section_html": supporting_html,
        "key_staffing_findings_section_html": findings_html,
        "staffing_findings": findings,
        "warnings": warnings,
    }
