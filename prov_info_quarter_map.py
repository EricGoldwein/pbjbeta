"""
Manual CMS processing-month → PBJ quarter labels (same rules as prov_info.py).

Kept free of Streamlit/pandas/plotly so pipeline scripts can import this without
loading the full prov_info Streamlit app.
"""

from __future__ import annotations

# Format: (processing_year, processing_month) -> "Qn YYYY" PBJ quarter label.
PROCESSING_MONTH_TO_PBJ_QUARTER: dict[tuple[int, int], str] = {
    (2018, 4): "Q4 2017",
    (2018, 5): "Q4 2017",
    (2018, 6): "Q4 2017",
    (2018, 7): "Q1 2018",
    (2018, 8): "Q1 2018",
    (2018, 9): "Q1 2018",
    (2018, 10): "Q2 2018",
    (2018, 11): "Q2 2018",
    (2018, 12): "Q2 2018",
    (2019, 1): "Q3 2018",
    (2019, 2): "Q3 2018",
    (2019, 3): "Q3 2018",
    (2019, 4): "Q4 2018",
    (2019, 5): "Q4 2018",
    (2019, 6): "Q4 2018",
    (2019, 7): "Q1 2019",
    (2019, 8): "Q1 2019",
    (2019, 9): "Q1 2019",
    (2019, 10): "Q2 2019",
    (2019, 11): "Q2 2019",
    (2020, 1): "Q3 2019",
    (2020, 2): "Q3 2019",
    (2020, 3): "Q3 2019",
    (2020, 4): "Q4 2019",
    (2020, 5): "Q4 2019",
    (2020, 6): "Q4 2019",
    (2020, 7): "Q4 2019",
    (2020, 8): "Q4 2019",
    (2020, 9): "Q4 2019",
    (2020, 10): "Q2 2020",
    (2020, 11): "Q2 2020",
    (2021, 1): "Q3 2020",
    (2021, 2): "Q3 2020",
    (2021, 3): "Q3 2020",
    (2021, 4): "Q4 2020",
    (2021, 5): "Q4 2020",
    (2021, 6): "Q4 2020",
    (2021, 7): "Q1 2021",
    (2021, 8): "Q1 2021",
    (2021, 9): "Q1 2021",
    (2021, 10): "Q2 2021",
    (2021, 11): "Q2 2021",
    (2022, 1): "Q3 2021",
    (2022, 2): "Q3 2021",
    (2022, 3): "Q3 2021",
    (2022, 4): "Q4 2021",
    (2022, 5): "Q4 2021",
    (2022, 6): "Q4 2021",
    (2022, 7): "Q1 2022",
    (2022, 8): "Q1 2022",
    (2022, 9): "Q1 2022",
    (2022, 10): "Q2 2022",
    (2022, 11): "Q2 2022",
    (2023, 1): "Q3 2022",
    (2023, 2): "Q3 2022",
    (2023, 3): "Q3 2022",
    (2023, 4): "Q4 2022",
    (2023, 5): "Q4 2022",
    (2023, 6): "Q4 2022",
    (2023, 7): "Q1 2023",
    (2023, 8): "Q1 2023",
    (2023, 9): "Q1 2023",
    (2023, 10): "Q2 2023",
    (2023, 11): "Q2 2023",
    (2024, 1): "Q3 2023",
    (2024, 2): "Q3 2023",
    (2024, 3): "Q3 2023",
    (2024, 4): "Q3 2023",
    (2024, 5): "Q3 2023",
    (2024, 6): "Q3 2023",
    (2024, 7): "Q1 2024",
    (2024, 8): "Q1 2024",
    (2024, 9): "Q1 2024",
    (2024, 10): "Q2 2024",
    (2024, 11): "Q2 2024",
    (2025, 2): "Q3 2024",
    (2025, 3): "Q3 2024",
    (2025, 4): "Q4 2024",
    (2025, 5): "Q4 2024",
    (2025, 6): "Q4 2024",
    (2025, 7): "Q1 2025",
    (2025, 8): "Q2 2025",
    (2025, 9): "Q2 2025",
    (2025, 10): "Q2 2025",
    (2025, 11): "Q2 2025",
    (2025, 12): "Q2 2025",
    (2026, 1): "Q2 2025",
    (2026, 2): "Q3 2025",
    (2026, 3): "Q4 2025",
    (2026, 4): "Q4 2025",
    (2026, 5): "Q4 2025",
    (2026, 6): "Q4 2025",
}


def get_manual_quarter_from_processing_month(proc_month: str | None) -> str | None:
    if proc_month is None:
        return None
    s = str(proc_month).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        year, month = map(int, s.split("-")[:2])
    except (ValueError, TypeError):
        return None
    return PROCESSING_MONTH_TO_PBJ_QUARTER.get((year, month))


def get_quarter_from_processing_month(
    proc_month: str | None,
    use_interval_fallback: bool = True,
) -> str | None:
    """Map CMS processing month (``YYYY-MM``) to PBJ quarter label without Streamlit/plotly."""
    manual = get_manual_quarter_from_processing_month(proc_month)
    if manual:
        return manual
    if not use_interval_fallback or not proc_month:
        return None
    return _interval_staffing_quarter_from_bundled_json(proc_month)


def _interval_staffing_quarter_from_bundled_json(proc_month: str) -> str | None:
    """Fallback using ``static/data/interval_quarter_mapping.json`` when shipped with the app."""
    try:
        year, month = map(int, str(proc_month).strip().split("-")[:2])
    except (ValueError, TypeError):
        return None
    mm_yyyy = f"{month:02d}-{year}"
    for row in _load_interval_quarter_mapping_rows():
        if str(row.get("processing_month") or "").strip() == mm_yyyy:
            quarter = str(row.get("interval_staffing_level_quarter") or "").strip()
            return quarter or None
    return None


_INTERVAL_JSON_ROWS: list[dict] | None = None


def _load_interval_quarter_mapping_rows() -> list[dict]:
    global _INTERVAL_JSON_ROWS
    if _INTERVAL_JSON_ROWS is not None:
        return _INTERVAL_JSON_ROWS
    import json
    from pathlib import Path

    candidates = [
        Path(__file__).resolve().parent / "static" / "data" / "interval_quarter_mapping.json",
        Path.cwd() / "static" / "data" / "interval_quarter_mapping.json",
    ]
    rows: list[dict] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            rows = list(data.get("rows") or [])
            break
        except Exception:
            continue
    _INTERVAL_JSON_ROWS = rows
    return rows
