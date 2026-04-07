"""
Manual CMS processing-month → PBJ quarter labels (same rules as prov_info.py).

Kept free of Streamlit/pandas/plotly so Flask/Vercel can import this for
`/data-matching` and other server paths without loading the full prov_info app.
"""

from __future__ import annotations

# Format: (processing_year, processing_month) -> "Qn YYYY" PBJ quarter label.
# Duplicate keys follow the same last-wins semantics as the original prov_info dict.
PROCESSING_MONTH_TO_PBJ_QUARTER: dict[tuple[int, int], str] = {
    # 2017Q4
    (2018, 4): "Q4 2017",
    (2018, 5): "Q4 2017",
    (2018, 6): "Q4 2017",
    # 2018Q1
    (2018, 7): "Q1 2018",
    (2018, 8): "Q1 2018",
    (2018, 9): "Q1 2018",
    # 2018Q2
    (2018, 10): "Q2 2018",
    (2018, 11): "Q2 2018",
    (2018, 12): "Q2 2018",
    # 2018Q3
    (2019, 1): "Q3 2018",
    (2019, 2): "Q3 2018",
    (2019, 3): "Q3 2018",
    # 2018Q4
    (2019, 4): "Q4 2018",
    (2019, 5): "Q4 2018",
    (2019, 6): "Q4 2018",
    # 2019Q1
    (2019, 7): "Q1 2019",
    (2019, 8): "Q1 2019",
    (2019, 9): "Q1 2019",
    # 2019Q2
    (2019, 10): "Q2 2019",
    (2019, 11): "Q2 2019",
    # 2019Q3
    (2020, 1): "Q3 2019",
    (2020, 2): "Q3 2019",
    (2020, 3): "Q3 2019",
    # 2019Q4 (spans 6 months!)
    (2020, 4): "Q4 2019",
    (2020, 5): "Q4 2019",
    (2020, 6): "Q4 2019",
    (2020, 7): "Q4 2019",
    (2020, 8): "Q4 2019",
    (2020, 9): "Q4 2019",
    # 2020Q2 (only 2 months!)
    (2020, 10): "Q2 2020",
    (2020, 11): "Q2 2020",
    # 2020Q3
    (2021, 1): "Q3 2020",
    (2021, 2): "Q3 2020",
    (2021, 3): "Q3 2020",
    # 2020Q4
    (2021, 4): "Q4 2020",
    (2021, 5): "Q4 2020",
    (2021, 6): "Q4 2020",
    # 2021Q1
    (2021, 7): "Q1 2021",
    (2021, 8): "Q1 2021",
    (2021, 9): "Q1 2021",
    # 2021Q2
    (2021, 10): "Q2 2021",
    (2021, 11): "Q2 2021",
    # 2021Q3
    (2022, 1): "Q3 2021",
    (2022, 2): "Q3 2021",
    (2022, 3): "Q3 2021",
    # 2021Q4
    (2022, 4): "Q4 2021",
    (2022, 5): "Q4 2021",
    (2022, 6): "Q4 2021",
    # 2022Q1
    (2022, 7): "Q1 2022",
    (2022, 8): "Q1 2022",
    (2022, 9): "Q1 2022",
    # 2022Q2
    (2022, 10): "Q2 2022",
    (2022, 11): "Q2 2022",
    # 2022Q3
    (2023, 1): "Q3 2022",
    (2023, 2): "Q3 2022",
    (2023, 3): "Q3 2022",
    # 2022Q4
    (2023, 4): "Q4 2022",
    (2023, 5): "Q4 2022",
    (2023, 6): "Q4 2022",
    # 2023Q1
    (2023, 7): "Q1 2023",
    (2023, 8): "Q1 2023",
    (2023, 9): "Q1 2023",
    # 2023Q2 (spans 9 months!)
    (2023, 10): "Q2 2023",
    (2023, 11): "Q2 2023",
    (2024, 1): "Q2 2023",
    (2024, 2): "Q2 2023",
    (2024, 3): "Q2 2023",
    (2024, 4): "Q2 2023",
    (2024, 5): "Q2 2023",
    (2024, 6): "Q2 2023",
    # 2023Q3 (spans 6 months! — last wins for 2024-01..06)
    (2024, 1): "Q3 2023",
    (2024, 2): "Q3 2023",
    (2024, 3): "Q3 2023",
    (2024, 4): "Q3 2023",
    (2024, 5): "Q3 2023",
    (2024, 6): "Q3 2023",
    # 2024Q1
    (2024, 7): "Q1 2024",
    (2024, 8): "Q1 2024",
    (2024, 9): "Q1 2024",
    # 2024Q2
    (2024, 10): "Q2 2024",
    (2024, 11): "Q2 2024",
    # 2024Q3 (only 2 months!)
    (2025, 2): "Q3 2024",
    (2025, 3): "Q3 2024",
    # 2024Q4
    (2025, 4): "Q4 2024",
    (2025, 5): "Q4 2024",
    (2025, 6): "Q4 2024",
    # 2025Q1
    (2025, 7): "Q1 2025",
    # 2025Q2 — December 2025 files contain Q2 2025 data
    (2025, 12): "Q2 2025",
}


def get_manual_quarter_from_processing_month(proc_month: str | None) -> str | None:
    """Return PBJ quarter label for ``YYYY-MM`` processing month, or None if unknown."""
    if proc_month is None:
        return None
    s = str(proc_month).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        year, month = map(int, s.split("-"))
    except (ValueError, TypeError):
        return None
    return PROCESSING_MONTH_TO_PBJ_QUARTER.get((year, month))
