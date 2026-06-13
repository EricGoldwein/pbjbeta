"""
data.cms.gov PBJ explorer rules shared by Python URL builders and imports.

Older quarterly slices expect lowercase ``provnum`` / ``workdate`` in filter JSON;
newer slices use ``PROVNUM`` / ``WorkDate``. The nurse daily staffing dataset also
uses a rolling root endpoint without ``/q1-2025`` for the current 2025 Q1 slice.

**Quarter coverage is not a flat “everything before 2020” rule.** For example
``2018Q1``–``2018Q3`` use uppercase ``PROVNUM`` / ``WorkDate``; ``2018Q4`` is in
the legacy set. Human-readable tables and code touch-points live in
``docs/CMS_DATAGOV_PBJ_EXPLORER.md``. The authoritative set is
``CMS_DATAGOV_LEGACY_PROV_WORKDATE_COLUMN_QUARTERS`` below—keep JS URL builders in
sync with this module (or call these helpers from Python only).
"""

from __future__ import annotations

import re
from typing import Optional

# Quarters where explorer filter JSON must use lowercase provnum + workdate (not PROVNUM/WorkDate).
CMS_DATAGOV_LEGACY_PROV_WORKDATE_COLUMN_QUARTERS: frozenset[str] = frozenset(
    {
        "2017Q1",
        "2017Q2",
        "2017Q3",
        "2017Q4",
        "2018Q4",
        "2019Q1",
        "2019Q2",
        "2019Q3",
        "2019Q4",
        "2020Q2",
        "2020Q3",
    }
)


def cms_datagov_canonical_quarter_key(quarter: str) -> Optional[str]:
    """
    Normalize ``CY2019Q3``, ``2019Q3``, or ``cy2019q3`` to ``2019Q3``.

    Returns:
        Canonical ``yyyyQn`` or None if the string does not match a calendar quarter.
    """
    s = str(quarter or "").strip().upper().replace("\ufeff", "")
    m = re.match(r"^(?:CY)?(\d{4})Q([1-4])$", s)
    if not m:
        return None
    return f"{m.group(1)}Q{m.group(2)}"


def cms_datagov_legacy_lowercase_prov_workdate_keys(quarter: str) -> bool:
    """True when daily PBJ / employee-detail explorer JSON should use lowercase provnum and workdate."""
    c = cms_datagov_canonical_quarter_key(quarter)
    return c in CMS_DATAGOV_LEGACY_PROV_WORKDATE_COLUMN_QUARTERS if c else False


# Backwards-compatible alias used in facility_ein_lib.
cms_datagov_explorer_legacy_lowercase_keys = cms_datagov_legacy_lowercase_prov_workdate_keys


def cms_datagov_nurse_daily_uses_rolling_data_endpoint(year: int, calendar_quarter: int) -> bool:
    """
    Nurse daily staffing: CMS currently serves 2025 Q1 from the rolling ``/data`` endpoint
    (no ``/data/q1-2025`` path segment), matching dashboard JS ``populateQuarterlyDataTable``.
    """
    return year == 2025 and calendar_quarter == 1
