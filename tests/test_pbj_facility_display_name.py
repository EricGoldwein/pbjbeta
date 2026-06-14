"""Smoke tests for server-side compact facility display names."""

from __future__ import annotations

from pathlib import Path

import pytest

from pbj_facility_display_name import (
    compact_facility_display_name,
    get_facility_name_for_context,
)

REPO = Path(__file__).resolve().parents[1]
V2_TEMPLATE = REPO / "templates" / "superdynamic_dashboard_v2.html"


@pytest.mark.parametrize(
    ("full_name", "expected_compact"),
    [
        (
            "Mount Holly Rehabilitation and Care Center",
            "Mount Holly Rehab",
        ),
        (
            "Sunny Acres Nursing and Rehabilitation Center",
            "Sunny Acres",
        ),
        (
            "Oakwood Healthcare Center",
            "Oakwood Health Center",
        ),
        (
            "Riverside Care Center",
            "Riverside",
        ),
    ],
)
def test_compact_facility_display_name_suffix_smoke(
    full_name: str, expected_compact: str
) -> None:
    assert compact_facility_display_name(full_name) == expected_compact
    assert get_facility_name_for_context(full_name, "compact") == expected_compact


def test_compact_name_without_js_suffix_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vercel/no-Node path: fallback suffix list still yields expected compact labels."""
    from pbj_facility_display_name import _load_removable_suffixes

    monkeypatch.setattr(
        "pbj_facility_display_name._JS_SUFFIXES_PATH",
        Path("/nonexistent/pbj_facility_display_name.js"),
    )
    _load_removable_suffixes.cache_clear()
    get_facility_name_for_context.cache_clear()

    assert (
        get_facility_name_for_context(
            "Mount Holly Rehabilitation and Care Center", "compact"
        )
        == "Mount Holly Rehab"
    )
    assert (
        get_facility_name_for_context("Riverside Care Center", "compact") == "Riverside"
    )


def test_v2_template_renders_compact_profile_title() -> None:
    full = "Mount Holly Rehabilitation and Care Center"
    compact = get_facility_name_for_context(full, "compact")
    snippet = V2_TEMPLATE.read_text(encoding="utf-8")
    assert (
        'id="pbjProfilePodTitle">Profile: {{ facility_name_compact|default(facility_name_display) }}'
        in snippet
    )
    rendered = f"Profile: {compact}"
    assert rendered == "Profile: Mount Holly Rehab"
    assert "Rehabilitation and Care Center" not in rendered


def test_profile_title_sync_is_noop_when_ssr_matches() -> None:
    js = V2_TEMPLATE.read_text(encoding="utf-8")
    assert "titleEl.textContent !== nextTitle" in js
