"""
PBJ staffing bridge suggestions for citation topics (context only, not causation).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Optional

from citation_date_confidence import pbj_window_ranges


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=1)
def load_pbj_bridge_config(path: Optional[str] = None) -> dict[str, Any]:
    p = path or os.path.join(_repo_root(), "config", "citation_pbj_bridge.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_pbj_context(
    citation: dict[str, Any],
    *,
    topic_id: str,
    topic_pbj_bridge: Optional[dict[str, Any]] = None,
    date_context: Optional[dict[str, Any]] = None,
    bridge_cfg: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Suggest PBJ role families and date windows for one citation under a topic."""
    cfg = bridge_cfg or load_pbj_bridge_config()
    topic_bridge = topic_pbj_bridge or {}
    defaults = cfg.get("topic_defaults", {}).get(topic_id) or {}

    strength = str(
        topic_bridge.get("bridge_strength") or defaults.get("bridge_strength") or "weak_context_only"
    )
    family_key = str(topic_bridge.get("role_family") or defaults.get("role_family") or "nursing_all")
    families = cfg.get("role_families", {})
    family = families.get(family_key) or families.get("nursing_all") or {}

    pdf = citation.get("pdf_enrichment") or citation.get("pdf_extract") or {}
    mapped = list(pdf.get("staff_positions_mapped") or pdf.get("staff_positions_mentioned") or [])
    role_labels = [str(p.get("position_label") or "") for p in mapped if p.get("position_label")]
    ein_codes: list[int] = []
    for p in mapped:
        for c in p.get("ein_job_codes") or []:
            try:
                ein_codes.append(int(c))
            except (TypeError, ValueError):
                pass
    if not ein_codes:
        ein_codes = list(family.get("ein_job_codes") or [])
    if not role_labels:
        role_labels = list(family.get("labels") or [])

    dc = date_context or {}
    anchor = dc.get("anchor_date_used") or citation.get("survey_date")
    window_types = list(
        topic_bridge.get("default_windows") or cfg.get("default_windows") or ["survey_minus_7_plus_7"]
    )

    return {
        "role_families": role_labels,
        "ein_job_codes": sorted(set(ein_codes)),
        "windows": pbj_window_ranges(anchor, window_types),
        "bridge_strength": strength,
        "disclaimer": str(cfg.get("disclaimer") or ""),
        "can_support_exact_daily_pbj_match": bool(dc.get("can_support_exact_daily_pbj_match")),
    }
