"""
Config-driven citation topics and narrative role mapping.

See ``docs/CITATION_TOPIC_REGISTRY_PLAN.md`` and ``config/citation_topic_registry.json``.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Optional

from citation_lib import (
    _severity_rank_for_code,
    enrich_citations_dataframe,
    get_facility_citations,
    is_abuse_related_citation_record,
    load_citation_severity_config,
    normalize_citation_record,
)
import pandas as pd


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=1)
def load_topic_registry(path: Optional[str] = None) -> dict[str, Any]:
    p = path or os.path.join(_repo_root(), "config", "citation_topic_registry.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_narrative_role_map(path: Optional[str] = None) -> dict[str, Any]:
    p = path or os.path.join(_repo_root(), "config", "citation_narrative_role_map.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def narrative_roles_from_config(
    role_mentions: list[str],
    *,
    role_map: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Map PDF role phrases using ``citation_narrative_role_map.json`` (conservative)."""
    cfg = role_map or load_narrative_role_map()
    compiled: list[tuple[re.Pattern[str], dict[str, Any]]] = []
    for entry in cfg.get("mappings", []):
        flags_raw = entry.get("flags", 0)
        if isinstance(flags_raw, str):
            flags = re.I if "i" in flags_raw.lower() else 0
        else:
            flags = int(flags_raw or 0)
        pat = re.compile(str(entry["pattern"]), flags)
        compiled.append((pat, entry))

    positions: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for raw in role_mentions or []:
        text = re.sub(r"\s+", " ", str(raw or "").strip())
        if not text:
            continue
        for pat, entry in compiled:
            if not pat.search(text):
                continue
            label = str(entry.get("position_label") or "").strip()
            if not label or label in seen_labels:
                break
            seen_labels.add(label)
            positions.append(
                {
                    "position_label": label,
                    "ein_job_codes": list(entry.get("ein_job_codes") or []),
                    "confidence": str(entry.get("confidence") or "medium"),
                    "narrative_mentions": [text],
                }
            )
            break
    return positions


def _blob(citation: dict[str, Any]) -> str:
    return " ".join(
        [
            str(citation.get("deficiency_category") or ""),
            str(citation.get("deficiency_description") or ""),
            str(citation.get("deficiency_tag") or ""),
        ]
    ).lower()


def _match_rule(citation: dict[str, Any], rule: dict[str, Any], *, cfg: dict[str, Any]) -> bool:
    tag = str(citation.get("deficiency_tag") or "").upper()
    cat = str(citation.get("deficiency_category") or "").lower()

    if rule.get("is_complaint") and not citation.get("is_complaint_deficiency"):
        return False
    if rule.get("is_infection_control") and not citation.get("is_infection_control_deficiency"):
        return False

    min_sev = rule.get("min_scope_severity")
    if min_sev:
        floor = str(min_sev).strip().upper()[:1]
        sev_cfg = load_citation_severity_config()
        if _severity_rank_for_code(str(citation.get("scope_severity_code") or ""), sev_cfg) < _severity_rank_for_code(
            floor, sev_cfg
        ):
            return False

    if "category_contains" in rule:
        needle = str(rule["category_contains"]).lower()
        if needle not in cat:
            return False
        return True

    exact = rule.get("tags_exact") or []
    if exact and tag in {str(t).upper() for t in exact}:
        return True

    prefixes = rule.get("tags_prefix") or []
    for pref in prefixes:
        p = str(pref).upper()
        if tag.startswith(p):
            return True

    for rx in rule.get("description_regex") or []:
        if re.search(rx, _blob(citation), re.I):
            return True

    if exact or prefixes or rule.get("description_regex"):
        return False
    if "category_contains" in rule:
        return False
    return bool(rule.get("is_complaint") or rule.get("is_infection_control") or min_sev)


def _topic_status(topic: dict[str, Any]) -> str:
    return str(topic.get("status") or "active").strip().lower()


def get_registry_topics(
    *,
    registry: Optional[dict[str, Any]] = None,
    active_only: bool = True,
    topic_ids: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    reg = registry or load_topic_registry()
    topics = list(reg.get("topics") or [])
    if topic_ids:
        want = {t.strip() for t in topic_ids if t}
        topics = [t for t in topics if str(t.get("id")) in want]
    if active_only:
        topics = [t for t in topics if _topic_status(t) == "active"]
    return topics


def match_citation_topics(
    citation: dict[str, Any],
    *,
    registry: Optional[dict[str, Any]] = None,
    active_only: bool = True,
) -> list[str]:
    """Return topic ids that match a normalized citation record."""
    reg = registry or load_topic_registry()
    tag = str(citation.get("deficiency_tag") or "").upper()
    matched: list[str] = []

    for topic in get_registry_topics(registry=reg, active_only=active_only):
        tid = str(topic.get("id") or "")
        if not tid:
            continue
        excluded = {str(t).upper() for t in topic.get("exclude_tags") or []}
        if tag in excluded:
            continue
        rules = topic.get("match_any") or []
        if not rules:
            continue
        if any(_match_rule(citation, rule, cfg=reg) for rule in rules):
            matched.append(tid)
    return matched


def _topic_sort_key(citation: dict[str, Any], topic: dict[str, Any]) -> tuple:
    cfg = load_citation_severity_config()
    cycle = citation.get("inspection_cycle")
    cycle_pri = 0 if cycle == 1 else 1
    complaint_pri = 0 if citation.get("is_complaint_deficiency") else 1
    sev_pri = -_severity_rank_for_code(str(citation.get("scope_severity_code") or ""), cfg)
    survey = citation.get("survey_date") or ""
    tag = str(citation.get("deficiency_tag") or "").upper()
    tag_order = {str(t).upper(): i for i, t in enumerate(topic.get("tag_priority") or [])}
    tag_pri = tag_order.get(tag, 999)
    return (cycle_pri, complaint_pri, sev_pri, tag_pri, survey)


def _dedupe_citations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for c in rows:
        key = (str(c.get("survey_date") or ""), str(c.get("deficiency_tag") or "").upper())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def get_facility_citation_slices(
    ccn: str,
    citations_df: pd.DataFrame,
    *,
    topic_ids: Optional[list[str]] = None,
    limit: Optional[int] = None,
    enrich_pdf: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """
    Return ranked citation lists per topic (default limit from registry, usually 10).

    Keys are topic ids; values are normalized citation dicts, newest/severe first.
    """
    reg = load_topic_registry()
    default_limit = int(limit if limit is not None else reg.get("default_limit", 10))
    all_c = get_facility_citations(ccn, citations_df)
    if enrich_pdf:
        from citation_lib import _enrich_citations_with_pdf_extract

        all_c = _enrich_citations_with_pdf_extract(ccn, all_c)

    include_todo = bool(topic_ids and any(
        str(t).strip() in {str(x.get("id")) for x in reg.get("topics", []) if _topic_status(x) == "todo"}
        for t in topic_ids
    ))
    topics = get_registry_topics(
        registry=reg,
        active_only=not include_todo,
        topic_ids=topic_ids,
    )

    out: dict[str, list[dict[str, Any]]] = {}
    for topic in topics:
        tid = str(topic.get("id") or "")
        if not tid:
            continue
        lim = int(topic.get("default_limit") or default_limit)
        pool = [c for c in all_c if tid in match_citation_topics(c, registry=reg)]
        pool = _dedupe_citations(pool)
        pool.sort(key=lambda c: _topic_sort_key(c, topic))
        out[tid] = pool[:lim]
    return out


def legacy_abuse_matches(citation: dict[str, Any]) -> bool:
    """Old heuristic vs topic pack — for regression comparisons."""
    return is_abuse_related_citation_record(citation)
