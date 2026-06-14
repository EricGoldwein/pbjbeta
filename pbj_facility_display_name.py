"""Server-side compact facility display names for SSR profile/header labels.

Mirrors the suffix-stripping compact path in static/js/pbj_facility_display_name.js.
Display-layer only; does not mutate the official CMS provider name.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_JS_SUFFIXES_PATH = (
    Path(__file__).resolve().parent / "static" / "js" / "pbj_facility_display_name.js"
)

# Minimal fallback if the JS source file is unavailable (e.g. partial deploy).
_FALLBACK_REMOVABLE_SUFFIXES: tuple[str, ...] = (
    "Rehabilitation and Nursing Center",
    "Rehabilitation & Nursing Center",
    "Nursing and Rehabilitation Center",
    "Nursing & Rehabilitation Center",
    "Rehabilitation and Nursing",
    "Rehabilitation & Nursing",
    "Healthcare Center",
    "Health Care Center",
    "Rehabilitation Center",
    "Nursing Center",
    "Care Center",
    "Healthcare and Rehabilitation Center",
    "Health Care and Rehabilitation Center",
)

_TITLE_MINOR = frozenset(
    {
        "and",
        "of",
        "at",
        "by",
        "the",
        "a",
        "an",
        "in",
        "on",
        "for",
        "to",
        "with",
    }
)


def _collapse_whitespace(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\u00a0", " ")).strip()


def _normalize_spacing(name: object) -> str:
    text = _collapse_whitespace(name)
    text = re.sub(r"\s*&\s*", " & ", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return _collapse_whitespace(text)


def _apply_abbreviations(name: str) -> str:
    out = str(name or "")
    out = re.sub(r"\bRehabilitation\b", "Rehab", out, flags=re.I)
    out = re.sub(r"\bRehabilita\b", "Rehab", out, flags=re.I)
    out = re.sub(r"\bHealth Care\b", "Health", out, flags=re.I)
    out = re.sub(r"\bHealthcare\b", "Health", out, flags=re.I)
    out = re.sub(r"\bPost-Acute\b", "Post Acute", out, flags=re.I)
    return _collapse_whitespace(out)


def _letters_mostly_upper(value: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", value)
    if not letters:
        return False
    upper = len(re.sub(r"[^A-Z]", "", letters))
    return upper / len(letters) >= 0.65


def _title_case_token(word: str, index: int) -> str:
    raw = str(word or "")
    if not raw:
        return raw
    if raw in {"&", "-"}:
        return raw
    key = raw.lower().replace(".", "")
    if re.match(r"^[A-Z]\.[A-Z]\.$", raw, flags=re.I):
        return raw[0].upper() + "." + raw[2].upper() + "."
    if re.match(r"^[A-Z]\.$", raw):
        return raw
    if re.match(r"^St\.?$", raw, flags=re.I):
        return "St."
    if re.match(r"^Mt\.?$", raw, flags=re.I):
        return "Mt."
    if index > 0 and key in _TITLE_MINOR:
        return key
    if len(raw) == 1:
        return raw.upper()
    return raw[:1].upper() + raw[1:].lower()


def _normalize_display_casing(label: str, raw_full: str) -> str:
    text = _collapse_whitespace(label)
    if not text:
        return text
    if not _letters_mostly_upper(raw_full or text) and not re.search(
        r"\b(?:Rehab|Health|Healthcare|Nursing|Health Care)\s+AND\s+[A-Z]", text
    ):
        return text
    return " ".join(_title_case_token(word, idx) for idx, word in enumerate(text.split()))


@lru_cache(maxsize=1)
def _load_removable_suffixes() -> tuple[str, ...]:
    if _JS_SUFFIXES_PATH.is_file():
        js = _JS_SUFFIXES_PATH.read_text(encoding="utf-8")
        match = re.search(r"var REMOVABLE_SUFFIXES = \[([\s\S]*?)\];", js)
        if match:
            parsed = tuple(
                item.strip().strip("'")
                for item in re.findall(r"'([^']*)'", match.group(1))
            )
            if parsed:
                return parsed
    return _FALLBACK_REMOVABLE_SUFFIXES


def _suffix_token_pattern(token: str) -> str:
    lower = token.lower().replace(".", "")
    if lower == "&":
        return r"(?:&|and)"
    if lower in {"center", "centre"}:
        return r"(?:Center|Centre|Cente|Cent|Cen|Ce)(?:er|re)?"
    if lower == "rehabilitation":
        return r"(?:Rehabilitation|Rehabilita(?:tion)?)"
    if lower == "nursing":
        return r"(?:Nursing|Nursin(?:g)?)"
    if lower == "facility":
        return r"(?:Facility|Facilit(?:y)?)"
    if lower == "healthcare":
        return r"(?:Healthcare|Health\s*Care)"
    if lower == "care" and token == "Care":
        return "Care"
    if lower == "health" and token == "Health":
        return "Health"
    return re.escape(token)


def _suffix_pattern_regex(suffix: str) -> re.Pattern[str]:
    parts = [_suffix_token_pattern(part) for part in _normalize_spacing(suffix).split()]
    pattern = r"(?:[\s,\-]+|^)" + r"\s+".join(parts) + r"\s*$"
    return re.compile(pattern, flags=re.I)


def _strip_trailing_junk(name: str) -> str:
    out = _collapse_whitespace(name)
    prev = None
    while out != prev:
        prev = out
        out = re.sub(r"\b(and|&|at|of|for|the|with|skilled|llc)\s*$", "", out, flags=re.I)
        out = _collapse_whitespace(out)
    return out


def _is_bad_short_name(candidate: str, full_name: str) -> bool:
    """Skip suffix-stripped labels JS would reject (mirrors healthPlaceOnly solo-word guard)."""
    short = _collapse_whitespace(candidate)
    full = _collapse_whitespace(full_name)
    if not short:
        return True
    words = short.split()
    if len(words) == 1:
        health_place = re.match(
            r"^(\S+)\s+(Health Care|Healthcare)\s+(Center|Centre|Cente|Cent|Facilit(?:y)?)\s*$",
            full,
            flags=re.I,
        )
        if health_place and health_place.group(1).lower() == words[0].lower():
            return True
    return False


def _remove_trailing_suffix(name: str, suffix_list: tuple[str, ...]) -> tuple[str, str | None]:
    working = _normalize_spacing(name)
    removed_suffix: str | None = None
    changed = True
    while changed:
        changed = False
        for suffix in suffix_list:
            regex = _suffix_pattern_regex(suffix)
            if not regex.search(working):
                continue
            candidate = _collapse_whitespace(regex.sub("", working))
            if not candidate or len(candidate) < 2:
                continue
            if _is_bad_short_name(candidate, name):
                continue
            working = candidate
            removed_suffix = removed_suffix or suffix
            changed = True
            break
    return working, removed_suffix


def compact_facility_display_name(full_name: object) -> str:
    """Return the compact profile/header label for a CMS facility name."""
    raw_cms = str(full_name or "").replace("\u00a0", " ").strip()
    full = _normalize_spacing(raw_cms)
    if not full:
        return ""
    suffixes = _load_removable_suffixes()
    prefix, removed = _remove_trailing_suffix(full, suffixes)
    if removed:
        working = _apply_abbreviations(_strip_trailing_junk(prefix))
    else:
        working = _apply_abbreviations(full)
    working = _normalize_display_casing(working, full)
    if not working:
        return _normalize_display_casing(full, full)
    return working


@lru_cache(maxsize=512)
def get_facility_name_for_context(full_name: str, context: str = "compact") -> str:
    """Return compact display label (compact context only; other contexts pass through trimmed)."""
    name = str(full_name or "").strip()
    if not name:
        return ""
    ctx = str(context or "compact").lower()
    if ctx in {"hero", "profile", "export", "methodology", "source", "legal", "citation"}:
        return _normalize_spacing(name)
    return compact_facility_display_name(name)
