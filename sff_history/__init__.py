"""Canonical PBJ320 SFF history data layer (Phase 1, CCN-backed era only).

Three layers, kept in separate immutable-vs-derived tables (see
``D:\\Projects\\pbj-sff-archive-audit\\outputs\\SFF_ARCHIVE_AUDIT.md`` S12 for
the full design rationale this module implements):

1. ``publications``   — one row per CMS SFF PDF publication (era3b_ccn only).
2. ``observations``    — one row per facility x table/category membership
   within one publication. Immutable raw evidence.
3. ``derived``          — OBSERVED_CHANGE / EXPLICIT_SOURCE_EVENT /
   DERIVED_INTERVAL facts computed from ``observations``. Never modifies
   layers 1-2.

Scope for this phase: March 2023 through the latest governed/current
publication (the CCN-backed "Era 3b" period only, per
``D:\\Projects\\pbj-sff-archive-audit\\outputs\\SFF_ARCHIVE_AUDIT.md``).
Pre-2023 (no-CCN) eras are explicitly out of scope.
"""
