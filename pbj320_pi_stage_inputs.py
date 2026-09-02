"""Provider Information Stage input graph and fingerprinting (baseline + overlay)."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

from active_release_registry import sha256_file

from pbj320_publication import (
    fingerprint_baseline_path,
    resolve_chain_performance_path,
    resolve_sff_facilities_path,
)

SOURCE_PI = "cms.provider_info"
SOURCE_SFF = "cms.sff_pdf_list"
SOURCE_NURSE = "cms.pbj_nurse_staffing"
SOURCE_CHAIN = "chain_performance"


def _overlay_input(
    *,
    source_id: str,
    release_id: str,
    rel_path: str,
    sha256: str,
    role: str,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "release_id": release_id,
        "mode": "GOVERNED_OVERLAY",
        "path": rel_path.replace("\\", "/"),
        "sha256": sha256,
        "role": role,
    }


def _baseline_input_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": row.get("source_id"),
        "release_id": row.get("release_id"),
        "mode": "PUBLISHED_BASELINE",
        "path": row.get("path"),
        "sha256": row.get("sha256"),
        "present_at_base": row.get("present_at_base"),
        "role": row.get("role"),
    }


def collect_pi_overlay_fingerprints(
    baseline_wt: Path,
    *,
    release_id: str,
    rel_paths: dict[str, str],
    norm_sha: str,
    nh_sha: str | None,
    combined_sha: str,
) -> dict[str, dict[str, Any]]:
    """Overlays applied to baseline worktree before builders run."""
    out: dict[str, dict[str, Any]] = {}
    out["norm"] = _overlay_input(
        source_id=SOURCE_PI,
        release_id=release_id,
        rel_path=rel_paths["norm"],
        sha256=norm_sha,
        role="provider_norm",
    )
    if nh_sha:
        out["nh_snapshot"] = _overlay_input(
            source_id=SOURCE_PI,
            release_id=release_id,
            rel_path=rel_paths["nh"],
            sha256=nh_sha,
            role="nh_snapshot_parity",
        )
    out["combined_latest"] = _overlay_input(
        source_id=SOURCE_PI,
        release_id=release_id,
        rel_path=rel_paths["combined_latest"],
        sha256=combined_sha,
        role="provider_combined_latest",
    )
    return out


def collect_shared_baseline_inputs(
    dev_pbj_root: Path,
    baseline_wt: Path,
    publication_base_sha: str,
) -> dict[str, dict[str, Any]]:
    """Fingerprint non-PI inputs at publication_base_sha (from baseline worktree tree)."""
    rows: dict[str, dict[str, Any]] = {}

    nurse_paths = [
        "facility_quarterly_metrics.csv",
        "national_quarterly_metrics.csv",
        "state_quarterly_metrics.csv",
    ]
    for rel in nurse_paths:
        fp = fingerprint_baseline_path(dev_pbj_root, publication_base_sha, rel)
        fp["source_id"] = SOURCE_NURSE
        fp["release_id"] = None
        fp["role"] = "staffing_quarterly"
        rows[rel.replace("\\", "/")] = fp

    states_rel = "states_list.json"
    fp = fingerprint_baseline_path(dev_pbj_root, publication_base_sha, states_rel)
    fp["source_id"] = None
    fp["release_id"] = None
    fp["role"] = "states_list"
    rows[states_rel] = fp

    sff_rel = resolve_sff_facilities_path(baseline_wt)
    if sff_rel:
        sff_fp = fingerprint_baseline_path(dev_pbj_root, publication_base_sha, sff_rel)
        sff_fp["source_id"] = SOURCE_SFF
        sff_fp["release_id"] = None
        sff_fp["role"] = "sff_facilities_json"
        rows[sff_rel] = sff_fp
    else:
        rows["__sff_missing__"] = {
            "source_id": SOURCE_SFF,
            "release_id": None,
            "mode": "PUBLISHED_BASELINE",
            "path": None,
            "sha256": None,
            "present_at_base": False,
            "role": "sff_facilities_json",
        }

    chain_rel = resolve_chain_performance_path(baseline_wt)
    if chain_rel:
        chain_fp = fingerprint_baseline_path(dev_pbj_root, publication_base_sha, chain_rel)
        chain_fp["source_id"] = SOURCE_CHAIN
        chain_fp["release_id"] = None
        chain_fp["role"] = "chain_performance_csv"
        rows[chain_rel] = chain_fp
    else:
        rows["__chain_missing__"] = {
            "source_id": SOURCE_CHAIN,
            "release_id": None,
            "mode": "PUBLISHED_BASELINE",
            "path": None,
            "sha256": None,
            "present_at_base": False,
            "role": "chain_performance_csv",
        }

    return rows


def inputs_for_provider_norm(overlay: dict[str, Any]) -> list[dict[str, Any]]:
    return [overlay]


def inputs_for_combined_latest(overlay: dict[str, Any], norm_overlay: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **norm_overlay,
            "mode": "GOVERNED_OVERLAY",
            "note": "combined built from governed Norm via PBJapp builder",
        }
    ]


def inputs_for_state_aggregates(
    *,
    overlay: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for key in ("norm", "combined_latest"):
        if key in overlay:
            inputs.append(_overlay_input_row(overlay[key]))
    for rel in (
        "facility_quarterly_metrics.csv",
        "national_quarterly_metrics.csv",
        "state_quarterly_metrics.csv",
    ):
        if rel in baseline:
            inputs.append(_baseline_input_row(baseline[rel]))
    return inputs


def inputs_for_search_index(
    *,
    overlay: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    nh_overlay: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    if nh_overlay:
        inputs.append(_overlay_input_row(nh_overlay))
    else:
        inputs.append(_overlay_input_row(overlay["combined_latest"]))
    for key, row in baseline.items():
        if key.startswith("__"):
            if not row.get("present_at_base") and row.get("role") in {
                "sff_facilities_json",
                "chain_performance_csv",
            }:
                continue
            inputs.append(_baseline_input_row(row))
        elif row.get("role") in {"sff_facilities_json", "chain_performance_csv", "states_list"}:
            inputs.append(_baseline_input_row(row))
    return inputs


def _overlay_input_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": row.get("source_id"),
        "release_id": row.get("release_id"),
        "mode": row.get("mode") or "GOVERNED_OVERLAY",
        "path": row.get("path"),
        "sha256": row.get("sha256"),
        "role": row.get("role"),
    }


def audit_pi_destination_inputs() -> list[dict[str, Any]]:
    """Read-only input graph for the four PI commit destinations."""
    return [
        {
            "destination": "provider_info/ProviderInfoNorm_2026_08.csv",
            "publication_class": "commit_destination",
            "inputs": [
                {"source_id": SOURCE_PI, "mode": "GOVERNED_OVERLAY", "note": "ACTIVE canonical Norm"},
            ],
        },
        {
            "destination": "provider_info_combined_latest.csv",
            "publication_class": "commit_destination",
            "inputs": [
                {"source_id": SOURCE_PI, "mode": "GOVERNED_OVERLAY", "note": "Norm via PBJapp combined builder"},
            ],
        },
        {
            "destination": "data/state_page_aggregates.json.gz",
            "publication_class": "shared_derived",
            "inputs": [
                {"source_id": SOURCE_PI, "mode": "GOVERNED_OVERLAY", "paths": ["Norm", "combined_latest"]},
                {"source_id": SOURCE_NURSE, "mode": "PUBLISHED_BASELINE", "paths": [
                    "facility_quarterly_metrics.csv",
                    "national_quarterly_metrics.csv",
                    "state_quarterly_metrics.csv",
                ]},
            ],
            "verified_from": "pbj-root/scripts/build_state_page_aggregates.py → app.py loaders",
        },
        {
            "destination": "search_index.json",
            "publication_class": "shared_derived",
            "inputs": [
                {"source_id": SOURCE_PI, "mode": "GOVERNED_OVERLAY", "paths": ["NH snapshot or combined_latest"]},
                {"source_id": SOURCE_SFF, "mode": "PUBLISHED_BASELINE", "paths": ["data/derived/sff/sff_facilities.json", "..."]},
                {"source_id": SOURCE_CHAIN, "mode": "PUBLISHED_BASELINE", "paths": ["ownership/Nursing_Home_Chain_Performance_Measures_*.csv"]},
                {"mode": "PUBLISHED_BASELINE", "paths": ["states_list.json"]},
            ],
            "verified_from": "pbj-root/generate_search_index.py",
        },
    ]


def unresolved_inputs(baseline: dict[str, dict[str, Any]]) -> list[str]:
    """Return UNRESOLVED roles — none expected when baseline fingerprints succeed."""
    unresolved: list[str] = []
    for key, row in baseline.items():
        if row.get("mode") == "UNRESOLVED":
            unresolved.append(str(row.get("role") or key))
    return unresolved
