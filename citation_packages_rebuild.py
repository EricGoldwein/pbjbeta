"""Post-activation facility citation package rebuild (local PBJapp slices only).

Orchestrates the same path as ``create_vercel_deployment`` citations domain:
``gate_citations_slice`` → ``build_facility_citations_csv`` → ``write_artifact_provenance``.
No Vercel deploy, no CMS re-download.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cms_data_paths
from active_release_registry import get_active_release, registry_path
from operator_freshness import count_stale_citation_packages


class CitationPackagesRebuildError(RuntimeError):
    pass


def _pbjapp_root() -> Path:
    configured = (os.environ.get("PBJ_REPO_ROOT") or "").strip()
    if configured:
        return Path(configured).resolve()
    return cms_data_paths.repo_root()


def _pbjapp_code_root() -> Path:
    """PBJapp tree that hosts ``active_release_client`` / ``citation_lib`` (may differ from data root in tests)."""
    candidate = _pbjapp_root()
    if (candidate / "active_release_client.py").is_file():
        return candidate
    sibling = Path(__file__).resolve().parent.parent / "PBJapp"
    if (sibling / "active_release_client.py").is_file():
        return sibling.resolve()
    return candidate


def _ensure_pbjapp_path() -> None:
    root_str = str(_pbjapp_code_root())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def citation_slice_needs_rebuild(pbj_root: Path, citations_out: str) -> tuple[bool, str]:
    """Registry gate from PBJapp ``active_release_client`` (same as ``create_vercel_deployment``)."""
    target = Path(citations_out)
    if not target.is_file():
        return True, "facility citations slice missing"
    _ensure_pbjapp_path()
    from active_release_client import artifact_release_state, load_active_release, validate_active_source

    release = load_active_release("cms.health_citations")
    validate_active_source(release)
    state, reason = artifact_release_state(str(target), release)
    return state == "BUILD", reason


def count_stale_citation_packages(
    *,
    root: Path | None = None,
    pbj_root: Path | None = None,
) -> tuple[int, int]:
    """Return (stale_count, checked_count) for facility citation slices vs ACTIVE national file."""
    root = root or cms_data_paths.repo_root()
    pbj_root = pbj_root or _pbjapp_root()
    deployments = pbj_root / "deployments"
    if not deployments.is_dir():
        return 0, 0

    stale = 0
    checked = 0
    for dep in sorted(deployments.glob("pbj320-*")):
        ccn = dep.name.removeprefix("pbj320-")
        cit_out = dep / f"facility_{ccn}_citations.csv"
        if not cit_out.is_file():
            continue
        checked += 1
        needs_rebuild, _reason = citation_slice_needs_rebuild(pbj_root, str(cit_out))
        if needs_rebuild:
            stale += 1
    return stale, checked


def audit_stale_citation_packages(
    *,
    root: Path | None = None,
    pbj_root: Path | None = None,
) -> dict[str, Any]:
    """Identify facility citation slices stale vs ACTIVE ``cms.health_citations``."""
    root = root or cms_data_paths.repo_root()
    pbj_root = pbj_root or _pbjapp_root()
    stale_packages: list[dict[str, str]] = []
    checked = 0
    deployments = pbj_root / "deployments"
    if deployments.is_dir():
        for dep in sorted(deployments.glob("pbj320-*")):
            ccn = dep.name.removeprefix("pbj320-")
            cit_out = dep / f"facility_{ccn}_citations.csv"
            if not cit_out.is_file():
                continue
            checked += 1
            needs_rebuild, reason = citation_slice_needs_rebuild(pbj_root, str(cit_out))
            if needs_rebuild:
                stale_packages.append(
                    {"ccn": ccn, "path": str(cit_out), "reason": reason or "stale vs ACTIVE registry"}
                )

    stale_count, checked_total = count_stale_citation_packages(root=root, pbj_root=pbj_root)
    active = get_active_release("cms.health_citations", registry_path(root)) or {}
    return {
        "active_release_id": active.get("active_release_id"),
        "stale_count": stale_count,
        "checked_count": checked_total or checked,
        "stale_packages": stale_packages,
        "is_stale": stale_count > 0,
    }


def rebuild_citation_packages(
    *,
    ccns: list[str] | tuple[str, ...] | None = None,
    root: Path | None = None,
    pbj_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rebuild stale ``facility_*_citations.csv`` slices from ACTIVE national file."""
    root = root or cms_data_paths.repo_root()
    pbj_root = pbj_root or _pbjapp_root()
    pre = audit_stale_citation_packages(root=root, pbj_root=pbj_root)
    targets = list(pre["stale_packages"])
    if ccns:
        wanted = {str(c).strip().zfill(6) for c in ccns if str(c).strip()}
        targets = [row for row in targets if row["ccn"] in wanted]
        if not targets and wanted:
            raise CitationPackagesRebuildError(
                f"no stale citation packages matched CCNs: {', '.join(sorted(wanted))}"
            )

    if dry_run:
        return {
            "dry_run": True,
            "pre_audit": pre,
            "would_rebuild": [row["ccn"] for row in targets],
            "stale_remaining": pre["stale_count"],
        }

    if not targets:
        return {
            "pre_audit": pre,
            "rebuilt": [],
            "failed": [],
            "post_audit": pre,
            "stale_remaining": 0,
        }

    _ensure_pbjapp_path()

    from active_release_client import (
        load_active_release,
        validated_source_paths,
        write_artifact_provenance,
    )
    from citation_lib import build_facility_citations_csv

    citations_release = load_active_release("cms.health_citations")
    source_paths = validated_source_paths(citations_release)
    if not source_paths:
        raise CitationPackagesRebuildError("ACTIVE cms.health_citations has no validated source path")
    source_csv = str(source_paths[0])

    rebuilt: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for row in targets:
        ccn = row["ccn"]
        out_path = row["path"]
        try:
            row_count = build_facility_citations_csv(
                ccn,
                out_path,
                source_csv=source_csv,
                root=str(pbj_root),
            )
            if row_count is None:
                failed.append({"ccn": ccn, "error": "build_facility_citations_csv returned None"})
                continue
            write_artifact_provenance(out_path, citations_release)
            rebuilt.append({"ccn": ccn, "path": out_path, "rows": row_count})
        except Exception as exc:  # noqa: BLE001 — collect per-facility failures
            failed.append({"ccn": ccn, "error": str(exc)})

    post = audit_stale_citation_packages(root=root, pbj_root=pbj_root)
    return {
        "pre_audit": pre,
        "rebuilt": rebuilt,
        "failed": failed,
        "post_audit": post,
        "stale_remaining": post["stale_count"],
    }
