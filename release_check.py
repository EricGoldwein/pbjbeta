"""One safe release check for every configured PBJ data source."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from active_release_registry import load_registry, registry_path
from release_control_plane import ReleaseState, load_candidates, record_candidate
from release_source_catalog import SOURCES, UpdateMechanism

ROOT = Path(__file__).resolve().parent
CHECK_STATE = ROOT / "state" / "release_checks.json"


def load_check_state(root: Path = ROOT) -> dict[str, Any]:
    path = root / "state" / "release_checks.json"
    if not path.is_file():
        return {"schema_version": 1, "checked_at": None, "datasets": []}
    return json.loads(path.read_text(encoding="utf-8"))


def derived_state(dataset_id: str, upstream: tuple[str, ...], *, root: Path = ROOT) -> dict[str, Any]:
    active = load_registry(registry_path(root)).get("datasets", {})
    target = active.get(dataset_id)
    if not target:
        return {"status": "MISSING", "new_release_available": True}
    recorded = ((target.get("metadata") or {}).get("upstream_releases") or {})
    current = {key: (active.get(key) or {}).get("active_release_id") for key in upstream}
    provenance_missing = not bool(recorded)
    if provenance_missing:
        stale = False
    else:
        stale = any(not value or recorded.get(key) != value for key, value in current.items())
    publisher_latest_release_id = None
    if upstream:
        publisher_latest_release_id = current.get(upstream[0])
    return {
        "status": "UNKNOWN" if provenance_missing else ("STALE" if stale else "CURRENT"),
        "new_release_available": stale,
        "provenance_missing": provenance_missing,
        "publisher_latest_release_id": publisher_latest_release_id,
        "upstream_active": current,
        "upstream_recorded": recorded,
    }


def check_releases(
    *,
    acquire: bool = True,
    root: Path = ROOT,
    external_handlers: dict[str, Callable[[bool], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Check all sources. Handlers may acquire/validate, but never promote."""
    handlers = external_handlers or {}
    active = load_registry(registry_path(root)).get("datasets", {})
    candidates = load_candidates(root).get("datasets", {})
    checked_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        row = source.as_dict()
        row.update({"checked_at": checked_at, "active_release": (active.get(source.dataset_id) or {}).get("active_release_id")})
        if source.mechanism == UpdateMechanism.EXTERNAL_RECURRING:
            handler = handlers.get(source.dataset_id)
            if handler is None:
                row.update({"status": "CHECK_UNAVAILABLE", "new_release_available": None, "detail": "external adapter not configured"})
            else:
                try:
                    result = handler(acquire)
                    row.update(result)
                except Exception as exc:  # fail closed; ACTIVE is untouched
                    pending = candidates.get(source.dataset_id)
                    release_id = str((pending or {}).get("release_id") or "check-failed")
                    record_candidate(source.dataset_id, release_id, ReleaseState.FAILED, validation={"status": "FAIL", "error": str(exc)}, root=root)
                    row.update({"status": "FAILED", "new_release_available": None, "detail": str(exc)})
        elif source.mechanism == UpdateMechanism.DERIVED:
            row.update(derived_state(source.dataset_id, source.upstream, root=root))
        elif source.mechanism == UpdateMechanism.MANUAL_VERSIONED:
            row.update({"status": "MANUAL / CURRENT" if row["active_release"] else "MANUAL / MISSING", "new_release_available": None})
        else:
            row.update({"status": "STATIC / CURRENT" if row["active_release"] else "STATIC / MISSING", "new_release_available": None})
        candidate = load_candidates(root).get("datasets", {}).get(source.dataset_id)
        row["candidate_state"] = (candidate or {}).get("state")
        row["pending_release"] = (candidate or {}).get("release_id") if (candidate or {}).get("state") != "ACTIVE" else None
        rows.append(row)
    payload = {"schema_version": 1, "checked_at": checked_at, "datasets": rows}
    from ownership_pairing import pairing_status
    payload["ownership_pairing"] = pairing_status(root)
    check_path = root / "state" / "release_checks.json"
    check_path.parent.mkdir(parents=True, exist_ok=True)
    check_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def production_handlers() -> dict[str, Callable[[bool], dict[str, Any]]]:
    """Adapters already proven in this repository; missing adapters stay explicit."""
    from cms_data_ops import acquire_nurse, acquire_provider_info, check_nurse_cms, check_provider_info_cms
    from cms_source_registry import CMS_ID_PBJ_NON_NURSE, CMS_ID_SNF_ALL_OWNERS, CMS_ID_SNF_ENROLLMENTS
    from generic_cms_csv import CsvFeed, run_feed
    from nonnurse_lifecycle import normalize_validate_candidate
    import cms_data_paths

    def provider(acquire: bool) -> dict[str, Any]:
        check = check_provider_info_cms()
        active_id = ((load_registry(registry_path(ROOT)).get("datasets") or {}).get("cms.provider_info") or {}).get("active_release_id")
        label = str(check["cms"]["data_vintage_label"])
        import re
        match = re.fullmatch(r"([A-Za-z]+) (20\d{2})", label)
        month = {name: index for index, name in enumerate(("", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))}.get(match.group(1)[:3].title()) if match else None
        if match and month and active_id == f"{match.group(2)}-{month:02d}":
            return {"status": "CURRENT", "new_release_available": False}
        if not check["cms_is_newer"]:
            return {"status": "CURRENT", "new_release_available": False}
        result = acquire_provider_info() if acquire else None
        state = ((load_candidates(ROOT).get("datasets") or {}).get("cms.provider_info") or {}).get("state")
        return {"status": state or "DETECTED", "new_release_available": True, "release_id": check["cms"]["data_vintage_label"], "acquisition": (result or {}).get("acquire_report")}

    def nurse(acquire: bool) -> dict[str, Any]:
        check = check_nurse_cms()
        active_id = ((load_registry(registry_path(ROOT)).get("datasets") or {}).get("cms.pbj_nurse_staffing") or {}).get("active_release_id")
        quarter_label = check["cms"]["quarter_label"]
        if active_id == quarter_label:
            return {
                "status": "CURRENT",
                "new_release_available": False,
                "release_id": quarter_label,
                "publisher_latest_release_id": quarter_label,
            }
        if not check["cms_is_newer"]:
            return {
                "status": "CURRENT",
                "new_release_available": False,
                "release_id": quarter_label,
                "publisher_latest_release_id": quarter_label,
            }
        result = acquire_nurse() if acquire else None
        state = ((load_candidates(ROOT).get("datasets") or {}).get("cms.pbj_nurse_staffing") or {}).get("state")
        return {"status": state or "DETECTED", "new_release_available": True, "release_id": check["cms"]["quarter_label"], "acquisition": (result or {}).get("acquire_report")}

    def health_citations(acquire: bool) -> dict[str, Any]:
        from health_citations_acquire import check_health_citations_cms

        check = check_health_citations_cms(root=cms_data_paths.repo_root())
        active_id = check.get("active_release_id")
        cms_release = (check.get("cms") or {}).get("release_id")
        if active_id and cms_release and str(active_id) == str(cms_release):
            return {
                "status": "CURRENT",
                "new_release_available": False,
                "release_id": cms_release,
                "publisher_latest_release_id": cms_release,
            }
        if not check.get("cms_is_newer"):
            return {
                "status": "CURRENT",
                "new_release_available": False,
                "release_id": cms_release,
                "publisher_latest_release_id": cms_release,
            }
        state = ((load_candidates(ROOT).get("datasets") or {}).get("cms.health_citations") or {}).get("state")
        return {
            "status": state or "DETECTED",
            "new_release_available": True,
            "release_id": cms_release,
            "publisher_latest_release_id": cms_release,
        }

    pbj_root = cms_data_paths.repo_root()
    generic = {
        "cms.pbj_non_nurse_staffing": CsvFeed("cms.pbj_non_nurse_staffing", CMS_ID_PBJ_NON_NURSE, r"PBJ_dailynonnursestaffing_CY\d{4}Q[1-4]\.csv$", cms_data_paths.nonnurse_raw_dir(), (("PROVNUM", "CCN"), ("WorkDate", "work_date")), False),
        "cms.snf_all_owners": CsvFeed("cms.snf_all_owners", CMS_ID_SNF_ALL_OWNERS, r"SNF.*Owners.*\.csv$", pbj_root / "ownership" / "_sources" / "cms_snf_all_owners" / "raw" / "downloaded", (("ENROLLMENT ID", "ENROLLMENT_ID"),), False),
        "cms.snf_enrollments": CsvFeed("cms.snf_enrollments", CMS_ID_SNF_ENROLLMENTS, r"SNF.*Enroll.*\.csv$", pbj_root / "ownership" / "_sources" / "cms_snf_enrollments" / "raw" / "downloaded", (("ENROLLMENT ID", "ENROLLMENT_ID"), ("CCN", "CMS Certification Number (CCN)")), False),
    }
    def nonnurse(acquire: bool) -> dict[str, Any]:
        result = run_feed(generic["cms.pbj_non_nurse_staffing"], acquire, root=ROOT)
        if acquire and result.get("status") == "ACQUIRED":
            pending = (load_candidates(ROOT).get("datasets") or {}).get("cms.pbj_non_nurse_staffing") or {}
            from urllib.parse import unquote, urlparse
            raw = unquote(urlparse(str(pending.get("source_uri") or "")).path)
            if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
                raw = raw[1:]
            normalize_validate_candidate(Path(raw), str(pending["release_id"]), control_root=ROOT)
            result["status"] = "VALIDATED"
        return result

    handlers = {
        "cms.provider_info": provider,
        "cms.pbj_nurse_staffing": nurse,
        "cms.pbj_non_nurse_staffing": nonnurse,
        "cms.health_citations": health_citations,
    }
    handlers.update({key: (lambda acquire, feed=feed: run_feed(feed, acquire, root=ROOT)) for key, feed in generic.items() if key != "cms.pbj_non_nurse_staffing"})
    return handlers


def main() -> int:
    parser = argparse.ArgumentParser(description="PBJ data-ops release control")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-releases", help="check/acquire configured releases; never promote")
    check.add_argument("--detect-only", action="store_true", help="record detections without acquisition")
    check.add_argument("--json", action="store_true")
    check.add_argument("--verbose", action="store_true", help="show derived/static datasets and detector detail")
    args = parser.parse_args()
    payload = check_releases(acquire=not args.detect_only, external_handlers=production_handlers())
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for row in payload["datasets"]:
            if not args.verbose and row["mechanism"] not in {"external recurring release", "manually versioned reference"}:
                continue
            if row.get("new_release_available") is True:
                display = f"NEW {row.get('pending_release') or row.get('release_id') or ''} -> {row.get('candidate_state') or row['status']}".strip()
            else:
                display = row["status"]
            print(f"{row['label']:<22} {display}")
            if args.verbose:
                print(f"  detector={row.get('detector') or '—'} mechanism={row['mechanism']} checked={row['checked_at']}")
    return 1 if any(row["status"] == "FAILED" for row in payload["datasets"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
