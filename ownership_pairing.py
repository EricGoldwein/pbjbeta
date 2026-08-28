"""Deterministic SNF owners/enrollment pairing gate."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from active_release_registry import load_registry, registry_path
from cms_source_registry import CMS_ID_SNF_ALL_OWNERS, CMS_ID_SNF_ENROLLMENTS
from generic_cms_csv import CsvFeed, detect, validate_local_csv
from release_control_plane import (
    ReleaseState,
    candidate_local_path,
    load_candidates,
    promote_active_pair,
    record_validated_pair,
)

OWNERS = "cms.snf_all_owners"
ENROLLMENTS = "cms.snf_enrollments"
PAIR_SOURCE_ID = "cms.snf_ownership_pair"

FetchJson = Callable[[str], Any]

_OWNERS_FEED = CsvFeed(
    OWNERS,
    CMS_ID_SNF_ALL_OWNERS,
    r"SNF.*Owners.*\.csv$",
    Path("."),
    (("ENROLLMENT ID", "ENROLLMENT_ID"),),
    False,
)
_ENROLL_FEED = CsvFeed(
    ENROLLMENTS,
    CMS_ID_SNF_ENROLLMENTS,
    r"SNF.*Enroll.*\.csv$",
    Path("."),
    (("ENROLLMENT ID", "ENROLLMENT_ID"), ("CCN", "CMS Certification Number (CCN)")),
    False,
)
_FEEDS = {OWNERS: _OWNERS_FEED, ENROLLMENTS: _ENROLL_FEED}


class PairValidationError(RuntimeError):
    """Pair cannot be advanced to governed VALIDATED."""


class PairPromotionUnavailable(RuntimeError):
    """Atomic ACTIVE pair promotion is not implemented — refuse mutation."""


def format_ownership_release_label(release_id: str | None) -> str:
    if not release_id:
        return "—"
    parts = str(release_id).split("-")
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
        month = int(parts[1])
        if 1 <= month <= 12:
            abbr = (
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            )[month - 1]
            return f"{abbr} {int(parts[2])}"
    return str(release_id)


def pairing_status(root: Path | None = None) -> dict[str, Any]:
    active = load_registry(registry_path(root)).get("datasets", {})
    pending = load_candidates(root).get("datasets", {})
    active_owner, active_enrollment = active.get(OWNERS) or {}, active.get(ENROLLMENTS) or {}
    owner_raw, enrollment_raw = pending.get(OWNERS) or {}, pending.get(ENROLLMENTS) or {}

    def _pair_pending(record: dict[str, Any] | None) -> dict[str, Any] | None:
        if not record:
            return None
        if str(record.get("state") or "").upper() in {ReleaseState.ACQUIRED.value, ReleaseState.VALIDATED.value}:
            return record
        return None

    owner, enrollment = _pair_pending(owner_raw), _pair_pending(enrollment_raw)
    blockers: list[str] = []
    if bool(owner) != bool(enrollment):
        blockers.append("both owners and enrollment candidates are required")
    if owner and enrollment and owner.get("release_id") != enrollment.get("release_id"):
        blockers.append(
            f"candidate periods differ: owners={owner.get('release_id')} enrollment={enrollment.get('release_id')}"
        )
    for label, record in (("owners", owner), ("enrollment", enrollment)):
        if record and record.get("state") not in {"ACQUIRED", "VALIDATED"}:
            blockers.append(f"{label} candidate state is {record.get('state')}, not ACQUIRED/VALIDATED")
        if record and (record.get("validation") or {}).get("status") != "PASS":
            blockers.append(f"{label} schema validation has not passed")
    review = "NO PENDING PAIR" if not owner and not enrollment else ("BLOCKED" if blockers else "READY FOR REVIEW")
    return {
        "active": {
            "owners_release": active_owner.get("active_release_id"),
            "enrollment_release": active_enrollment.get("active_release_id"),
            "aligned": bool(
                active_owner
                and active_enrollment
                and active_owner.get("active_release_id") == active_enrollment.get("active_release_id")
            ),
        },
        "pending": {
            "owners_release": (owner or {}).get("release_id"),
            "owners_state": (owner or {}).get("state"),
            "enrollment_release": (enrollment or {}).get("release_id"),
            "enrollment_state": (enrollment or {}).get("state"),
        },
        "review_state": review,
        "blocking_reasons": blockers,
    }


def require_ready_pair(root: Path | None = None) -> dict[str, Any]:
    status = pairing_status(root)
    if status["review_state"] != "READY FOR REVIEW":
        raise RuntimeError("ownership pair blocked: " + "; ".join(status["blocking_reasons"] or [status["review_state"]]))
    return status


def detect_cms_publisher_release(
    source_id: str,
    *,
    fetch_json: FetchJson | None = None,
) -> dict[str, Any]:
    """Official CMS data-api resources identity for one ownership dataset."""
    feed = _FEEDS.get(source_id)
    if feed is None:
        raise PairValidationError(f"no CMS data-api feed for {source_id}")
    found = detect(feed, fetch_json=fetch_json)
    row = found.get("row") if isinstance(found.get("row"), dict) else {}
    return {
        "source_id": source_id,
        "cms_dataset_id": feed.cms_dataset_id,
        "release_id": found.get("release_id"),
        "filename": found.get("filename"),
        "url": found.get("url"),
        "last_updated": row.get("last_updated") or row.get("modified") or row.get("updated"),
        "file_uuid": row.get("file_uuid"),
        "type": row.get("type") or row.get("media_bundle"),
    }


def pair_lifecycle_action(status: dict[str, Any]) -> dict[str, Any]:
    """Canonical next pair action from governed candidate state (not UI guesswork)."""
    pending = status.get("pending") or {}
    owners_state = str(pending.get("owners_state") or "").upper()
    enrollment_state = str(pending.get("enrollment_state") or "").upper()
    both_validated = owners_state == "VALIDATED" and enrollment_state == "VALIDATED"
    either_acquired = "ACQUIRED" in {owners_state, enrollment_state}
    ready = status.get("review_state") == "READY FOR REVIEW"
    release_id = pending.get("owners_release") or pending.get("enrollment_release")
    panel_args = {"source_id": PAIR_SOURCE_ID}

    if either_acquired and not both_validated:
        return {
            "label": "Validate pair",
            "detail": "Both members are acquired and have not reached governed VALIDATED.",
            "kind": "validate_pair",
            "endpoint": "source_detail_panel",
            "endpoint_args": panel_args,
            "opens_panel": True,
            "wired": True,
            "read_only": True,
            "page_endpoint": "source_detail",
            "page_endpoint_args": panel_args,
            "submit_endpoint": "action_ownership_pair_validate",
            "submit_wired": True,
            "submit_method": "post",
            "promotion_available": False,
        }
    if both_validated and ready:
        return {
            "label": "Activate pair",
            "detail": "Pair is validated and aligned. Promote both members to ACTIVE together.",
            "kind": "activate_pair",
            "endpoint": "source_detail_panel",
            "endpoint_args": panel_args,
            "opens_panel": False,
            "wired": True,
            "read_only": False,
            "submit_endpoint": "action_ownership_pair_activate",
            "submit_wired": True,
            "submit_method": "post",
            "promotion_available": True,
        }
    if blockers := status.get("blocking_reasons"):
        detail = "; ".join(blockers)
        label = "Resolve pair"
    else:
        detail = "Pair alignment required"
        label = "Resolve pair"
    return {
        "label": label,
        "detail": detail,
        "kind": "resolve_pair",
        "endpoint": "source_detail_panel",
        "endpoint_args": panel_args,
        "opens_panel": True,
        "wired": True,
        "read_only": True,
        "submit_wired": False,
        "promotion_available": False,
    }


def _member_view(
    source_id: str,
    *,
    pending_record: dict[str, Any] | None,
    active_record: dict[str, Any] | None,
    publisher: dict[str, Any] | None,
) -> dict[str, Any]:
    pending_record = pending_record or {}
    active_record = active_record or {}
    validation = pending_record.get("validation") if isinstance(pending_record.get("validation"), dict) else {}
    candidate_id = pending_record.get("release_id")
    publisher_id = (publisher or {}).get("release_id")
    matches = bool(publisher_id and candidate_id and str(publisher_id) == str(candidate_id))
    source_path = candidate_local_path(pending_record.get("source_uri"))
    return {
        "source_id": source_id,
        "human_name": "SNF All Owners" if source_id == OWNERS else "SNF Enrollments",
        "cms_dataset_id": (publisher or {}).get("cms_dataset_id")
        or (_FEEDS[source_id].cms_dataset_id if source_id in _FEEDS else None),
        "active_release_id": active_record.get("active_release_id"),
        "active_release_label": format_ownership_release_label(active_record.get("active_release_id")),
        "candidate_release_id": candidate_id,
        "candidate_release_label": format_ownership_release_label(candidate_id),
        "candidate_state": pending_record.get("state"),
        "validation_status": validation.get("status"),
        "validated_at": validation.get("validated_at"),
        "candidate_hash": pending_record.get("hash") or validation.get("hash"),
        "candidate_filename": source_path.name if source_path else None,
        "candidate_path": str(source_path) if source_path else pending_record.get("source_uri"),
        "publisher_release_id": publisher_id,
        "publisher_release_label": format_ownership_release_label(publisher_id),
        "publisher_filename": (publisher or {}).get("filename"),
        "publisher_url": (publisher or {}).get("url"),
        "publisher_last_updated": (publisher or {}).get("last_updated"),
        "publisher_matches_candidate": matches if publisher else None,
        "publisher_error": (publisher or {}).get("error"),
    }


def build_pair_operator_context(
    *,
    root: Path | None = None,
    fetch_json: FetchJson | None = None,
    check_cms: bool = True,
) -> dict[str, Any] | None:
    """Paired review surface: ACTIVE, candidates, CMS publisher identity, one next action."""
    status = pairing_status(root)
    pending = status.get("pending") or {}
    if not pending.get("owners_release") and not pending.get("enrollment_release"):
        return None

    candidates = load_candidates(root).get("datasets") or {}
    active = load_registry(registry_path(root)).get("datasets") or {}
    publishers: dict[str, dict[str, Any]] = {}
    if check_cms:
        for source_id in (OWNERS, ENROLLMENTS):
            try:
                publishers[source_id] = detect_cms_publisher_release(source_id, fetch_json=fetch_json)
            except Exception as exc:  # noqa: BLE001 — fail closed in the member row, not via filename
                publishers[source_id] = {"source_id": source_id, "error": str(exc)}

    owners_view = _member_view(
        OWNERS,
        pending_record=candidates.get(OWNERS),
        active_record=active.get(OWNERS),
        publisher=publishers.get(OWNERS),
    )
    enroll_view = _member_view(
        ENROLLMENTS,
        pending_record=candidates.get(ENROLLMENTS),
        active_record=active.get(ENROLLMENTS),
        publisher=publishers.get(ENROLLMENTS),
    )
    owners_state = str(pending.get("owners_state") or "").upper()
    enrollment_state = str(pending.get("enrollment_state") or "").upper()
    both_validated = owners_state == "VALIDATED" and enrollment_state == "VALIDATED"
    candidate_aligned = bool(
        pending.get("owners_release")
        and pending.get("owners_release") == pending.get("enrollment_release")
    )
    publisher_aligned = bool(
        owners_view.get("publisher_release_id")
        and owners_view.get("publisher_release_id") == enroll_view.get("publisher_release_id")
    )
    action = pair_lifecycle_action(status)
    release_id = pending.get("owners_release") or pending.get("enrollment_release")
    if both_validated and status.get("review_state") == "READY FOR REVIEW":
        status_label = "Ready to activate"
        status_tone = "current"
    else:
        status_label = "NEEDS ATTENTION"
        status_tone = "attention"

    return {
        "source_id": PAIR_SOURCE_ID,
        "human_name": "SNF Owners / Enrollments",
        "zweli_applicable": False,
        "release_id": release_id,
        "release_label": format_ownership_release_label(release_id),
        "status_label": status_label,
        "status_tone": status_tone,
        "active": status.get("active") or {},
        "active_label": format_ownership_release_label((status.get("active") or {}).get("owners_release")),
        "pair_alignment": "ALIGNED" if candidate_aligned else "MISALIGNED",
        "publisher_alignment": "ALIGNED" if publisher_aligned else ("UNKNOWN" if not check_cms else "MISALIGNED"),
        "review_state": status.get("review_state"),
        "blocking_reasons": status.get("blocking_reasons") or [],
        "owners": owners_view,
        "enrollments": enroll_view,
        "next_action": action,
        "cms_queried": check_cms,
    }


def validate_ownership_pair(
    *,
    root: Path | None = None,
    fetch_json: FetchJson | None = None,
) -> dict[str, Any]:
    """Advance both ACQUIRED pair members to VALIDATED. Does not promote ACTIVE."""
    status = pairing_status(root)
    pending_meta = status.get("pending") or {}
    candidates = load_candidates(root).get("datasets") or {}
    owner = candidates.get(OWNERS)
    enrollment = candidates.get(ENROLLMENTS)
    if not isinstance(owner, dict) or not isinstance(enrollment, dict):
        raise PairValidationError("both owners and enrollment candidates are required")
    if owner.get("release_id") != enrollment.get("release_id"):
        raise PairValidationError(
            f"candidate periods differ: owners={owner.get('release_id')} enrollment={enrollment.get('release_id')}"
        )
    release_id = str(owner.get("release_id") or "")
    if not release_id:
        raise PairValidationError("pair candidates are missing a release_id")

    now = datetime.now(timezone.utc).isoformat()
    validated: dict[str, dict[str, Any]] = {}
    publishers: dict[str, dict[str, Any]] = {}
    for source_id, record in ((OWNERS, owner), (ENROLLMENTS, enrollment)):
        state = str(record.get("state") or "").upper()
        if state not in {"ACQUIRED", "VALIDATED"}:
            raise PairValidationError(f"{source_id} state is {state}, not ACQUIRED/VALIDATED")
        publisher = detect_cms_publisher_release(source_id, fetch_json=fetch_json)
        publishers[source_id] = publisher
        if str(publisher.get("release_id") or "") != release_id:
            raise PairValidationError(
                f"{source_id} candidate {release_id} does not match CMS publisher "
                f"{publisher.get('release_id')} ({publisher.get('filename')})"
            )
        source_path = candidate_local_path(record.get("source_uri"))
        if source_path is None or not source_path.is_file():
            raise PairValidationError(f"{source_id} candidate artifact is missing")
        validation = validate_local_csv(source_path, _FEEDS[source_id].required_column_groups)
        validation["validated_at"] = now
        metadata = dict(record.get("metadata") or {})
        metadata.update(
            {
                "cms_publisher_release_id": publisher.get("release_id"),
                "cms_publisher_filename": publisher.get("filename"),
                "cms_publisher_url": publisher.get("url"),
                "pair_validated_at": now,
            }
        )
        validated[source_id] = {
            **record,
            "dataset_id": source_id,
            "release_id": release_id,
            "state": ReleaseState.VALIDATED.value,
            "source_uri": source_path.as_uri(),
            "hash": validation.get("hash") or record.get("hash"),
            "validation": validation,
            "metadata": metadata,
        }

    record_validated_pair(validated, root=root)
    return {
        "release_id": release_id,
        "state": ReleaseState.VALIDATED.value,
        "owners": validated[OWNERS],
        "enrollments": validated[ENROLLMENTS],
        "publishers": publishers,
        "pending": pending_meta,
    }


def promote_ownership_pair(*, root: Path | None = None) -> dict[str, Any]:
    """Promote both VALIDATED pair members to ACTIVE in one governed transaction."""
    status = require_ready_pair(root)
    pending = status.get("pending") or {}
    owners_state = str(pending.get("owners_state") or "").upper()
    enrollment_state = str(pending.get("enrollment_state") or "").upper()
    if owners_state != "VALIDATED" or enrollment_state != "VALIDATED":
        raise PairPromotionUnavailable("both pair members must be VALIDATED before activation")
    result = promote_active_pair((OWNERS, ENROLLMENTS), root=root)
    return {
        **result,
        "pair_review_state": pairing_status(root).get("review_state"),
        "downstream_capabilities": list(
            {
                "facility.snf_owners",
                "ownership.enrollment_ccn_bridge",
            }
        ),
        "next_operator_action": "Rebuild downstream",
    }
