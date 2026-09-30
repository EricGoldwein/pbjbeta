"""Deterministic SNF owners/enrollment pairing gate."""

from __future__ import annotations

import csv
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from active_release_registry import load_registry, registry_path, sha256_file
from cms_source_registry import CMS_ID_SNF_ALL_OWNERS, CMS_ID_SNF_ENROLLMENTS
from generic_cms_csv import CsvFeed, compare_publisher_artifact_identity, detect, validate_local_csv
from release_control_plane import (
    ReleaseState,
    candidate_local_path,
    load_candidates,
    promote_active_pair,
    promote_candidate,
    record_validated_pair,
    what_would_change,
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
    cms_product_path=(
        "/provider-characteristics/hospitals-and-other-facilities/"
        "skilled-nursing-facility-all-owners"
    ),
    cms_product_name="Skilled Nursing Facility All Owners",
)
_ENROLL_FEED = CsvFeed(
    ENROLLMENTS,
    CMS_ID_SNF_ENROLLMENTS,
    r"SNF.*Enroll.*\.csv$",
    Path("."),
    (("ENROLLMENT ID", "ENROLLMENT_ID"), ("CCN", "CMS Certification Number (CCN)")),
    False,
    cms_product_path=(
        "/provider-characteristics/hospitals-and-other-facilities/"
        "skilled-nursing-facility-enrollments"
    ),
    cms_product_name="Skilled Nursing Facility Enrollments",
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


def _active_artifact_health(record: dict[str, Any] | None) -> tuple[bool, str | None]:
    record = record or {}
    source = candidate_local_path(record.get("source_uri"))
    if source is None or not source.is_file():
        return False, "unchanged ACTIVE partner artifact is missing"
    expected_hash = str(record.get("hash") or "")
    if not expected_hash:
        return False, "unchanged ACTIVE partner has no governed hash"
    if sha256_file(source).lower() != expected_hash.lower():
        return False, "unchanged ACTIVE partner hash does not match its governed artifact"
    return True, None


def _is_same_release_revision(candidate: dict[str, Any], active: dict[str, Any]) -> bool:
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    active_metadata = active.get("metadata") if isinstance(active.get("metadata"), dict) else {}
    changes, _matches = compare_publisher_artifact_identity(active_metadata, metadata)
    return bool(
        metadata.get("publisher_revision_changed")
        and str(candidate.get("release_id") or "") == str(active.get("active_release_id") or "")
        and changes
    )


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
    roles = {OWNERS: "CANDIDATE" if owner else None, ENROLLMENTS: "CANDIDATE" if enrollment else None}
    mode = "TWO_CANDIDATE_RELEASE" if owner and enrollment else None
    candidate = owner or enrollment
    candidate_id = OWNERS if owner else (ENROLLMENTS if enrollment else None)
    partner_id = ENROLLMENTS if candidate_id == OWNERS else (OWNERS if candidate_id == ENROLLMENTS else None)
    if bool(owner) != bool(enrollment):
        candidate_active = active.get(candidate_id) or {}
        partner_active = active.get(partner_id) or {}
        if not candidate or not _is_same_release_revision(candidate, candidate_active):
            blockers.append("a one-sided candidate is allowed only for a proven same-release publisher revision")
        else:
            mode = "ONE_SIDED_REVISION"
            roles[partner_id] = "UNCHANGED_ACTIVE"
            if str(partner_active.get("active_release_id") or "") != str(candidate.get("release_id") or ""):
                blockers.append(
                    f"unchanged ACTIVE partner release {partner_active.get('active_release_id') or '—'} "
                    f"does not match revision {candidate.get('release_id') or '—'}"
                )
            healthy, reason = _active_artifact_health(partner_active)
            if not healthy and reason:
                blockers.append(reason)
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
    owner_release = (owner or {}).get("release_id")
    enrollment_release = (enrollment or {}).get("release_id")
    owner_state = (owner or {}).get("state")
    enrollment_state = (enrollment or {}).get("state")
    if mode == "ONE_SIDED_REVISION":
        if roles[OWNERS] == "UNCHANGED_ACTIVE":
            owner_release = active_owner.get("active_release_id")
            owner_state = "ACTIVE"
        if roles[ENROLLMENTS] == "UNCHANGED_ACTIVE":
            enrollment_release = active_enrollment.get("active_release_id")
            enrollment_state = "ACTIVE"
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
            "owners_release": owner_release,
            "owners_state": owner_state,
            "owners_role": roles[OWNERS],
            "enrollment_release": enrollment_release,
            "enrollment_state": enrollment_state,
            "enrollment_role": roles[ENROLLMENTS],
        },
        "mode": mode,
        "candidate_source_ids": [source_id for source_id, role in roles.items() if role == "CANDIDATE"],
        "active_partner_source_ids": [source_id for source_id, role in roles.items() if role == "UNCHANGED_ACTIVE"],
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
        "dataset_version_id": found.get("dataset_version_id"),
        "dataset_version_label": found.get("dataset_version_label"),
        "dataset_version_modified": found.get("dataset_version_modified"),
        "type": row.get("type") or row.get("media_bundle"),
    }


def pair_lifecycle_action(status: dict[str, Any]) -> dict[str, Any]:
    """Canonical next pair action from governed candidate state (not UI guesswork)."""
    pending = status.get("pending") or {}
    owners_state = str(pending.get("owners_state") or "").upper()
    enrollment_state = str(pending.get("enrollment_state") or "").upper()
    candidate_states = [
        owners_state if source_id == OWNERS else enrollment_state
        for source_id in status.get("candidate_source_ids") or []
    ]
    all_candidates_validated = bool(candidate_states) and all(state == "VALIDATED" for state in candidate_states)
    either_acquired = "ACQUIRED" in {owners_state, enrollment_state}
    ready = status.get("review_state") == "READY FOR REVIEW"
    revision = status.get("mode") == "ONE_SIDED_REVISION"
    panel_args = {"source_id": PAIR_SOURCE_ID}

    if either_acquired and not all_candidates_validated:
        return {
            "label": "Validate revision" if revision else "Validate pair",
            "detail": (
                "Validate the revised candidate against the unchanged healthy ACTIVE partner."
                if revision
                else "Both members are acquired and have not reached governed VALIDATED."
            ),
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
    if all_candidates_validated and ready:
        candidate_names = [
            "SNF All Owners" if source_id == OWNERS else "SNF Enrollments"
            for source_id in status.get("candidate_source_ids") or []
        ]
        return {
            "label": f"Make {candidate_names[0]} ACTIVE" if revision and len(candidate_names) == 1 else "Make pair ACTIVE",
            "detail": (
                "The revision is validated with the unchanged ACTIVE partner. Only the revised artifact will be promoted."
                if revision
                else "Pair is validated and aligned. Promote both members to ACTIVE together."
            ),
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
    pair_role: str | None,
) -> dict[str, Any]:
    pending_record = pending_record or {}
    active_record = active_record or {}
    display_record = active_record if pair_role == "UNCHANGED_ACTIVE" else pending_record
    validation = display_record.get("validation") if isinstance(display_record.get("validation"), dict) else {}
    candidate_id = (
        active_record.get("active_release_id") if pair_role == "UNCHANGED_ACTIVE" else pending_record.get("release_id")
    )
    publisher_id = (publisher or {}).get("release_id")
    matches = bool(publisher_id and candidate_id and str(publisher_id) == str(candidate_id))
    source_path = candidate_local_path(display_record.get("source_uri"))
    summary = _csv_summary(source_path) if source_path else {}
    active_path = candidate_local_path(active_record.get("source_uri"))
    active_summary = _csv_summary(active_path) if active_path else {}
    display_metadata = display_record.get("metadata") if isinstance(display_record.get("metadata"), dict) else {}
    return {
        "source_id": source_id,
        "human_name": "SNF All Owners (PECOS)" if source_id == OWNERS else "SNF Enrollments",
        "cms_dataset_id": (publisher or {}).get("cms_dataset_id")
        or (_FEEDS[source_id].cms_dataset_id if source_id in _FEEDS else None),
        "active_release_id": active_record.get("active_release_id"),
        "active_release_label": format_ownership_release_label(active_record.get("active_release_id")),
        "active_hash": active_record.get("hash"),
        "active_filename": active_path.name if active_path else active_record.get("source_filename"),
        "active_path": str(active_path) if active_path else active_record.get("source_uri"),
        "active_row_count": active_summary.get("row_count"),
        "candidate_release_id": candidate_id,
        "candidate_release_label": format_ownership_release_label(candidate_id),
        "candidate_state": "UNCHANGED ACTIVE" if pair_role == "UNCHANGED_ACTIVE" else pending_record.get("state"),
        "pair_role": pair_role,
        "validation_status": "PASS" if pair_role == "UNCHANGED_ACTIVE" else validation.get("status"),
        "validated_at": validation.get("validated_at"),
        "candidate_hash": display_record.get("hash") or validation.get("hash"),
        "candidate_filename": source_path.name if source_path else None,
        "candidate_path": str(source_path) if source_path else pending_record.get("source_uri"),
        "candidate_row_count": validation.get("row_count") or summary.get("row_count"),
        "preview": summary.get("preview") or [],
        "preview_columns": summary.get("preview_columns") or [],
        "change_kind": (pending_record.get("metadata") or {}).get("change_kind") if pair_role == "CANDIDATE" else "UNCHANGED",
        "publisher_release_id": publisher_id,
        "publisher_release_label": format_ownership_release_label(publisher_id),
        "publisher_filename": (publisher or {}).get("filename") or display_metadata.get("cms_publisher_filename"),
        "publisher_url": (publisher or {}).get("url") or display_metadata.get("cms_publisher_url"),
        "publisher_last_updated": (publisher or {}).get("last_updated") or display_metadata.get("cms_dataset_version_modified"),
        "publisher_file_uuid": (publisher or {}).get("file_uuid") or display_metadata.get("cms_file_uuid"),
        "publisher_version_id": (publisher or {}).get("dataset_version_id") or display_metadata.get("cms_dataset_version_id"),
        "publisher_version_label": (publisher or {}).get("dataset_version_label") or display_metadata.get("cms_dataset_version_label"),
        "publisher_version_modified": (publisher or {}).get("dataset_version_modified") or display_metadata.get("cms_dataset_version_modified"),
        "publisher_matches_candidate": matches if publisher else None,
        "publisher_error": (publisher or {}).get("error"),
    }


@lru_cache(maxsize=32)
def _csv_summary_cached(path_text: str, size: int, mtime_ns: int) -> dict[str, Any]:
    path = Path(path_text)
    preview: list[dict[str, str]] = []
    row_count = 0
    preferred = {
        "ENROLLMENT ID",
        "CCN",
        "ASSOCIATE ID",
        "ORGANIZATION NAME",
        "TYPE - OWNER",
        "ROLE TEXT - OWNER",
        "ORGANIZATION NAME - OWNER",
        "FIRST NAME - OWNER",
        "LAST NAME - OWNER",
        "PERCENTAGE OWNERSHIP",
    }
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = [column for column in (reader.fieldnames or []) if column in preferred][:8]
        if not columns:
            columns = list(reader.fieldnames or [])[:8]
        for row in reader:
            if not any(str(value or "").strip() for value in row.values()):
                continue
            row_count += 1
            if len(preview) < 5:
                preview.append({column: str(row.get(column) or "") for column in columns})
    return {"row_count": row_count, "preview_columns": columns, "preview": preview}


def _csv_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    stat = path.stat()
    return _csv_summary_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


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
        pair_role=pending.get("owners_role"),
    )
    enroll_view = _member_view(
        ENROLLMENTS,
        pending_record=candidates.get(ENROLLMENTS),
        active_record=active.get(ENROLLMENTS),
        publisher=publishers.get(ENROLLMENTS),
        pair_role=pending.get("enrollment_role"),
    )
    owners_state = str(pending.get("owners_state") or "").upper()
    enrollment_state = str(pending.get("enrollment_state") or "").upper()
    candidate_states = [
        owners_state if source_id == OWNERS else enrollment_state
        for source_id in status.get("candidate_source_ids") or []
    ]
    all_candidates_validated = bool(candidate_states) and all(state == "VALIDATED" for state in candidate_states)
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
    if all_candidates_validated and status.get("review_state") == "READY FOR REVIEW":
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
        "mode": status.get("mode"),
        "change_kind": "REVISED" if status.get("mode") == "ONE_SIDED_REVISION" else "NEWER",
        "pair_alignment": "ALIGNED" if candidate_aligned else "MISALIGNED",
        "publisher_alignment": "ALIGNED" if publisher_aligned else ("UNKNOWN" if not check_cms else "MISALIGNED"),
        "review_state": status.get("review_state"),
        "blocking_reasons": status.get("blocking_reasons") or [],
        "owners": owners_view,
        "enrollments": enroll_view,
        "next_action": action,
        "downstream_impact": sorted(
            {
                capability
                for source_id in status.get("candidate_source_ids") or []
                for capability in what_would_change(source_id).get("would_mark_stale") or []
            }
        ),
        "cms_queried": check_cms,
    }


def validate_ownership_pair(
    *,
    root: Path | None = None,
    fetch_json: FetchJson | None = None,
) -> dict[str, Any]:
    """Validate candidate members against candidates or an unchanged ACTIVE partner."""
    status = pairing_status(root)
    if status.get("review_state") != "READY FOR REVIEW":
        raise PairValidationError(
            "ownership pair blocked: "
            + "; ".join(status.get("blocking_reasons") or [str(status.get("review_state") or "unknown")])
        )
    pending_meta = status.get("pending") or {}
    candidates = load_candidates(root).get("datasets") or {}
    active = load_registry(registry_path(root)).get("datasets") or {}
    candidate_ids = list(status.get("candidate_source_ids") or [])
    partner_ids = list(status.get("active_partner_source_ids") or [])
    if not candidate_ids:
        raise PairValidationError("no ownership candidate is available for validation")
    release_id = str((candidates.get(candidate_ids[0]) or {}).get("release_id") or "")
    if not release_id:
        raise PairValidationError("pair candidates are missing a release_id")

    now = datetime.now(timezone.utc).isoformat()
    validated: dict[str, dict[str, Any]] = {}
    publishers: dict[str, dict[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for source_id in (OWNERS, ENROLLMENTS):
        is_candidate = source_id in candidate_ids
        record = (candidates if is_candidate else active).get(source_id)
        if not isinstance(record, dict):
            raise PairValidationError(f"{source_id} pair member is missing")
        records[source_id] = record
        if is_candidate:
            state = str(record.get("state") or "").upper()
            if state not in {"ACQUIRED", "VALIDATED"}:
                raise PairValidationError(f"{source_id} state is {state}, not ACQUIRED/VALIDATED")
            member_release = str(record.get("release_id") or "")
        else:
            state = str(record.get("status") or "").upper()
            if state != "ACTIVE":
                raise PairValidationError(f"{source_id} unchanged partner is not ACTIVE")
            member_release = str(record.get("active_release_id") or "")
        if member_release != release_id:
            raise PairValidationError(
                f"pair periods differ: {source_id}={member_release or '—'} expected={release_id}"
            )
        publisher = detect_cms_publisher_release(source_id, fetch_json=fetch_json)
        publishers[source_id] = publisher
        if str(publisher.get("release_id") or "") != release_id:
            raise PairValidationError(
                f"{source_id} candidate {release_id} does not match CMS publisher "
                f"{publisher.get('release_id')} ({publisher.get('filename')})"
            )
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        identity_changes, identity_matches = compare_publisher_artifact_identity(metadata, publisher)
        if identity_changes:
            role = "candidate" if is_candidate else "unchanged ACTIVE partner"
            raise PairValidationError(
                f"{source_id} {role} does not match current CMS artifact ({', '.join(identity_changes)})"
            )
        if not identity_matches:
            raise PairValidationError(f"{source_id} CMS artifact identity cannot be proven")
        source_path = candidate_local_path(record.get("source_uri"))
        if source_path is None or not source_path.is_file():
            raise PairValidationError(f"{source_id} candidate artifact is missing")
        validation = validate_local_csv(source_path, _FEEDS[source_id].required_column_groups)
        validation["validated_at"] = now
        validations[source_id] = validation
        if not is_candidate:
            if validation.get("hash") != record.get("hash"):
                raise PairValidationError(f"{source_id} unchanged ACTIVE partner hash changed on disk")
            continue
        candidate_metadata = dict(metadata)
        candidate_metadata.update(
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
            "metadata": candidate_metadata,
        }

    linkage = _validate_enrollment_linkage(
        candidate_local_path(records[OWNERS].get("source_uri")),
        candidate_local_path(records[ENROLLMENTS].get("source_uri")),
    )
    partner_provenance = {
        source_id: {
            "release_id": active[source_id].get("active_release_id"),
            "hash": active[source_id].get("hash"),
            "source_uri": active[source_id].get("source_uri"),
            "cms_publisher_url": ((active[source_id].get("metadata") or {}).get("cms_publisher_url")),
        }
        for source_id in partner_ids
    }
    for source_id, record in validated.items():
        metadata = dict(record.get("metadata") or {})
        metadata["pair_validation"] = {
            "status": "PASS",
            "mode": status.get("mode"),
            "release_id": release_id,
            "validated_at": now,
            "linkage": linkage,
            "active_partners": partner_provenance,
        }
        record["metadata"] = metadata

    record_validated_pair(validated, root=root)
    return {
        "release_id": release_id,
        "state": ReleaseState.VALIDATED.value,
        "mode": status.get("mode"),
        "owners": validated.get(OWNERS) or active.get(OWNERS),
        "enrollments": validated.get(ENROLLMENTS) or active.get(ENROLLMENTS),
        "validations": validations,
        "linkage": linkage,
        "publishers": publishers,
        "pending": pending_meta,
    }


def _validate_enrollment_linkage(owners_path: Path | None, enrollments_path: Path | None) -> dict[str, Any]:
    if owners_path is None or enrollments_path is None:
        raise PairValidationError("ownership pair artifacts are missing")

    def _ids(path: Path) -> tuple[set[str], int]:
        values: set[str] = set()
        rows = 0
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            field = next(
                (name for name in (reader.fieldnames or []) if name.strip().upper().replace("_", " ") == "ENROLLMENT ID"),
                None,
            )
            if not field:
                raise PairValidationError(f"{path.name} has no ENROLLMENT ID column")
            for row in reader:
                rows += 1
                value = str(row.get(field) or "").strip()
                if value:
                    values.add(value)
        return values, rows

    owner_ids, owner_rows = _ids(owners_path)
    enrollment_ids, enrollment_rows = _ids(enrollments_path)
    missing = owner_ids - enrollment_ids
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        raise PairValidationError(
            f"{len(missing)} Owners enrollment IDs are absent from Enrollments (examples: {examples})"
        )
    return {
        "status": "PASS",
        "owners_row_count": owner_rows,
        "enrollments_row_count": enrollment_rows,
        "owners_enrollment_ids": len(owner_ids),
        "enrollments_enrollment_ids": len(enrollment_ids),
        "matched_owners_enrollment_ids": len(owner_ids & enrollment_ids),
        "missing_owners_enrollment_ids": 0,
    }


def promote_ownership_pair(*, root: Path | None = None) -> dict[str, Any]:
    """Promote a normal pair atomically, or only the member revised in place."""
    status = require_ready_pair(root)
    pending = status.get("pending") or {}
    owners_state = str(pending.get("owners_state") or "").upper()
    enrollment_state = str(pending.get("enrollment_state") or "").upper()
    candidate_ids = list(status.get("candidate_source_ids") or [])
    if status.get("mode") == "ONE_SIDED_REVISION":
        if len(candidate_ids) != 1:
            raise PairPromotionUnavailable("same-release revision requires exactly one candidate")
        candidate_id = candidate_ids[0]
        candidate = (load_candidates(root).get("datasets") or {}).get(candidate_id) or {}
        if str(candidate.get("state") or "").upper() != "VALIDATED":
            raise PairPromotionUnavailable("revised member must be VALIDATED before activation")
        pair_validation = ((candidate.get("metadata") or {}).get("pair_validation") or {})
        partners = pair_validation.get("active_partners") if isinstance(pair_validation, dict) else {}
        active = load_registry(registry_path(root)).get("datasets") or {}
        for partner_id in status.get("active_partner_source_ids") or []:
            expected = (partners or {}).get(partner_id) or {}
            current = active.get(partner_id) or {}
            if (
                expected.get("release_id") != current.get("active_release_id")
                or expected.get("hash") != current.get("hash")
                or expected.get("source_uri") != current.get("source_uri")
            ):
                raise PairPromotionUnavailable(
                    f"{partner_id} ACTIVE partner changed after pair validation; validate again"
                )
        promoted = promote_candidate(candidate_id, root=root)
        result = {
            "release_id": promoted.get("active_release_id"),
            "promoted_at": promoted.get("promoted_at"),
            "datasets": {candidate_id: promoted},
            "unchanged_active_partners": list(status.get("active_partner_source_ids") or []),
        }
    else:
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
