"""Source-specific Release Review policy — lifecycle-aware review vs activation decisions.

Reuses control-plane candidates, ownership pairing, and provenance helpers.
No parallel workflow state machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cms_data_paths
from cms_source_registry import get_source
from data_ops_approval import has_acknowledgement
from data_ops_zweli import ZweliState
from ownership_pairing import ENROLLMENTS, OWNERS, PAIR_SOURCE_ID, pair_lifecycle_action, pairing_status
from cms_theme_publication import publication_availability_for_source
from provenance_freshness import build_source_provenance_freshness

# ACQUIRED candidates validate on Sources — not activation review.
_ACQUIRED_REVIEW_EXCLUDED = frozenset(
    {
        "cms.health_citations",
        "cms.provider_info",
        "cms.pbj_nurse_staffing",
        OWNERS,
        ENROLLMENTS,
    }
)


def zweli_applies(source_id: str) -> bool:
    record = get_source(source_id)
    return bool(record and record.zweli_quality_profile)


def release_review_query(source_id: str, release_id: str | None = None) -> dict[str, str]:
    """Query params for focused Release Review deep links."""
    args: dict[str, str] = {"source_id": source_id}
    if release_id:
        args["release_id"] = release_id
    return args


def filter_release_review_focus(
    items: list[dict[str, Any]],
    *,
    source_id: str | None,
    release_id: str | None = None,
) -> list[dict[str, Any]]:
    if not source_id:
        return items
    if source_id == PAIR_SOURCE_ID:
        return [item for item in items if item.get("source_id") == PAIR_SOURCE_ID]
    matched = [
        item
        for item in items
        if item.get("source_id") == source_id
        and (not release_id or str(item.get("release_id") or "") == release_id)
    ]
    return matched


def _promote_permitted(candidate: dict[str, Any]) -> bool:
    return str(candidate.get("state") or "").upper() == "VALIDATED"


def _format_release_month_label(release_id: str | None) -> str | None:
    import re

    if not release_id:
        return None
    match = re.fullmatch(r"(\d{4})-(\d{2})", str(release_id).strip())
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    if month < 1 or month > 12:
        return None
    abbr = (
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    )[month - 1]
    return f"{abbr} {year}"


def _release_label(release_id: str | None) -> str:
    if not release_id:
        return "—"
    label = _format_release_month_label(release_id)
    if label:
        return label
    if str(release_id).count("-") == 2:
        parts = str(release_id).split("-")
        month = int(parts[1])
        abbr = (
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        )[month - 1]
        return f"{abbr} {int(parts[2])}"
    return str(release_id)


def _uri_to_local_path(source_uri: str | None) -> Path | None:
    if not source_uri:
        return None
    raw = str(source_uri).strip()
    if raw.startswith("file:///"):
        return Path(raw[8:])
    if raw.startswith("file://"):
        return Path(raw[7:])
    return Path(raw)


def load_governed_candidate(
    source_id: str,
    release_id: str,
    *,
    root: Path | None = None,
) -> dict[str, Any] | None:
    from release_control_plane import load_candidates

    root = root or cms_data_paths.repo_root()
    candidate = (load_candidates(root).get("datasets") or {}).get(source_id)
    if not isinstance(candidate, dict):
        return None
    if str(candidate.get("release_id") or "") != release_id:
        return None
    return candidate


def structural_status_from_candidate(candidate: dict[str, Any] | None) -> str:
    """Canonical structural gate from release_candidates.json — not runtime probe."""
    if not candidate:
        return "NOT_RUN"
    state = str(candidate.get("state") or "").upper()
    validation = candidate.get("validation") if isinstance(candidate.get("validation"), dict) else {}
    validation_status = str(validation.get("status") or "").upper()
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    meta_structural = str(metadata.get("structural_status") or "").upper()

    if validation_status == "PASS" or meta_structural == "PASS":
        return "PASS"
    if validation_status in {"FAIL", "ERROR"} or state in {"STRUCTURAL_FAIL", "ERROR"}:
        return "FAIL"
    if validation_status == "UNKNOWN" or meta_structural == "UNKNOWN":
        return "UNKNOWN"
    if state == "VALIDATED":
        return "UNKNOWN"
    return "NOT_RUN"


def build_candidate_review_provenance(
    source_id: str,
    release_id: str,
    *,
    root: Path | None = None,
    theme_publication: Any | None = None,
) -> dict[str, Any]:
    """Review-surface provenance for the candidate under review (not ACTIVE)."""
    from active_release_registry import get_active_release, registry_path
    from release_control_plane import load_candidates

    root = root or cms_data_paths.repo_root()
    active = get_active_release(source_id, registry_path(root)) or {}
    candidate = load_governed_candidate(source_id, release_id, root=root) or {}

    theme_fields = publication_availability_for_source(
        source_id,
        active_release_id=active.get("active_release_id"),
        publication=theme_publication,
    ) or {}

    candidate_path = _uri_to_local_path(candidate.get("source_uri"))
    active_path = _uri_to_local_path(active.get("source_uri"))

    return {
        "source_id": source_id,
        "release_id": release_id,
        "candidate_artifact_path": str(candidate_path) if candidate_path else candidate.get("source_uri"),
        "candidate_sha256": candidate.get("hash"),
        "candidate_validated_at": (candidate.get("validation") or {}).get("validated_at"),
        "active_release_id": active.get("active_release_id"),
        "active_artifact_path": str(active_path) if active_path else active.get("source_uri"),
        "active_sha256": active.get("hash"),
        "cms_publication_date": theme_fields.get("cms_publication_date"),
        "cms_publication_id": theme_fields.get("cms_publication_id"),
    }


def _evidence_summary(
    source_id: str,
    *,
    release_id: str | None,
    active_release_id: str | None,
    validation_status: str | None,
    provenance: dict[str, Any] | None,
    review_provenance: dict[str, Any] | None = None,
) -> list[str]:
    lines: list[str] = []
    if validation_status:
        lines.append(f"Validation {validation_status}")
    if active_release_id:
        lines.append(f"Active now: {_release_label(active_release_id)}")
    review = review_provenance or {}
    pub_date = review.get("cms_publication_date") or (provenance or {}).get("cms_publication_date")
    if pub_date:
        lines.append(f"CMS {pub_date} publication")
    candidate_path = review.get("candidate_artifact_path")
    if candidate_path:
        lines.append(Path(str(candidate_path)).name)
    elif provenance and provenance.get("sha256"):
        lines.append("Provenance/hash on file")
    if source_id == "cms.pbj_nurse_staffing" and release_id:
        lines.append("PBJ nurse staffing")
    return lines


def _provider_info_approvable(
    source_id: str,
    release_id: str,
    *,
    zweli_status: str,
    candidate: dict[str, Any],
) -> bool:
    if source_id != "cms.provider_info" or not _promote_permitted(candidate):
        return False
    if zweli_status in {ZweliState.PASS.value, ZweliState.NOT_RUN.value}:
        return True
    if zweli_status == ZweliState.REQUIRES_REVIEW.value and has_acknowledgement(source_id, release_id):
        return True
    return False


def evaluate_governed_candidate_review(
    candidate: dict[str, Any],
    *,
    snap: dict[str, Any] | None,
    active_release_id: str | None,
    root: Path | None = None,
    theme_publication: Any | None = None,
) -> dict[str, Any] | None:
    """Return a Release Review card dict, or None when review is not the next human step."""
    source_id = str(candidate.get("source_id") or "")
    release_id = str(candidate.get("release_id") or "")
    state = str(candidate.get("state") or "").upper()
    if not source_id or not release_id:
        return None

    if source_id in {OWNERS, ENROLLMENTS}:
        return None

    if state == "ACQUIRED":
        if source_id in _ACQUIRED_REVIEW_EXCLUDED:
            return None
        return None

    if state not in {"VALIDATED", "STRUCTURAL_FAIL", "ERROR", "ZWELI_REQUIRES_REVIEW", "ZWELI_BLOCKED"}:
        return None

    record = get_source(source_id)
    human_name = (record.human_name if record else None) or (snap or {}).get("human_name") or source_id
    release_label = _release_label(release_id)
    validation_status = candidate.get("validation_status")
    if validation_status is None and isinstance(candidate.get("validation"), dict):
        validation_status = candidate["validation"].get("status")

    zweli_status = candidate.get("zweli_status")
    if zweli_status is None and snap is not None:
        zweli_status = snap.get("zweli_status")
    zweli_applicable = zweli_applies(source_id)
    if not zweli_applicable:
        zweli_status = None

    provenance = build_source_provenance_freshness(
        source_id,
        root=root,
        theme_publication=theme_publication,
    )
    review_provenance = build_candidate_review_provenance(
        source_id,
        release_id,
        root=root,
        theme_publication=theme_publication,
    )
    evidence = _evidence_summary(
        source_id,
        release_id=release_id,
        active_release_id=active_release_id,
        validation_status=validation_status,
        provenance=provenance,
        review_provenance=review_provenance,
    )

    human_state = "Needs review"
    primary_label = f"Review {release_label}"
    approvable = False
    primary_kind = "review"

    if state == "VALIDATED":
        if source_id == "cms.health_citations":
            human_state = "Ready to activate"
            primary_label = f"Activate {release_label}"
            approvable = True
            primary_kind = "activate"
        elif source_id == "cms.provider_info":
            approvable = _provider_info_approvable(
                source_id,
                release_id,
                zweli_status=zweli_status or ZweliState.NOT_RUN.value,
                candidate=candidate,
            )
            human_state = "Ready to activate" if approvable else "Complete quality gates"
            primary_label = f"Activate {release_label}" if approvable else "Review quality gates"
            primary_kind = "activate" if approvable else "review"
        elif source_id == "cms.pbj_nurse_staffing":
            human_state = "Ready to activate"
            primary_label = f"Activate {release_label}"
            approvable = True
            primary_kind = "activate"
        else:
            human_state = "Ready to activate"
            primary_label = f"Activate {release_label}"
            approvable = _promote_permitted(candidate)
            primary_kind = "activate" if approvable else "review"
    elif state in {"STRUCTURAL_FAIL", "ERROR"}:
        human_state = "Structural validation failed"
        primary_label = "Inspect validation"
        primary_kind = "inspect"
    elif zweli_applicable and zweli_status == ZweliState.REQUIRES_REVIEW.value:
        human_state = "Quality review required"
        primary_label = "Acknowledge findings"
        primary_kind = "acknowledge"

    if state == "VALIDATED" and source_id == "cms.provider_info" and active_release_id and not candidate:
        return None

    return {
        "source_id": source_id,
        "human_name": human_name,
        "release_id": release_id,
        "release_label": release_label,
        "human_state": human_state,
        "primary_action_label": primary_label,
        "primary_action_kind": primary_kind,
        "approvable": approvable,
        "governed": True,
        "pending_state": state,
        "validation_status": validation_status,
        "zweli_applicable": zweli_applicable,
        "zweli_status": zweli_status if zweli_applicable else None,
        "evidence_lines": evidence,
        "provenance": provenance,
        "review_provenance": review_provenance,
        "structural_status": structural_status_from_candidate(
            {**candidate, "validation_status": validation_status}
        ),
        "detail": " · ".join(evidence) if evidence else human_state,
        "acknowledged": has_acknowledgement(source_id, release_id),
        "zweli_report": (snap or {}).get("zweli_report") if snap else None,
    }


def build_ownership_pair_review_item(
    *,
    root: Path | None = None,
    control: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    pair = pairing_status(root)
    pending = pair.get("pending") or {}
    owners_release = pending.get("owners_release")
    enrollment_release = pending.get("enrollment_release")
    if not owners_release and not enrollment_release:
        return None

    release_id = owners_release or enrollment_release
    release_label = _release_label(release_id)
    review_state = str(pair.get("review_state") or "")
    owners_state = str(pending.get("owners_state") or "").upper()
    enrollment_state = str(pending.get("enrollment_state") or "").upper()
    action = pair_lifecycle_action(pair)
    both_validated = owners_state == "VALIDATED" and enrollment_state == "VALIDATED"
    ready = review_state == "READY FOR REVIEW" and both_validated

    if action.get("kind") == "validate_pair":
        human_state = "Pair acquired — validate together"
    elif action.get("kind") == "activate_pair":
        human_state = "Ready to activate pair"
    else:
        human_state = "Pair alignment required"

    blockers = pair.get("blocking_reasons") or []
    evidence = [f"Owners {owners_state or '—'}", f"Enrollments {enrollment_state or '—'}"]
    if blockers:
        evidence.extend(blockers[:2])

    return {
        "source_id": PAIR_SOURCE_ID,
        "human_name": "SNF Ownership Pair",
        "release_id": release_id,
        "release_label": release_label,
        "human_state": human_state,
        "primary_action_label": action.get("label"),
        "primary_action_kind": action.get("kind") or "review_pair",
        "approvable": False,
        "opens_panel": True,
        "governed": True,
        "pending_state": owners_state or enrollment_state,
        "validation_status": "PASS" if ready else None,
        "zweli_applicable": False,
        "zweli_status": None,
        "evidence_lines": evidence,
        "provenance": None,
        "detail": " · ".join(evidence),
        "acknowledged": False,
        "zweli_report": None,
        "pair_members": [OWNERS, ENROLLMENTS],
        "pair_review_state": review_state,
        "next_action": action,
    }


def assert_promotion_eligible(
    source_id: str,
    release_id: str,
    *,
    root: Path | None = None,
    control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed before authoritative promotion — UI state is not sufficient."""
    from data_ops_approval import ApprovalError
    from release_control_plane import control_panel_payload

    root = root or cms_data_paths.repo_root()
    if control is None:
        control = control_panel_payload(root)
    pending_by = {
        row["dataset_id"]: row.get("pending")
        for row in control.get("datasets") or []
        if row.get("dataset_id")
    }
    pending = pending_by.get(source_id) or {}
    if str(pending.get("release_id") or "") != release_id:
        raise ApprovalError(f"No pending candidate {source_id} {release_id} for activation")
    candidate = {
        "source_id": source_id,
        "release_id": release_id,
        "state": pending.get("state"),
        "validation": pending.get("validation"),
        "validation_status": (pending.get("validation") or {}).get("status"),
        "metadata": pending.get("metadata"),
        "zweli_status": pending.get("zweli_status"),
    }
    structural = structural_status_from_candidate(candidate)
    if structural != "PASS":
        raise ApprovalError(
            f"Structural validation must pass before approval ({structural})"
        )
    active_row = next(
        (row for row in control.get("datasets") or [] if row.get("dataset_id") == source_id),
        None,
    )
    active_release_id = ((active_row or {}).get("active") or {}).get("active_release_id")
    item = evaluate_governed_candidate_review(
        candidate,
        snap=None,
        active_release_id=active_release_id,
        root=root,
    )
    if item is None or not item.get("approvable"):
        raise ApprovalError(
            f"Promotion not eligible for {source_id} {release_id} at current lifecycle state"
        )
    return item


def post_activation_operator_target(
    source_id: str,
    *,
    release_id: str,
    root: Path | None = None,
    control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Next operator surface after successful activation (redirect + flash copy)."""
    from cms_data_ops import build_source_operator_workflow, minimal_record_for_dataset
    from release_control_plane import control_panel_payload, stale_derived_consumers

    root = root or cms_data_paths.repo_root()
    if control is None:
        control = control_panel_payload(root)
    control_row = next(
        (row for row in control.get("datasets") or [] if row.get("dataset_id") == source_id),
        None,
    )
    record = get_source(source_id)
    record_dict = record.to_dict() if record else minimal_record_for_dataset(source_id)
    workflow = build_source_operator_workflow(
        source_id,
        record=record_dict,
        snapshot=None,
        control_row=control_row,
        root=root,
    )
    release_label = _release_label(release_id)
    active_label = _release_label(((control_row or {}).get("active") or {}).get("active_release_id"))
    stale = stale_derived_consumers(root).get(source_id) or []
    next_action = workflow.get("next_action") or {}
    flash = f"{record_dict.get('human_name') or source_id} · {release_label} ACTIVE"
    redirect_endpoint = "sources"
    redirect_args: dict[str, str] = {}

    if stale:
        flash = f"{flash} — downstream artifacts stale ({len(stale)})"
        redirect_endpoint = "source_detail"
        redirect_args = {"source_id": source_id}
    elif next_action.get("endpoint") == "source_detail":
        redirect_endpoint = "source_detail"
        redirect_args = dict(next_action.get("endpoint_args") or {"source_id": source_id})
    elif next_action.get("endpoint") == "release_review" and next_action.get("endpoint_args"):
        redirect_endpoint = "release_review"
        redirect_args = dict(next_action.get("endpoint_args"))

    return {
        "flash": flash,
        "redirect_endpoint": redirect_endpoint,
        "redirect_args": redirect_args,
        "next_action": next_action,
        "active_label": active_label,
        "stale_capabilities": stale,
    }
