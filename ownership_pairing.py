"""Deterministic SNF owners/enrollment pairing gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from active_release_registry import load_registry, registry_path
from release_control_plane import load_candidates

OWNERS = "cms.snf_all_owners"
ENROLLMENTS = "cms.snf_enrollments"


def pairing_status(root: Path | None = None) -> dict[str, Any]:
    active = load_registry(registry_path(root)).get("datasets", {})
    pending = load_candidates(root).get("datasets", {})
    active_owner, active_enrollment = active.get(OWNERS) or {}, active.get(ENROLLMENTS) or {}
    owner, enrollment = pending.get(OWNERS) or {}, pending.get(ENROLLMENTS) or {}
    blockers: list[str] = []
    if bool(owner) != bool(enrollment):
        blockers.append("both owners and enrollment candidates are required")
    if owner and enrollment and owner.get("release_id") != enrollment.get("release_id"):
        blockers.append(f"candidate periods differ: owners={owner.get('release_id')} enrollment={enrollment.get('release_id')}")
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
            "aligned": bool(active_owner and active_enrollment and active_owner.get("active_release_id") == active_enrollment.get("active_release_id")),
        },
        "pending": {
            "owners_release": owner.get("release_id"), "owners_state": owner.get("state"),
            "enrollment_release": enrollment.get("release_id"), "enrollment_state": enrollment.get("state"),
        },
        "review_state": review,
        "blocking_reasons": blockers,
    }


def require_ready_pair(root: Path | None = None) -> dict[str, Any]:
    status = pairing_status(root)
    if status["review_state"] != "READY FOR REVIEW":
        raise RuntimeError("ownership pair blocked: " + "; ".join(status["blocking_reasons"] or [status["review_state"]]))
    return status
