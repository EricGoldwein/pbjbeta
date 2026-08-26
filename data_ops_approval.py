"""Release approval + audit trail for Data Ops (human gate).

BLOCKED (Zweli) cannot be approved in V0.
REQUIRES_REVIEW requires explicit acknowledgement before approval.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from data_ops_zweli import ZweliState


class ApprovalAction(str, Enum):
    ACKNOWLEDGE_REVIEW = "acknowledge_review"
    APPROVE = "approve"
    REJECT = "reject"


@dataclass
class AuditEntry:
    timestamp: str
    source_id: str
    release_id: str
    action: str
    note: str = ""
    actor: str = "data_ops_operator"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_audit_path(root: Path | None = None) -> Path:
    base = root or Path(os.environ.get("PBJ_REPO_ROOT") or Path.cwd())
    return base / "provider_info" / "_manifests" / "_data_ops_audit.jsonl"


def append_audit(entry: AuditEntry, path: Path | None = None) -> Path:
    p = path or _default_audit_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict()) + "\n")
    return p


def read_audit(path: Path | None = None, *, limit: int = 200) -> list[dict[str, Any]]:
    p = path or _default_audit_path()
    if not p.is_file():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def has_acknowledgement(
    source_id: str,
    release_id: str,
    path: Path | None = None,
) -> bool:
    for e in read_audit(path):
        if (
            e.get("source_id") == source_id
            and e.get("release_id") == release_id
            and e.get("action") == ApprovalAction.ACKNOWLEDGE_REVIEW.value
        ):
            return True
    return False


def has_approval(source_id: str, release_id: str, path: Path | None = None) -> bool:
    for e in read_audit(path):
        if (
            e.get("source_id") == source_id
            and e.get("release_id") == release_id
            and e.get("action") == ApprovalAction.APPROVE.value
        ):
            return True
    return False


class ApprovalError(Exception):
    pass


def acknowledge_requires_review(
    source_id: str,
    release_id: str,
    *,
    note: str = "",
    audit_path: Path | None = None,
) -> AuditEntry:
    entry = AuditEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        source_id=source_id,
        release_id=release_id,
        action=ApprovalAction.ACKNOWLEDGE_REVIEW.value,
        note=note,
    )
    append_audit(entry, audit_path)
    return entry


def approve_release(
    source_id: str,
    release_id: str,
    *,
    zweli_state: ZweliState | str,
    note: str = "",
    audit_path: Path | None = None,
) -> AuditEntry:
    state = ZweliState(zweli_state) if not isinstance(zweli_state, ZweliState) else zweli_state
    if state == ZweliState.BLOCKED:
        raise ApprovalError("BLOCKED releases cannot be approved in Data Ops V0")
    if state == ZweliState.REQUIRES_REVIEW and not has_acknowledgement(
        source_id, release_id, audit_path
    ):
        raise ApprovalError(
            "REQUIRES_REVIEW releases need explicit acknowledgement before approval"
        )
    if state == ZweliState.NOT_RUN:
        raise ApprovalError("Cannot approve release with Zweli NOT_RUN")
    entry = AuditEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        source_id=source_id,
        release_id=release_id,
        action=ApprovalAction.APPROVE.value,
        note=note,
    )
    append_audit(entry, audit_path)
    return entry
