"""Durable, machine-readable ACTIVE CMS release registry.

This is the data-ops/PBJapp boundary.  It deliberately has no Flask dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
REGISTRY_ENV = "PBJ_ACTIVE_RELEASE_REGISTRY"


class ActiveReleaseError(RuntimeError):
    pass


def registry_path(root: Path | None = None) -> Path:
    configured = (os.environ.get(REGISTRY_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    base = (root or Path(os.environ.get("PBJ_REPO_ROOT") or Path.cwd())).resolve()
    return base / "state" / "active_releases.json"


def empty_registry() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "updated_at": None, "datasets": {}}


def load_registry(path: Path | None = None, *, root: Path | None = None) -> dict[str, Any]:
    target = path or registry_path(root)
    if not target.is_file():
        return empty_registry()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActiveReleaseError(f"invalid active release registry: {target}: {exc}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION or not isinstance(payload.get("datasets"), dict):
        raise ActiveReleaseError(f"unsupported active release registry schema: {target}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote_release(
    source_id: str,
    release_id: str,
    source_path: str | Path,
    *,
    status: str = "ACTIVE",
    release_date: str | None = None,
    downloaded_at: str | None = None,
    validated_at: str | None = None,
    source_hash: str | None = None,
    metadata: dict[str, Any] | None = None,
    path: Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Atomically promote one validated local source release."""
    source_id, release_id = source_id.strip(), release_id.strip()
    if not source_id or not release_id:
        raise ActiveReleaseError("source_id and release_id are required")
    if status != "ACTIVE" or not validated_at:
        raise ActiveReleaseError("only validated ACTIVE releases may be promoted")
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise ActiveReleaseError(f"canonical source file missing: {source}")
    actual_hash = sha256_file(source)
    if source_hash and source_hash.lower() != actual_hash:
        raise ActiveReleaseError("source hash does not match canonical source file")
    target = path or registry_path(root)
    payload = load_registry(target)
    now = datetime.now(timezone.utc).isoformat()
    clean_metadata = dict(metadata or {})
    raw_source_set = clean_metadata.get("source_set")
    if raw_source_set is not None:
        if not isinstance(raw_source_set, list) or not raw_source_set:
            raise ActiveReleaseError("metadata.source_set must be a non-empty list")
        normalized_set: list[dict[str, Any]] = []
        roles: set[str] = set()
        for item in raw_source_set:
            if not isinstance(item, dict):
                raise ActiveReleaseError("metadata.source_set members must be objects")
            role = str(item.get("role") or "").strip()
            member_path = Path(str(item.get("source_path") or "")).expanduser().resolve()
            if not role or role in roles or not member_path.is_file():
                raise ActiveReleaseError("source_set requires unique roles and existing source_path files")
            roles.add(role)
            member_hash = sha256_file(member_path)
            claimed_hash = str(item.get("hash") or "").strip().lower()
            if claimed_hash and claimed_hash != member_hash:
                raise ActiveReleaseError(f"source_set hash mismatch for role {role}")
            normalized_set.append(
                {
                    "role": role,
                    "source_filename": member_path.name,
                    "source_uri": member_path.as_uri(),
                    "hash": member_hash,
                }
            )
        clean_metadata["source_set"] = normalized_set
    record = {
        "dataset_id": source_id,
        "active_release_id": release_id,
        "source_filename": source.name,
        "source_uri": source.as_uri(),
        "release_date": release_date,
        "downloaded_at": downloaded_at,
        "validated_at": validated_at,
        "hash": actual_hash,
        "status": "ACTIVE",
        "schema_version": SCHEMA_VERSION,
        "metadata": clean_metadata,
        "promoted_at": now,
    }
    payload["datasets"][source_id] = record
    payload["updated_at"] = now
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return record


def get_active_release(source_id: str, path: Path | None = None) -> dict[str, Any] | None:
    record = load_registry(path).get("datasets", {}).get(source_id)
    return record if isinstance(record, dict) and record.get("status") == "ACTIVE" else None
