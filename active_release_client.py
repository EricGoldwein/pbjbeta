"""PBJapp-compatible client for the data-ops ACTIVE release contract."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

REGISTRY_ENV = "PBJ_ACTIVE_RELEASE_REGISTRY"
SCHEMA_VERSION = 1


class ReleaseRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActiveRelease:
    dataset_id: str
    active_release_id: str
    source_filename: str
    source_uri: str
    release_date: str | None
    downloaded_at: str | None
    validated_at: str
    hash: str
    status: str
    schema_version: int
    metadata: dict[str, Any]

    @property
    def local_path(self) -> Path:
        parsed = urlparse(self.source_uri)
        if parsed.scheme != "file":
            raise ReleaseRegistryError(f"no reader configured for {self.source_uri}")
        raw = unquote(parsed.path)
        if parsed.netloc:
            raw = f"//{parsed.netloc}{raw}"
        if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
            raw = raw[1:]
        return Path(raw)


def configured_registry_path(path: str | Path | None = None) -> Path:
    raw = str(path or os.environ.get(REGISTRY_ENV) or "").strip()
    if not raw:
        raise ReleaseRegistryError(f"{REGISTRY_ENV} is required; refusing filename fallback")
    return Path(raw).expanduser().resolve()


def load_active_release(dataset_id: str, path: str | Path | None = None) -> ActiveRelease:
    target = configured_registry_path(path)
    if not target.is_file():
        raise ReleaseRegistryError(f"active release registry missing: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseRegistryError(f"invalid active release registry: {target}: {exc}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseRegistryError(f"unsupported active release registry schema: {target}")
    raw = (payload.get("datasets") or {}).get(dataset_id)
    if not isinstance(raw, dict) or raw.get("status") != "ACTIVE":
        raise ReleaseRegistryError(f"no validated ACTIVE release for {dataset_id}")
    required = ("active_release_id", "source_filename", "source_uri", "validated_at", "hash")
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ReleaseRegistryError(f"ACTIVE release {dataset_id} missing: {', '.join(missing)}")
    return ActiveRelease(
        dataset_id=dataset_id,
        active_release_id=str(raw["active_release_id"]),
        source_filename=str(raw["source_filename"]),
        source_uri=str(raw["source_uri"]),
        release_date=raw.get("release_date"),
        downloaded_at=raw.get("downloaded_at"),
        validated_at=str(raw["validated_at"]),
        hash=str(raw["hash"]),
        status="ACTIVE",
        schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
        metadata=dict(raw.get("metadata") or {}),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_active_source(release: ActiveRelease) -> Path:
    source = release.local_path
    if not source.is_file():
        raise ReleaseRegistryError(f"ACTIVE source unavailable: {source}")
    if sha256_file(source) != release.hash:
        raise ReleaseRegistryError(f"ACTIVE source hash mismatch: {source}")
    return source


def validated_source_paths(release: ActiveRelease) -> list[Path]:
    raw_set = release.metadata.get("source_set")
    if not raw_set:
        return [validate_active_source(release)]
    if not isinstance(raw_set, list):
        raise ReleaseRegistryError(f"invalid source_set for {release.dataset_id}")
    paths: list[Path] = []
    for item in raw_set:
        if not isinstance(item, dict) or not item.get("source_uri") or not item.get("hash"):
            raise ReleaseRegistryError(f"invalid source_set member for {release.dataset_id}")
        member = ActiveRelease(
            dataset_id=release.dataset_id,
            active_release_id=release.active_release_id,
            source_filename=str(item.get("source_filename") or "source"),
            source_uri=str(item["source_uri"]),
            release_date=release.release_date,
            downloaded_at=release.downloaded_at,
            validated_at=release.validated_at,
            hash=str(item["hash"]),
            status="ACTIVE",
            schema_version=release.schema_version,
            metadata={},
        )
        paths.append(validate_active_source(member))
    return paths


def artifact_provenance_path(artifact: str | Path) -> Path:
    return Path(f"{Path(artifact)}.source.json")


def write_artifact_provenance(
    artifact: str | Path,
    release: ActiveRelease,
    *,
    sidecar_path: str | Path | None = None,
) -> Path:
    target = Path(artifact)
    if not target.is_file():
        raise ReleaseRegistryError(f"cannot record provenance for missing artifact: {target}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": release.dataset_id,
        "source_release": release.active_release_id,
        "source_hash": release.hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_hash": sha256_file(target),
    }
    sidecar = Path(sidecar_path) if sidecar_path else artifact_provenance_path(target)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def artifact_release_state(artifact: str | Path, release: ActiveRelease) -> tuple[str, str]:
    target = Path(artifact)
    if not target.is_file():
        return "BUILD", "artifact missing"
    sidecar = artifact_provenance_path(target)
    if not sidecar.is_file():
        return "BUILD", "artifact provenance missing"
    try:
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "BUILD", "artifact provenance invalid"
    if provenance.get("source_dataset") != release.dataset_id:
        return "BUILD", "artifact source dataset mismatch"
    if provenance.get("source_release") != release.active_release_id or provenance.get("source_hash") != release.hash:
        return "BUILD", "artifact source release is STALE"
    if provenance.get("artifact_hash") != sha256_file(target):
        return "BUILD", "artifact changed since provenance was recorded"
    return "REUSE", f"validated against {release.dataset_id} {release.active_release_id}"
