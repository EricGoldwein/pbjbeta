"""Artifact access modes for Data Ops (where data lives vs how it is processed).

V0 implements local_filesystem and cms_http probes only.
remote_store / api / mcp are reserved hooks — not implemented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from cms_source_registry import AccessMode


class RuntimeAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    NOT_AVAILABLE_IN_THIS_RUNTIME = "NOT_AVAILABLE_IN_THIS_RUNTIME"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to an artifact without assuming it lives in the git checkout."""

    label: str
    access_mode: AccessMode
    path: Optional[str] = None
    url: Optional[str] = None
    release_id: Optional[str] = None
    availability: RuntimeAvailability = RuntimeAvailability.UNKNOWN
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["access_mode"] = self.access_mode.value
        d["availability"] = self.availability.value
        return d


def local_file_ref(
    label: str,
    path: Path | None,
    *,
    release_id: str | None = None,
    expected_elsewhere: bool = False,
) -> ArtifactRef:
    if path is not None and path.is_file() and path.stat().st_size > 0:
        return ArtifactRef(
            label=label,
            access_mode=AccessMode.LOCAL_FILESYSTEM,
            path=str(path.resolve()),
            release_id=release_id,
            availability=RuntimeAvailability.AVAILABLE,
            detail="Readable on local filesystem for this runtime",
        )
    if expected_elsewhere:
        return ArtifactRef(
            label=label,
            access_mode=AccessMode.LOCAL_FILESYSTEM,
            path=str(path) if path else None,
            release_id=release_id,
            availability=RuntimeAvailability.NOT_AVAILABLE_IN_THIS_RUNTIME,
            detail=(
                "NOT AVAILABLE IN THIS RUNTIME — may exist on another machine "
                "(e.g. PBJ_DATA_ROOT / operator workstation) or only at CMS"
            ),
        )
    return ArtifactRef(
        label=label,
        access_mode=AccessMode.UNAVAILABLE,
        path=str(path) if path else None,
        release_id=release_id,
        availability=RuntimeAvailability.NOT_AVAILABLE_IN_THIS_RUNTIME,
        detail="No accessible artifact in this runtime",
    )


def cms_http_ref(label: str, url: str | None, *, release_id: str | None = None) -> ArtifactRef:
    if not url:
        return ArtifactRef(
            label=label,
            access_mode=AccessMode.CMS_HTTP,
            availability=RuntimeAvailability.UNKNOWN,
            detail="No CMS URL configured",
        )
    return ArtifactRef(
        label=label,
        access_mode=AccessMode.CMS_HTTP,
        url=url,
        release_id=release_id,
        availability=RuntimeAvailability.AVAILABLE,
        detail="CMS HTTP distribution / landing (fetch may still be required)",
    )
