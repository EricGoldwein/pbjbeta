"""Generic PBJ320 Stage/Publish contract (baseline+overlay model)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pbj320_publication import stage_artifact_cache_path

PUBLICATION_CONTRACT_VERSION = 2
STAGE_STATUS_STAGED = "STAGED"


@dataclass(frozen=True)
class StagePublishSpec:
    """Source-specific publish requirements plugged into the generic contract."""

    source_id: str
    required_destination_roles: tuple[str, ...]
    shared_derived_roles: frozenset[str] = field(default_factory=frozenset)
    min_schema_version: int = PUBLICATION_CONTRACT_VERSION
    require_artifact_cache: bool = True


def evaluate_stage_publish_eligibility(
    manifest: dict[str, Any] | None,
    spec: StagePublishSpec,
    *,
    root: Path | None = None,
    release_id: str | None = None,
) -> dict[str, Any]:
    """Return publishable/stale/reasons for a Stage manifest against the current contract."""
    reasons: list[str] = []
    if not manifest:
        return {
            "publishable": False,
            "stale": False,
            "contract_version": PUBLICATION_CONTRACT_VERSION,
            "reasons": ["no stage manifest"],
            "detail": "no stage manifest",
        }

    status = str(manifest.get("status") or "")
    is_staged = status == STAGE_STATUS_STAGED
    if not is_staged:
        return {
            "publishable": False,
            "stale": False,
            "contract_version": PUBLICATION_CONTRACT_VERSION,
            "reasons": [f"status is {status or '—'}"],
            "detail": f"status is {status or '—'}",
        }

    manifest_schema = int(manifest.get("schema_version") or 1)
    if manifest_schema < spec.min_schema_version:
        reasons.append(
            f"schema v{manifest_schema} predates contract v{spec.min_schema_version}"
        )

    contract_version = manifest.get("publication_contract_version")
    if contract_version is not None and int(contract_version) < PUBLICATION_CONTRACT_VERSION:
        reasons.append("publication_contract_version out of date")

    if not str(manifest.get("publication_base_sha") or "").strip():
        reasons.append("missing publication_base_sha")

    gates = manifest.get("validation_gates") or []
    if not gates:
        reasons.append("validation gates missing")
    elif not all(bool(g.get("passed")) for g in gates):
        reasons.append("validation gates not all PASS")

    artifacts_by_role = {
        str(row.get("destination_id") or ""): row for row in (manifest.get("artifacts") or [])
    }
    for role in spec.required_destination_roles:
        row = artifacts_by_role.get(role)
        if not row:
            reasons.append(f"missing destination artifact: {role}")
            continue
        if not str(row.get("proposed_sha256") or ""):
            reasons.append(f"missing proposed_sha256: {role}")
        if role in spec.shared_derived_roles:
            if str(row.get("publication_class") or "") != "shared_derived":
                reasons.append(f"{role} not marked shared_derived")
            inputs = row.get("inputs") or []
            if not inputs:
                reasons.append(f"missing input fingerprints: {role}")
            else:
                for inp in inputs:
                    mode = str(inp.get("mode") or "")
                    if mode not in {"GOVERNED_OVERLAY", "PUBLISHED_BASELINE"}:
                        continue
                    if mode == "PUBLISHED_BASELINE" and inp.get("present_at_base") is False:
                        continue
                    if not str(inp.get("sha256") or ""):
                        inp_label = str(inp.get("role") or inp.get("path") or "input")
                        reasons.append(f"incomplete fingerprint ({inp_label}) on {role}")

    rel_release_id = release_id or str(manifest.get("active_release_id") or "")
    if spec.require_artifact_cache and rel_release_id:
        cache = stage_artifact_cache_path(spec.source_id, rel_release_id, root=root)
        if not cache.is_dir():
            reasons.append("stage artifact cache missing")
        else:
            for role in spec.required_destination_roles:
                row = artifacts_by_role.get(role)
                if not row:
                    continue
                rel = str(row.get("path") or "").replace("\\", "/")
                if rel and not (cache / rel.replace("/", os.sep)).is_file():
                    reasons.append(f"artifact cache missing: {rel}")

    publishable = len(reasons) == 0
    return {
        "publishable": publishable,
        "stale": is_staged and not publishable,
        "contract_version": PUBLICATION_CONTRACT_VERSION,
        "reasons": reasons,
        "detail": None if publishable else "; ".join(reasons),
    }


def merge_destination_layers_for_display(
    stage_manifest: dict[str, Any] | None,
    publication_record: dict[str, Any] | None,
) -> dict[str, str]:
    """Publication record layers supersede Stage-only committed/pushed/deployed/verified."""
    layers: dict[str, str] = dict((stage_manifest or {}).get("destination_layers") or {})
    pub_layers = (publication_record or {}).get("destination_layers") or {}
    for key in ("committed", "pushed", "deployed", "production_verified"):
        if key in pub_layers and pub_layers[key] not in {None, ""}:
            layers[key] = str(pub_layers[key])
    if publication_record and publication_record.get("push_succeeded"):
        layers.setdefault("committed", "YES")
        layers.setdefault("pushed", "YES")
    if publication_record and publication_record.get("production_verified"):
        layers["production_verified"] = "YES"
    return layers


def publication_state_summary(
    display_layers: dict[str, str] | None,
    publication_record: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Truthful operator copy from merged Stage + publication + verification layers."""
    layers = display_layers or {}
    committed = str(layers.get("committed") or "NO").upper()
    pushed = str(layers.get("pushed") or "NO").upper()
    deployed = str(layers.get("deployed") or "UNKNOWN").upper()
    verified = str(layers.get("production_verified") or "NO").upper()
    dest = str(layers.get("pbj320_destination") or "STAGED")

    if verified == "YES":
        return {
            "headline": "Production verified.",
            "detail": (
                "Live artifacts match the staged publication. "
                f"deployed={deployed or 'YES'} · production_verified=YES."
            ),
            "trail": "committed · pushed · deployed · verified",
            "callout_class": "ok",
            "pbj320_destination": dest,
        }
    if pushed == "YES" or (publication_record or {}).get("push_succeeded"):
        return {
            "headline": "Published to PBJ320.",
            "detail": "Commit pushed to production branch. Run verification to confirm live artifacts.",
            "trail": "committed · pushed · deploy pending verification",
            "callout_class": "ok",
            "pbj320_destination": dest,
        }
    if committed == "YES":
        return {
            "headline": "Committed locally.",
            "detail": "Publication commit recorded. Not yet pushed or verified on production.",
            "trail": "committed · no push · no deploy",
            "callout_class": "attention",
            "pbj320_destination": dest,
        }
    return {
        "headline": "Staged candidate.",
        "detail": (
            f"Artifact cache ready (pbj320_destination={dest}). "
            "Not committed, not pushed, not deployed."
        ),
        "trail": "no commit · no push · no deploy",
        "callout_class": "ok",
        "pbj320_destination": dest,
    }
