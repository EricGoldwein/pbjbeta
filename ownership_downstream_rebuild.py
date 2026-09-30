"""Post-activation ownership downstream audit and rebuild (orchestrates existing PBJapp builders)."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from active_release_registry import get_active_release, load_registry, registry_path, sha256_file
from ownership_pairing import ENROLLMENTS, OWNERS, format_ownership_release_label
from release_control_plane import (
    DEPENDENCY_GRAPH,
    candidate_local_path,
    load_facility_index,
    refresh_facility_index,
)

OWNERSHIP_DOWNSTREAM_SOURCE_ID = "ownership.downstream"
BRIDGE_SUBDIR = Path("ownership") / "_derived" / "cms_snf_ownership_ccn_bridge"
PAIRING_MANIFEST_REL = (
    Path("ownership")
    / "_sources"
    / "cms_snf_enrollments"
    / "ownership_release_pairing_manifest.csv"
)
ENROLL_DOWNLOAD_REL = Path("ownership") / "_sources" / "cms_snf_enrollments" / "raw" / "downloaded"
OWNERS_DOWNLOAD_REL = Path("ownership") / "_sources" / "cms_snf_all_owners" / "raw" / "downloaded"
BUILD_LOOKUP_SCRIPT = BRIDGE_SUBDIR / "build_release_lookup.py"


class OwnershipRebuildError(RuntimeError):
    pass


def _pbjapp_root() -> Path:
    configured = (os.environ.get("PBJ_REPO_ROOT") or "").strip()
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parent.parent / "PBJapp"


def _uri_to_path(uri: str | None) -> Path | None:
    if not uri or not str(uri).startswith("file:///"):
        return None
    raw = unquote(urlparse(str(uri)).path)
    if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    return Path(raw)


def _active_pair_release(root: Path | None = None) -> tuple[str, dict[str, Any], dict[str, Any]]:
    active = load_registry(registry_path(root)).get("datasets") or {}
    owners = active.get(OWNERS) or {}
    enroll = active.get(ENROLLMENTS) or {}
    owners_id = str(owners.get("active_release_id") or "")
    enroll_id = str(enroll.get("active_release_id") or "")
    if not owners_id or owners_id != enroll_id:
        raise OwnershipRebuildError(
            f"ACTIVE ownership pair misaligned: owners={owners_id or '—'} enrollments={enroll_id or '—'}"
        )
    return owners_id, owners, enroll


def _policy_path(pbj_root: Path) -> Path:
    return pbj_root / "ownership" / "ownership_release_policy.json"


def _load_policy(pbj_root: Path) -> dict[str, Any]:
    path = _policy_path(pbj_root)
    if not path.is_file():
        raise OwnershipRebuildError(f"missing ownership_release_policy.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _bridge_lookup_path(pbj_root: Path, release_id: str) -> Path:
    return pbj_root / BRIDGE_SUBDIR / f"release_{release_id}_lookup.json"


def _facility_package_provenance_mismatches(
    facility_index: dict[str, Any],
    *,
    owners_active: dict[str, Any],
    enroll_active: dict[str, Any],
) -> list[dict[str, Any]]:
    """Read package manifests and fail closed on ownership provenance mismatches."""
    expected = {
        OWNERS: owners_active,
        ENROLLMENTS: enroll_active,
    }
    mismatches: list[dict[str, Any]] = []
    for ccn, facility in (facility_index.get("facilities") or {}).items():
        manifest_raw = str(facility.get("manifest") or "").strip()
        if not manifest_raw:
            continue
        manifest_path = Path(manifest_raw)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            for dataset_id, active in expected.items():
                mismatches.append(
                    {
                        "ccn": str(ccn),
                        "artifact": "PACKAGE_MANIFEST.json",
                        "dataset_id": dataset_id,
                        "actual_release": None,
                        "expected_release": active.get("active_release_id"),
                        "actual_hash": None,
                        "expected_hash": active.get("hash"),
                        "reason": f"package manifest unreadable: {exc}",
                    }
                )
            continue

        provenance = manifest.get("source_release_provenance") or []
        for dataset_id, active in expected.items():
            entries = [
                item
                for item in provenance
                if isinstance(item, dict) and item.get("source_dataset") == dataset_id
            ]
            if not entries:
                mismatches.append(
                    {
                        "ccn": str(ccn),
                        "artifact": "PACKAGE_MANIFEST.json",
                        "dataset_id": dataset_id,
                        "actual_release": None,
                        "expected_release": active.get("active_release_id"),
                        "actual_hash": None,
                        "expected_hash": active.get("hash"),
                        "reason": "package manifest has no provenance entry for ACTIVE source",
                    }
                )
                continue
            for item in entries:
                release_ok = item.get("source_release") == active.get("active_release_id")
                hash_ok = item.get("source_hash") == active.get("hash")
                if release_ok and hash_ok:
                    continue
                mismatches.append(
                    {
                        "ccn": str(ccn),
                        "artifact": item.get("artifact") or "(unknown artifact)",
                        "dataset_id": dataset_id,
                        "actual_release": item.get("source_release"),
                        "expected_release": active.get("active_release_id"),
                        "actual_hash": item.get("source_hash"),
                        "expected_hash": active.get("hash"),
                        "reason": "package source release/hash does not match ACTIVE",
                    }
                )
    return mismatches


def audit_ownership_downstream_stale(
    *,
    root: Path | None = None,
    pbj_root: Path | None = None,
) -> dict[str, Any]:
    """Fingerprint-based stale audit for ownership-derived capabilities."""
    pbj_root = pbj_root or _pbjapp_root()
    release_id, owners_active, enroll_active = _active_pair_release(root)
    stale_caps: list[str] = []
    reasons: list[str] = []
    evidence: dict[str, Any] = {
        "active_release_id": release_id,
        "owners_hash": owners_active.get("hash"),
        "enrollments_hash": enroll_active.get("hash"),
    }

    policy = _load_policy(pbj_root)
    policy_active = str(policy.get("active_release_date") or "")
    evidence["policy_active_release_date"] = policy_active
    if policy_active != release_id:
        stale_caps.extend(DEPENDENCY_GRAPH.get(OWNERS, ()))
        reasons.append(
            f"ownership_release_policy active {policy_active} != registry ACTIVE {release_id}"
        )
    policy_release = (policy.get("releases") or {}).get(release_id) or {}
    policy_fingerprints = {
        OWNERS: policy_release.get("ownership_source_sha256"),
        ENROLLMENTS: policy_release.get("enrollment_source_sha256"),
    }
    evidence["policy_source_hashes"] = policy_fingerprints
    for dataset_id, active_record in ((OWNERS, owners_active), (ENROLLMENTS, enroll_active)):
        recorded_hash = str(policy_fingerprints.get(dataset_id) or "")
        active_hash = str(active_record.get("hash") or "")
        if recorded_hash == active_hash and recorded_hash:
            continue
        for cap in DEPENDENCY_GRAPH.get(dataset_id, ()):
            if cap not in stale_caps:
                stale_caps.append(cap)
        reasons.append(
            f"ownership_release_policy {dataset_id} hash {recorded_hash or 'missing'} "
            f"!= registry ACTIVE {active_hash or 'missing'}"
        )

    lookup_path = _bridge_lookup_path(pbj_root, release_id)
    evidence["bridge_lookup_path"] = str(lookup_path)
    evidence["bridge_lookup_present"] = lookup_path.is_file()
    if not lookup_path.is_file():
        for cap in DEPENDENCY_GRAPH.get(ENROLLMENTS, ()):
            if cap not in stale_caps:
                stale_caps.append(cap)
        reasons.append(f"missing bridge lookup for ACTIVE {release_id}")

    facility = load_facility_index(root)
    package_mismatches = _facility_package_provenance_mismatches(
        facility,
        owners_active=owners_active,
        enroll_active=enroll_active,
    )
    evidence["facility_package_provenance_mismatches"] = package_mismatches
    for mismatch in package_mismatches:
        dataset_id = str(mismatch.get("dataset_id") or "")
        for cap in DEPENDENCY_GRAPH.get(dataset_id, (dataset_id,)):
            if cap not in stale_caps:
                stale_caps.append(cap)
        reasons.append(
            "facility {ccn} {artifact}: {dataset} provenance {actual_release}/{actual_hash} "
            "!= ACTIVE {expected_release}/{expected_hash}".format(
                ccn=mismatch.get("ccn"),
                artifact=mismatch.get("artifact"),
                dataset=dataset_id,
                actual_release=mismatch.get("actual_release") or "missing",
                actual_hash=mismatch.get("actual_hash") or "missing",
                expected_release=mismatch.get("expected_release") or "missing",
                expected_hash=mismatch.get("expected_hash") or "missing",
            )
        )
    ccn_caps: dict[str, str] = {}
    for _ccn, row in (facility.get("facilities") or {}).items():
        for cap, status in (row.get("capabilities") or {}).items():
            if cap in {"facility.snf_owners", "ownership.enrollment_ccn_bridge"}:
                ccn_caps[cap] = status
    evidence["facility_capability_status"] = ccn_caps
    for cap, status in ccn_caps.items():
        if status == "STALE" and cap not in stale_caps:
            stale_caps.append(cap)
            reasons.append(f"facility index marks {cap} STALE")

    stale_caps = sorted(set(stale_caps))
    return {
        "release_id": release_id,
        "release_label": format_ownership_release_label(release_id),
        "stale_capabilities": stale_caps,
        "blocking_reasons": reasons,
        "evidence": evidence,
        "is_stale": bool(stale_caps),
    }


def _copy_active_source(src_uri: str | None, dst: Path) -> Path:
    src = _uri_to_path(src_uri)
    if src is None or not src.is_file():
        raise OwnershipRebuildError(f"ACTIVE source file missing: {src_uri}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def _ensure_pairing_manifest_row(pbj_root: Path, release_id: str, *, owners: dict[str, Any], enroll: dict[str, Any]) -> None:
    manifest_path = pbj_root / PAIRING_MANIFEST_REL
    if not manifest_path.is_file():
        raise OwnershipRebuildError(f"missing pairing manifest: {manifest_path}")
    owners_name = Path(str(owners.get("source_filename") or "")).name
    enroll_name = Path(str(enroll.get("source_filename") or "")).name
    if not owners_name or not enroll_name:
        raise OwnershipRebuildError("ACTIVE ownership records missing source_filename")

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if (row.get("ownership_release_date") or "").strip() == release_id:
                continue
            rows.append({k: (row.get(k) or "") for k in fieldnames})

    enroll_rel = str(ENROLL_DOWNLOAD_REL / enroll_name).replace("\\", "/")
    new_row = {k: "" for k in fieldnames}
    new_row.update(
        {
            "ownership_source_filename": owners_name,
            "ownership_release_date": release_id,
            "enrollment_source_filename": enroll_name,
            "enrollment_release_date": release_id,
            "enrollment_raw_relative_path": enroll_rel,
            "pairing_status": "exact_release_date_match",
            "pairing_rationale": "data-ops governed ACTIVE pair",
        }
    )
    rows.append(new_row)
    rows.sort(key=lambda r: r.get("ownership_release_date") or "")

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _update_ownership_release_policy(
    pbj_root: Path,
    release_id: str,
    *,
    owners: dict[str, Any],
    enroll: dict[str, Any],
) -> None:
    policy = _load_policy(pbj_root)
    owners_name = Path(str(owners.get("source_filename") or "")).name
    enroll_name = Path(str(enroll.get("source_filename") or "")).name
    owners_hash = str(owners.get("hash") or "")
    enroll_hash = str(enroll.get("hash") or "")
    if not owners_hash or not enroll_hash:
        raise OwnershipRebuildError("ACTIVE ownership records missing hash")

    releases = dict(policy.get("releases") or {})
    releases[release_id] = {
        "ownership_source_filename": owners_name,
        "ownership_source_sha256": owners_hash,
        "bridge_lookup_filename": f"release_{release_id}_lookup.json",
        "bridge_pairing_status": "exact_release_date_match",
        "status": "active",
        "enrollment_source_filename": enroll_name,
        "enrollment_source_sha256": enroll_hash,
        "enrollment_release_date": release_id,
        "allow_enrollment_date_mismatch": False,
        "build_timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "notes": "Promoted via Data Ops ownership downstream rebuild",
    }
    for key, entry in list(releases.items()):
        if key == release_id:
            continue
        if isinstance(entry, dict) and entry.get("status") == "active":
            entry = dict(entry)
            entry["status"] = "historical_supported"
            releases[key] = entry

    policy["active_release_date"] = release_id
    policy["releases"] = releases
    handoff = policy.get("active_release_handoff")
    if isinstance(handoff, dict):
        handoff = dict(handoff)
        handoff["inbound_sources"] = [
            {
                "source_id": "cms_download",
                "kind": "repo_relative",
                "repo_role": "pbjapp_repo",
                "relative_path": f"ownership/{owners_name}",
            }
        ]
        policy["active_release_handoff"] = handoff

    path = _policy_path(pbj_root)
    path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_build_release_lookup(pbj_root: Path, release_id: str) -> dict[str, Any]:
    script = pbj_root / BUILD_LOOKUP_SCRIPT
    if not script.is_file():
        raise OwnershipRebuildError(f"missing bridge builder: {script}")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--ownership-release",
            release_id,
            "--update-index",
        ],
        cwd=str(pbj_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise OwnershipRebuildError(
            f"build_release_lookup failed: {proc.stderr.strip() or proc.stdout.strip() or proc.returncode}"
        )
    for line in reversed(proc.stdout.splitlines()):
        text = line.strip()
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue
    text = proc.stdout.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"stdout_tail": proc.stdout.strip()[-500:]}


def rebuild_ownership_downstream(
    *,
    root: Path | None = None,
    pbj_root: Path | None = None,
) -> dict[str, Any]:
    """Rebuild ownership bridge + policy from governed ACTIVE pair (local only)."""
    pbj_root = pbj_root or _pbjapp_root()
    release_id, owners_active, enroll_active = _active_pair_release(root)

    owners_dst = pbj_root / OWNERS_DOWNLOAD_REL / Path(str(owners_active.get("source_filename") or "")).name
    enroll_dst = pbj_root / ENROLL_DOWNLOAD_REL / Path(str(enroll_active.get("source_filename") or "")).name
    staged_owners = pbj_root / "ownership" / owners_dst.name

    _copy_active_source(str(owners_active.get("source_uri")), owners_dst)
    _copy_active_source(str(enroll_active.get("source_uri")), enroll_dst)
    _copy_active_source(str(owners_active.get("source_uri")), staged_owners)

    _ensure_pairing_manifest_row(pbj_root, release_id, owners=owners_active, enroll=enroll_active)
    lookup_summary = _run_build_release_lookup(pbj_root, release_id)
    _update_ownership_release_policy(pbj_root, release_id, owners=owners_active, enroll=enroll_active)

    if str(pbj_root) not in sys.path:
        sys.path.insert(0, str(pbj_root))
    from ownership.ownership_active_release_handoff import ensure_active_ownership_source_staged

    handoff = ensure_active_ownership_source_staged(pbj_root)

    refresh_facility_index(pbj_root, root=root, ccns=("335581",))

    post = audit_ownership_downstream_stale(root=root, pbj_root=pbj_root)
    return {
        "release_id": release_id,
        "release_label": format_ownership_release_label(release_id),
        "lookup_summary": lookup_summary,
        "handoff_action": handoff.get("action"),
        "post_audit": post,
        "stale_capabilities_remaining": post.get("stale_capabilities") or [],
    }
