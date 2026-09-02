"""Generic production verification helpers (Norm diff, live surfaces, persistence)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

NORM_DIFF_COLUMNS = (
    "provider_name",
    "overall_rating",
    "health_inspection_rating",
    "sff_status",
)

SEARCH_INDEX_DIFF_KEYS = ("n", "h")


def normalize_ccn(raw: str) -> str:
    ccn = str(raw or "").strip().upper()
    if ccn.isdigit() and len(ccn) < 6:
        ccn = ccn.zfill(6)
    return ccn


def normalize_cell(raw: object) -> str:
    return str(raw or "").strip()


def normalize_rating(raw: object) -> str:
    s = normalize_cell(raw)
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def prior_release_id(release_id: str) -> str | None:
    parts = (release_id or "").strip().split("-", 1)
    if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
        return None
    year, month = int(parts[0]), int(parts[1])
    if month == 1:
        return f"{year - 1}-12"
    return f"{year}-{month - 1:02d}"


def norm_rel_path_for_release(release_id: str) -> str:
    parts = (release_id or "").strip().split("-", 1)
    if len(parts) == 2 and len(parts[0]) == 4 and len(parts[1]) == 2:
        return f"provider_info/ProviderInfoNorm_{parts[0]}_{parts[1]}.csv"
    return ""


def norm_rel_path_from_manifest(manifest: dict[str, Any], release_id: str) -> str:
    for row in manifest.get("artifacts") or []:
        if str(row.get("destination_id") or "") == "provider_norm":
            rel = str(row.get("path") or "").replace("\\", "/")
            if rel:
                return rel
    return norm_rel_path_for_release(release_id)


def load_norm_rows_from_bytes(data: bytes) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    if not data:
        return rows
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    for row in reader:
        ccn = normalize_ccn(row.get("ccn") or row.get("PROVNUM") or "")
        if ccn:
            rows[ccn] = {str(k): normalize_cell(v) for k, v in row.items()}
    return rows


def load_norm_rows_from_path(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    return load_norm_rows_from_bytes(path.read_bytes())


def git_show_bytes(dev_pbj_root: Path, sha: str, rel_path: str) -> bytes | None:
    if not sha or not rel_path or not (dev_pbj_root / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "-C", str(dev_pbj_root), "show", f"{sha}:{rel_path.replace(chr(92), '/')}"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    return proc.stdout


def resolve_dev_pbj_root_for_verify(
    explicit: Path | str | None = None,
    publication_record: dict[str, Any] | None = None,
) -> Path:
    """Git repo for `git show` baseline Norm (prefer publication record dev_pbj_root)."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    pub_root = str((publication_record or {}).get("dev_pbj_root") or "").strip()
    if pub_root:
        return Path(pub_root).resolve()
    env = (os.environ.get("PBJ_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    from cms_data_paths import repo_root as data_ops_repo_root

    return (data_ops_repo_root().parent / "pbj-root").resolve()


def resolve_stage_artifact_cache_dir(manifest: dict[str, Any]) -> Path | None:
    """Absolute stage artifact cache from manifest (not control_plane_root/env-relative)."""
    raw = str(manifest.get("stage_artifact_cache") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def load_baseline_norm_rows(
    *,
    dev_pbj_root: Path,
    baseline_sha: str,
    norm_rel: str,
    release_id: str,
) -> dict[str, dict[str, str]]:
    """Baseline Norm at publication base; fall back to prior-month Norm when path was absent."""
    rows, _meta = load_baseline_norm_rows_with_meta(
        dev_pbj_root=dev_pbj_root,
        baseline_sha=baseline_sha,
        norm_rel=norm_rel,
        release_id=release_id,
    )
    return rows


def load_baseline_norm_rows_with_meta(
    *,
    dev_pbj_root: Path,
    baseline_sha: str,
    norm_rel: str,
    release_id: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Baseline Norm at publication_base_sha; prior-month Norm when current month absent at base."""
    meta: dict[str, Any] = {
        "baseline_sha": baseline_sha,
        "norm_rel": norm_rel.replace("\\", "/"),
        "source_rel": None,
        "source_mode": None,
        "dev_pbj_root": str(dev_pbj_root),
    }
    data = git_show_bytes(dev_pbj_root, baseline_sha, norm_rel)
    if data:
        meta["source_rel"] = norm_rel.replace("\\", "/")
        meta["source_mode"] = "git_show_at_publication_base"
        rows = load_norm_rows_from_bytes(data)
        meta["row_count"] = len(rows)
        return rows, meta

    prior = prior_release_id(release_id)
    if prior:
        prior_rel = norm_rel_path_for_release(prior)
        data = git_show_bytes(dev_pbj_root, baseline_sha, prior_rel)
        if data:
            meta["source_rel"] = prior_rel
            meta["source_mode"] = "git_show_prior_month_at_publication_base"
            meta["prior_release_id"] = prior
            rows = load_norm_rows_from_bytes(data)
            meta["row_count"] = len(rows)
            return rows, meta

    meta["source_mode"] = "absent"
    meta["row_count"] = 0
    return {}, meta


def load_expected_norm_rows(
    *,
    cache_dir: Path,
    dev_pbj_root: Path,
    commit_sha: str,
    norm_rel: str,
) -> dict[str, dict[str, str]]:
    """Staged/publication Norm: artifact cache → published commit → dev tree."""
    rows, _meta = load_expected_norm_rows_with_meta(
        cache_dir=cache_dir,
        dev_pbj_root=dev_pbj_root,
        commit_sha=commit_sha,
        norm_rel=norm_rel,
    )
    return rows


def load_expected_norm_rows_from_stage_cache(
    *,
    manifest: dict[str, Any],
    release_id: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Expected Norm from manifest stage_artifact_cache only (production verification source)."""
    cache_dir = resolve_stage_artifact_cache_dir(manifest)
    norm_rel = norm_rel_path_from_manifest(manifest, release_id)
    rel_posix = norm_rel.replace("\\", "/")
    path = (cache_dir / rel_posix.replace("/", os.sep)) if cache_dir else None
    meta: dict[str, Any] = {
        "source_mode": "stage_artifact_cache",
        "cache_dir": str(cache_dir) if cache_dir else None,
        "norm_rel": rel_posix,
        "path": str(path) if path else None,
        "present": bool(path and path.is_file()),
    }
    if not path or not path.is_file():
        meta["row_count"] = 0
        return {}, meta
    rows = load_norm_rows_from_path(path)
    meta["row_count"] = len(rows)
    return rows, meta


def load_expected_norm_rows_with_meta(
    *,
    cache_dir: Path,
    dev_pbj_root: Path,
    commit_sha: str,
    norm_rel: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Staged/publication Norm: artifact cache → published commit → dev tree."""
    rel_posix = norm_rel.replace("\\", "/")
    cache_path = cache_dir / rel_posix.replace("/", os.sep)
    rows = load_norm_rows_from_path(cache_path)
    if rows:
        return rows, {
            "source_mode": "artifact_cache",
            "path": str(cache_path),
            "row_count": len(rows),
        }
    if commit_sha:
        data = git_show_bytes(dev_pbj_root, commit_sha, rel_posix)
        if data:
            rows = load_norm_rows_from_bytes(data)
            if rows:
                return rows, {
                    "source_mode": "git_show_commit_sha",
                    "commit_sha": commit_sha,
                    "norm_rel": rel_posix,
                    "row_count": len(rows),
                }
    disk_path = dev_pbj_root / rel_posix.replace("/", os.sep)
    rows = load_norm_rows_from_path(disk_path)
    return rows, {
        "source_mode": "dev_pbj_root_disk",
        "path": str(disk_path),
        "row_count": len(rows),
    }


def prepare_norm_release_diff_candidates(
    *,
    manifest: dict[str, Any],
    release_id: str,
    dev_pbj_root: Path,
    publication_record: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive Norm diff candidates from publication_base Git baseline + stage artifact cache."""
    baseline_sha = str(
        manifest.get("publication_base_sha")
        or (publication_record or {}).get("publish_base_sha")
        or ""
    )
    norm_rel = norm_rel_path_from_manifest(manifest, release_id)
    processing_prefix = release_processing_prefix(release_id) or ""

    baseline_rows, baseline_meta = load_baseline_norm_rows_with_meta(
        dev_pbj_root=dev_pbj_root,
        baseline_sha=baseline_sha,
        norm_rel=norm_rel,
        release_id=release_id,
    )
    expected_rows, expected_meta = load_expected_norm_rows_from_stage_cache(
        manifest=manifest,
        release_id=release_id,
    )

    candidates = derive_norm_release_diff_candidates(
        baseline_rows=baseline_rows,
        expected_rows=expected_rows,
        processing_prefix=processing_prefix,
    )
    provenance = {
        "baseline": baseline_meta,
        "expected": expected_meta,
        "processing_prefix": processing_prefix,
        "candidate_count": len(candidates),
    }
    return candidates, provenance


def derive_norm_release_diff_candidates(
    *,
    baseline_rows: dict[str, dict[str, str]],
    expected_rows: dict[str, dict[str, str]],
    processing_prefix: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for ccn, exp_row in expected_rows.items():
        pd = normalize_cell(exp_row.get("processing_date"))
        if processing_prefix and pd and not pd.startswith(processing_prefix):
            continue
        base_row = baseline_rows.get(ccn)
        if not base_row:
            candidates.append(
                {
                    "ccn": ccn,
                    "fields": list(NORM_DIFF_COLUMNS),
                    "expected": {col: normalize_cell(exp_row.get(col)) for col in NORM_DIFF_COLUMNS},
                    "reason": "new_in_release",
                    "priority": 2,
                }
            )
            continue
        changed = [
            col
            for col in NORM_DIFF_COLUMNS
            if normalize_cell(base_row.get(col)) != normalize_cell(exp_row.get(col))
        ]
        if changed:
            candidates.append(
                {
                    "ccn": ccn,
                    "fields": changed,
                    "expected": {col: normalize_cell(exp_row.get(col)) for col in changed},
                    "reason": "baseline_diff",
                    "priority": 0 if any(c.endswith("_rating") for c in changed) else 1,
                }
            )
    candidates.sort(key=lambda c: (c.get("priority", 9), str(c.get("ccn") or "")))
    return candidates[:50]


def facility_map(search_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in search_index.get("f") or []:
        if not isinstance(row, dict):
            continue
        ccn = normalize_ccn(str(row.get("c") or row.get("ccn") or ""))
        if ccn:
            out[ccn] = row
    return out


def derive_search_index_diff_candidates(
    *,
    expected_index: dict[str, Any],
    baseline_index: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Fallback: baseline→expected search_index field diffs (fields actually on /search_index.json)."""
    expected_map = facility_map(expected_index)
    baseline_map = facility_map(baseline_index or {})
    candidates: list[dict[str, Any]] = []
    for ccn, exp_row in expected_map.items():
        base_row = baseline_map.get(ccn)
        if not base_row:
            continue
        changed = [
            key
            for key in SEARCH_INDEX_DIFF_KEYS
            if normalize_cell(exp_row.get(key)) != normalize_cell(base_row.get(key))
        ]
        if changed:
            field_map = {"n": "provider_name", "h": "sff_status"}
            candidates.append(
                {
                    "ccn": ccn,
                    "fields": [field_map.get(k, k) for k in changed],
                    "expected": {
                        field_map.get(k, k): normalize_cell(exp_row.get(k)) for k in changed
                    },
                    "reason": "search_index_diff",
                    "priority": 0,
                    "search_index_keys": changed,
                }
            )
    candidates.sort(key=lambda c: (c.get("priority", 9), str(c.get("ccn") or "")))
    return candidates[:50]


def http_get_bytes(url: str, *, timeout: float = 45.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "PBJ320-DataOps-Verify/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(getattr(resp, "status", 200) or 200), resp.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read() if exc.fp else b""


def http_get_json(url: str, *, timeout: float = 45.0) -> tuple[int, dict[str, Any] | None]:
    status, body = http_get_bytes(url, timeout=timeout)
    if status != 200 or not body:
        return status, None
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return status, None
    return status, payload if isinstance(payload, dict) else None


def fetch_provider_json(
    origin: str,
    ccn: str,
    cache: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    if ccn in cache:
        return cache[ccn]
    url = f"{origin.rstrip('/')}/api/public/provider/{ccn}.json"
    _status, payload = http_get_json(url)
    cache[ccn] = payload
    return payload


def verify_norm_field_on_live(
    *,
    origin: str,
    ccn: str,
    field: str,
    expected: str,
    live_index: dict[str, Any],
    provider_cache: dict[str, dict[str, Any] | None],
    search_index_keys: list[str] | None = None,
) -> tuple[bool, str]:
    live_row = facility_map(live_index).get(ccn) or {}

    if field == "provider_name":
        live_name = normalize_cell(live_row.get("n"))
        if live_name and live_name == expected[:80]:
            return True, "search_index.n"
        provider = fetch_provider_json(origin, ccn, provider_cache) or {}
        api_name = normalize_cell((provider.get("facility") or {}).get("name"))
        if api_name and api_name == expected:
            return True, "api_provider.name"
        return False, "provider_name_unverified"

    if field in {"overall_rating", "health_inspection_rating"}:
        provider = fetch_provider_json(origin, ccn, provider_cache) or {}
        ratings = provider.get("cms_ratings") or {}
        api_key = "overall" if field == "overall_rating" else "health_inspection"
        live_val = normalize_rating(ratings.get(api_key))
        if live_val and live_val == normalize_rating(expected):
            return True, f"api_provider.cms_ratings.{api_key}"
        return False, f"{field}_unverified"

    if field == "sff_status":
        if search_index_keys and "h" in search_index_keys:
            live_h = normalize_cell(live_row.get("h"))
            exp_up = expected.upper()
            if not expected:
                if live_h and ("SFF" in live_h.upper() or "CANDIDATE" in live_h.upper()):
                    return False, "sff_status_unverified"
                return True, "search_index.h (no sff tag)"
            if exp_up in live_h.upper() or ("CANDIDATE" in exp_up and "CANDIDATE" in live_h.upper()):
                return True, "search_index.h"
        return False, "sff_status_unverified"

    return False, "no_public_surface"


def verify_data_level_release_diff(
    *,
    origin: str,
    candidates: list[dict[str, Any]],
    live_index: dict[str, Any],
) -> tuple[bool, str, str, list[str]]:
    provider_cache: dict[str, dict[str, Any] | None] = {}
    for cand in candidates:
        ccn = normalize_ccn(str(cand.get("ccn") or ""))
        expected_by_field = dict(cand.get("expected") or {})
        index_keys = [str(k) for k in (cand.get("search_index_keys") or [])]
        for field in [str(f) for f in (cand.get("fields") or [])]:
            expected = normalize_cell(expected_by_field.get(field))
            ok, surface = verify_norm_field_on_live(
                origin=origin,
                ccn=ccn,
                field=field,
                expected=expected,
                live_index=live_index,
                provider_cache=provider_cache,
                search_index_keys=index_keys or None,
            )
            if ok:
                return True, ccn, field, [surface]
    tried = len(candidates)
    return False, "", "", [f"no matching candidate among {tried}"]


def check_row(
    *,
    check_id: str,
    target: str,
    expected: str,
    actual: str,
    passed: bool,
    artifact_sha: str | None = None,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "target": target,
        "expected": expected,
        "actual": actual,
        "result": "PASS" if passed else "FAIL",
        "artifact_sha": artifact_sha,
        "verification_timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def persist_publication_verification(
    pub: dict[str, Any],
    *,
    checks: list[dict[str, Any]],
    dry_run: bool,
    write_record: Callable[[dict[str, Any]], None],
) -> None:
    all_pass = all(c.get("result") == "PASS" for c in checks)
    verified_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if dry_run:
        return
    pub_record = dict(pub)
    layers = dict(pub_record.get("destination_layers") or {})
    if all_pass:
        layers.update(
            {
                "committed": "YES",
                "pushed": "YES",
                "deployed": "YES",
                "production_verified": "YES",
            }
        )
        pub_record["status"] = "VERIFIED"
        pub_record["production_verified"] = True
        pub_record["production_verified_at"] = verified_at
    else:
        layers["production_verified"] = "NO"
        pub_record["production_verified"] = False
    pub_record["destination_layers"] = layers
    pub_record["production_verification_checks"] = checks
    write_record(pub_record)


def git_commit_on_branch(dev_pbj_root: Path, commit_sha: str, *, remote: str, branch: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(dev_pbj_root), "branch", "-r", "--contains", commit_sha],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False
    needle = f"{remote}/{branch}"
    return any(needle in line for line in (proc.stdout or "").splitlines())


def release_processing_prefix(release_id: str) -> str | None:
    parts = (release_id or "").strip().split("-", 1)
    if len(parts) != 2:
        return None
    year, month = parts[0], parts[1]
    if len(year) == 4 and len(month) == 2 and month.isdigit():
        return f"{year}-{month}"
    return None


def load_search_index_json(data: bytes) -> dict[str, Any]:
    if not data:
        return {}
    try:
        payload = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def search_index_sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest() if body else ""
