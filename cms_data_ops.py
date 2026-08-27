"""PBJ Data Ops status probes and Provider Info action wrappers.

Control plane over canonical services — UI must call these modules, not
duplicate ETL. Provider Info check/acquire → scripts/cms_provider_info_acquire.py.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

_ROOT = Path(__file__).resolve().parent
_SCRIPTS = _ROOT / "scripts"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import cms_data_paths  # noqa: E402
from cms_source_registry import (  # noqa: E402
    CMS_SOURCE_REGISTRY,
    AccessMode,
    CmsSourceRecord,
    OpsStatus,
    SourceFamily,
    get_derived_signals,
    get_registry,
    get_source,
)
from data_ops_access import (  # noqa: E402
    ArtifactRef,
    RuntimeAvailability,
    cms_http_ref,
    local_file_ref,
)
from data_ops_approval import has_acknowledgement, has_approval, read_audit  # noqa: E402
from active_release_registry import sha256_file  # noqa: E402
from data_ops_zweli import (  # noqa: E402
    BaselineAvailability,
    ZweliReport,
    ZweliState,
    expected_prior_month,
    not_run_report,
    run_provider_info_zweli,
    write_zweli_report,
)

FetchJson = Callable[[str], Any]


@dataclass
class SourceOpsSnapshot:
    source_id: str
    human_name: str
    source_family: str
    formats: list[str]
    containers: list[str]
    cadence: str
    automation_maturity: str
    cms_dataset_id: Optional[str]
    publisher_latest: Optional[str]
    raw_available: str
    processed: Optional[str]
    quality_reviewed: str
    approved: str
    structural_status: str
    zweli_status: str
    runtime_access: str
    status: str
    last_checked: str
    last_successful_local_processing: Optional[str]
    local_raw_present: bool = False
    local_derived_present: bool = False
    detail: str = ""
    actions_enabled: list[str] = field(default_factory=list)
    error: Optional[str] = None
    # Compat with initial #64 UI fields
    cms_latest: Optional[str] = None
    pbjapp_latest: Optional[str] = None
    automation_level: Optional[str] = None
    release_id: Optional[str] = None
    zweli_report: Optional[dict[str, Any]] = None
    canonical_source_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mtime_iso(path: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _latest_by_glob(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.is_dir():
        return None
    files = [p for p in directory.glob(pattern) if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _quarter_label_from_name(name: str) -> Optional[str]:
    m = re.search(r"CY(\d{4})Q([1-4])", name, re.I)
    if m:
        return f"CY{m.group(1)}Q{m.group(2)}"
    return None


_MONTH_NAME_TO_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _chain_label_from_name(name: str) -> Optional[str]:
    m = re.search(
        r"Nursing_Home_(?:Chain_Performance|Affiliated_Entity)_Measures_([A-Za-z]+)_(\d{4})",
        name,
    )
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return None


def _chain_sort_key(path: Path) -> tuple[int, int, float]:
    m = re.search(
        r"Nursing_Home_(?:Chain_Performance|Affiliated_Entity)_Measures_([A-Za-z]+)_(\d{4})",
        path.name,
    )
    if not m:
        return (0, 0, path.stat().st_mtime)
    mon = _MONTH_NAME_TO_NUM.get(m.group(1).lower(), 0)
    return (int(m.group(2)), mon, path.stat().st_mtime)


def _base_snap(record: CmsSourceRecord) -> SourceOpsSnapshot:
    formats = [f.value for f in record.formats] if record.formats else [c.value for c in record.containers]
    return SourceOpsSnapshot(
        source_id=record.source_id,
        human_name=record.human_name,
        source_family=record.source_family.value,
        formats=formats,
        containers=[c.value for c in record.containers],
        cadence=record.cadence.value,
        automation_maturity=record.automation_maturity.value,
        cms_dataset_id=record.cms_dataset_id,
        publisher_latest=None,
        raw_available="—",
        processed=None,
        quality_reviewed=ZweliState.NOT_RUN.value,
        approved="no",
        structural_status="NOT_RUN",
        zweli_status=ZweliState.NOT_RUN.value,
        runtime_access=AccessMode.UNAVAILABLE.value,
        status=OpsStatus.UNKNOWN.value,
        last_checked=_utc_now_iso(),
        last_successful_local_processing=None,
        actions_enabled=list(record.actions_enabled),
        automation_level=record.automation_maturity.value,
    )


def _apply_raw_ref(snap: SourceOpsSnapshot, ref: ArtifactRef) -> None:
    snap.runtime_access = ref.access_mode.value
    if ref.availability == RuntimeAvailability.AVAILABLE:
        snap.raw_available = ref.release_id or "yes"
        snap.local_raw_present = ref.access_mode == AccessMode.LOCAL_FILESYSTEM
        snap.canonical_source_path = ref.path
    elif ref.availability == RuntimeAvailability.NOT_AVAILABLE_IN_THIS_RUNTIME:
        snap.raw_available = "NOT AVAILABLE IN THIS RUNTIME"
        snap.status = OpsStatus.NOT_AVAILABLE_IN_THIS_RUNTIME.value
    else:
        snap.raw_available = "—"


def _probe_provider_info(
    record: CmsSourceRecord,
    *,
    check_cms: bool,
    fetch_json: FetchJson | None,
    root: Path,
    run_zweli: bool = True,
) -> SourceOpsSnapshot:
    import cms_provider_info_acquire as acq
    import cms_provider_release_lib as cpr

    snap = _base_snap(record)
    try:
        handoff_ready = False
        release_id = None
        local = acq.latest_local_provider_info(root)
        raw_path = None
        norm_path = None
        if local:
            ly, lm, lpath = local
            raw_path = lpath
            release_id = f"{ly:04d}-{lm:02d}"
            snap.release_id = release_id
            snap.pbjapp_latest = f"{acq._MONTH_ABBR[lm]} {ly}"
            snap.processed = None
            snap.local_raw_present = True
            _apply_raw_ref(
                snap,
                local_file_ref("provider_info_raw", lpath, release_id=release_id),
            )
            norm_path = (
                cms_data_paths.provider_info_normalized_dir(root)
                / f"ProviderInfoNorm_{ly}_{lm:02d}.csv"
            )
            snap.local_derived_present = norm_path.is_file() and norm_path.stat().st_size > 0
            if snap.local_derived_present:
                snap.processed = release_id
            key = cpr.release_key(ly, lm)
            acq_path = (
                cms_data_paths.provider_release_manifest_dir(key.label, root) / "acquisition.json"
            )
            handoff_path = (
                cms_data_paths.provider_release_manifest_dir(key.label, root)
                / "pbj_root_handoff.json"
            )
            if acq_path.is_file():
                try:
                    acq_data = json.loads(acq_path.read_text(encoding="utf-8"))
                    snap.last_successful_local_processing = acq_data.get("acquired_at") or _mtime_iso(
                        acq_path
                    )
                    snap.structural_status = "PASS" if acq_data.get("validation") else "UNKNOWN"
                except (OSError, json.JSONDecodeError):
                    snap.last_successful_local_processing = _mtime_iso(acq_path)
            elif snap.local_derived_present:
                snap.last_successful_local_processing = _mtime_iso(norm_path)
                snap.structural_status = "PASS"
            else:
                snap.last_successful_local_processing = _mtime_iso(lpath)

            if handoff_path.is_file():
                try:
                    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
                    promo = handoff.get("provider_promotion") or {}
                    handoff_ready = bool(promo.get("ready_for_pbj_commit")) or bool(
                        (handoff.get("pbj_root_sync") or {}).get("sha256")
                    )
                except (OSError, json.JSONDecodeError):
                    handoff_ready = True

            if has_approval(record.source_id, release_id):
                snap.approved = "yes"
        else:
            _apply_raw_ref(
                snap,
                local_file_ref(
                    "provider_info_raw",
                    cms_data_paths.provider_info_dir(root) / "(none)",
                    expected_elsewhere=True,
                ),
            )

        cms = None
        if check_cms:
            cms = acq.resolve_cms_provider_info_release(fetch_json=fetch_json)
            snap.publisher_latest = cms.data_vintage_label
            snap.cms_latest = cms.data_vintage_label

        # Zweli — comparable prior = previous calendar month for monthly PI.
        # Resolve the baseline from the authoritative ACTIVE registry first;
        # legacy same-directory snapshots are only eligible when no matching
        # ACTIVE record exists. Never infer "latest" here.
        if run_zweli and raw_path and raw_path.is_file() and local:
            ey, em = expected_prior_month(local[0], local[1])
            expected_rel = f"{ey:04d}-{em:02d}"
            baseline_path = None
            from active_release_registry import get_active_release, registry_path
            from urllib.parse import unquote, urlparse
            active = get_active_release("cms.provider_info", registry_path(Path(__file__).resolve().parent))
            if active and active.get("active_release_id") == expected_rel:
                uri = str(active.get("source_uri") or "")
                parsed = urlparse(uri)
                if parsed.scheme == "file":
                    active_raw = unquote(parsed.path)
                    if os.name == "nt" and active_raw.startswith("/") and len(active_raw) > 2 and active_raw[2] == ":":
                        active_raw = active_raw[1:]
                    candidate = Path(active_raw)
                    if candidate.is_file() and sha256_file(candidate) == active.get("hash"):
                        baseline_path = candidate
            if baseline_path is None and not active:
                for y, m, p in acq.list_local_provider_info_snapshots(root):
                    if (y, m) == (ey, em):
                        baseline_path = p
                        break
            current_compare_path = norm_path if norm_path and norm_path.is_file() else raw_path
            if baseline_path is not None:
                report = run_provider_info_zweli(
                    current_compare_path,
                    release_id or "unknown",
                    baseline_csv=baseline_path,
                    baseline_release=expected_rel,
                    baseline_availability=BaselineAvailability.PRESENT,
                    expected_baseline_release=expected_rel,
                )
            else:
                report = run_provider_info_zweli(
                    current_compare_path,
                    release_id or "unknown",
                    baseline_csv=None,
                    baseline_release=expected_rel,
                    baseline_availability=BaselineAvailability.UNAVAILABLE_IN_RUNTIME,
                    expected_baseline_release=expected_rel,
                )
            snap.zweli_status = report.state.value
            snap.quality_reviewed = report.state.value
            snap.zweli_report = report.to_dict()
            if release_id:
                try:
                    write_zweli_report(
                        report,
                        cms_data_paths.provider_release_manifest_dir(release_id, root)
                        / "zweli_report.json",
                    )
                except OSError:
                    pass
        else:
            snap.zweli_status = ZweliState.NOT_RUN.value
            snap.quality_reviewed = ZweliState.NOT_RUN.value

        if cms is not None and acq.cms_is_newer_than_local(cms, root):
            snap.status = OpsStatus.CMS_NEWER.value
            snap.detail = "CMS metastore vintage newer than local NH_ProviderInfo CSV"
            return snap

        if not snap.local_raw_present:
            snap.status = OpsStatus.NOT_AVAILABLE_IN_THIS_RUNTIME.value
            snap.detail = "No real (non-LFS) local NH_ProviderInfo CSV in this runtime"
            return snap

        if not snap.local_derived_present:
            snap.status = OpsStatus.PROCESSING_REQUIRED.value
            snap.detail = "Raw Provider Info present; normalized output missing"
            return snap

        if snap.zweli_status == ZweliState.BLOCKED.value:
            snap.status = OpsStatus.ERROR.value
            snap.detail = "Zweli BLOCKED — not ready for handoff"
            return snap

        if handoff_ready and cms is not None:
            snap.status = OpsStatus.READY_FOR_HANDOFF.value
            snap.detail = "Local matches CMS; Norm + handoff artifact present"
            return snap

        if cms is not None:
            snap.status = OpsStatus.CURRENT.value
            snap.detail = "Local Provider Info vintage matches CMS"
            return snap

        snap.status = OpsStatus.UNKNOWN.value
        snap.detail = "Local processed snapshot present; CMS not checked"
        return snap
    except Exception as exc:  # noqa: BLE001
        snap.status = OpsStatus.ERROR.value
        snap.error = str(exc)
        snap.detail = f"Provider Info probe failed: {exc}"
        return snap


def _probe_nurse(
    record: CmsSourceRecord,
    root: Path,
    *,
    check_cms: bool,
    fetch_json: FetchJson | None = None,
) -> SourceOpsSnapshot:
    snap = _probe_quarterly_csv_family(
        record,
        root,
        raw_dir=cms_data_paths.nurse_raw_dir(root),
        std_dir=cms_data_paths.standardized_nurse_dir(root),
        raw_glob="PBJ_dailynurse*.csv",
        std_glob="PBJ_dailynurse*.csv",
    )
    if not check_cms:
        return snap
    try:
        import cms_pbj_nurse_acquire as nurse_acq

        cms = nurse_acq.resolve_cms_nurse_release(fetch_json=fetch_json)
        snap.publisher_latest = cms.quarter_label
        snap.cms_latest = cms.quarter_label
        identity = nurse_acq.assess_local_release_identity(cms, root=root)
        snap.detail = identity.detail
        if identity.verdict == nurse_acq.IDENTITY_MANIFEST_MISMATCH:
            snap.status = OpsStatus.ERROR.value
            snap.structural_status = "FAIL"
            snap.error = identity.detail
            snap.detail = f"Provenance mismatch: {identity.detail}"
        elif identity.cryptographically_identical:
            if snap.local_derived_present:
                snap.status = OpsStatus.CURRENT.value
                snap.detail = (
                    "Local nurse quarter cryptographically matches CMS Primary "
                    "(manifest SHA + raw + standardized)"
                )
            else:
                snap.status = OpsStatus.PROCESSING_REQUIRED.value
                snap.detail = (
                    "Manifest-identical raw present; standardization required"
                )
        elif identity.verdict == nurse_acq.IDENTITY_UNMANIFESTED_OK:
            snap.status = OpsStatus.UNKNOWN.value
            snap.detail = (
                f"CMS {cms.quarter_label}: {identity.detail}"
            )
            if snap.local_raw_present and not snap.local_derived_present:
                snap.status = OpsStatus.PROCESSING_REQUIRED.value
        elif nurse_acq.cms_is_newer_than_local(cms, root):
            snap.status = OpsStatus.CMS_NEWER.value
            snap.detail = (
                f"CMS Primary {cms.quarter_label} newer/missing vs local "
                f"{snap.pbjapp_latest or '(none)'}"
            )
            if not snap.local_raw_present:
                snap.raw_available = "NOT AVAILABLE IN THIS RUNTIME"
        elif snap.local_raw_present and not snap.local_derived_present:
            snap.status = OpsStatus.PROCESSING_REQUIRED.value
            snap.detail = "CMS vintage present raw; standardization required"
        else:
            snap.status = OpsStatus.CMS_NEWER.value
            snap.detail = "CMS Primary known; raw not available in this runtime"
            snap.raw_available = "NOT AVAILABLE IN THIS RUNTIME"
    except Exception as exc:  # noqa: BLE001
        snap.status = OpsStatus.ERROR.value
        snap.error = str(exc)
        snap.detail = f"Nurse CMS probe failed: {exc}"
    return snap


def _probe_quarterly_csv_family(
    record: CmsSourceRecord,
    root: Path,
    *,
    raw_dir: Path,
    std_dir: Path,
    raw_glob: str,
    std_glob: str,
) -> SourceOpsSnapshot:
    snap = _base_snap(record)
    def _quarter_key(path: Path) -> tuple[int, int]:
        label = _quarter_label_from_name(path.name) or ""
        match = re.search(r"CY(\d{4})Q([1-4])", label)
        return (int(match.group(1)), int(match.group(2))) if match else (0, 0)

    raw_files = [p for p in raw_dir.glob(raw_glob) if p.is_file()] if raw_dir.is_dir() else []
    std_files = [p for p in std_dir.glob(std_glob) if p.is_file()] if std_dir.is_dir() else []
    raw = max(raw_files, key=_quarter_key) if raw_files else None
    std = max(std_files, key=_quarter_key) if std_files else None
    canonical = max([p for p in (raw, std) if p], key=_quarter_key, default=None)
    label = _quarter_label_from_name(canonical.name) if canonical else None
    snap.pbjapp_latest = label
    snap.release_id = label
    snap.processed = label if std else None
    if canonical:
        _apply_raw_ref(snap, local_file_ref("raw", canonical, release_id=label))
        snap.local_raw_present = True
    else:
        _apply_raw_ref(
            snap,
            local_file_ref("raw", raw_dir / "(none)", expected_elsewhere=True),
        )
        snap.status = OpsStatus.NOT_AVAILABLE_IN_THIS_RUNTIME.value
        snap.detail = (
            "Raw not accessible in this runtime (may exist under PBJ_DATA_ROOT / "
            "operator machine). CMS dataset ID is registered."
        )
        return snap
    snap.local_derived_present = bool(std)
    snap.last_successful_local_processing = _mtime_iso(std) if std else None
    if raw and not std:
        snap.status = OpsStatus.PROCESSING_REQUIRED.value
        snap.structural_status = "PENDING"
        snap.detail = "Raw present; standardized output missing"
    else:
        snap.status = OpsStatus.UNKNOWN.value
        snap.structural_status = "UNKNOWN"
        snap.detail = (
            f"Local artifacts present; publisher comparison not automated for "
            f"{record.cms_dataset_id}"
        )
    snap.zweli_status = ZweliState.NOT_RUN.value
    snap.quality_reviewed = ZweliState.NOT_RUN.value
    return snap


def _probe_ein(record: CmsSourceRecord, root: Path) -> SourceOpsSnapshot:
    snap = _base_snap(record)
    mono = cms_data_paths.ein_monolithic_dir(root)
    quarters = cms_data_paths.ein_quarters_dir(root)
    extracted = cms_data_paths.ein_extracted_dir(root)
    ein_root = cms_data_paths.ein_root(root)
    mono_zips = list(mono.glob("*.zip")) if mono.is_dir() else []
    q_zips = list(quarters.glob("*.zip")) if quarters.is_dir() else []
    root_zips = list(ein_root.glob("*.zip")) if ein_root.is_dir() else []
    raw_present = bool(mono_zips or q_zips or root_zips)
    derived = _latest_by_glob(extracted, "CY*.csv") if extracted.is_dir() else None
    label = None
    if q_zips:
        labels = [x for x in (_quarter_label_from_name(z.name) for z in q_zips) if x]
        if labels:
            label = max(labels)
    if label is None and mono_zips:
        label = "monolithic PUF"
    elif label is None and root_zips:
        label = root_zips[0].name
    snap.pbjapp_latest = label
    snap.processed = label if derived else None
    if raw_present:
        path = (q_zips or mono_zips or root_zips)[0]
        _apply_raw_ref(snap, local_file_ref("ein_zip", path, release_id=label))
        snap.local_raw_present = True
        snap.status = (
            OpsStatus.LOCAL_RAW_ONLY.value if not derived else OpsStatus.UNKNOWN.value
        )
        snap.detail = "EIN zip(s) in runtime; broken-legacy ingest scripts not restored"
    else:
        _apply_raw_ref(
            snap, local_file_ref("ein_zip", ein_root / "(none)", expected_elsewhere=True)
        )
        snap.status = OpsStatus.NOT_AVAILABLE_IN_THIS_RUNTIME.value
        snap.detail = "EIN zips not available in this runtime"
    snap.local_derived_present = bool(derived)
    snap.last_successful_local_processing = _mtime_iso(derived) if derived else None
    return snap


def _probe_snf_all_owners(record: CmsSourceRecord, root: Path) -> SourceOpsSnapshot:
    snap = _base_snap(record)
    own = cms_data_paths.ownership_dir(root)
    raw = None
    policy_path = own / "ownership_release_policy.json"
    if policy_path.is_file():
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            release = str(policy.get("active_release_date") or "")
            filename = str(((policy.get("releases") or {}).get(release) or {}).get("ownership_source_filename") or "")
            candidate = own / filename
            raw = candidate if candidate.is_file() else None
        except (OSError, json.JSONDecodeError):
            raw = None
    if raw:
        _apply_raw_ref(snap, local_file_ref("snf_all_owners", raw, release_id=raw.name))
        snap.release_id = release or raw.name
        snap.pbjapp_latest = raw.name
        snap.status = OpsStatus.LOCAL_RAW_ONLY.value
        snap.detail = "CSV present; normalize scripts absent on main"
        snap.last_successful_local_processing = _mtime_iso(raw)
        snap.local_raw_present = True
    else:
        _apply_raw_ref(
            snap, local_file_ref("snf_all_owners", own / "(none)", expected_elsewhere=True)
        )
        snap.status = OpsStatus.NOT_AVAILABLE_IN_THIS_RUNTIME.value
        snap.detail = "SNF_All_Owners not in this runtime (dataset ID registered)"
    return snap


def _probe_snf_enrollments(record: CmsSourceRecord, root: Path) -> SourceOpsSnapshot:
    snap = _base_snap(record)
    policy_path = cms_data_paths.ownership_dir(root) / "ownership_release_policy.json"
    raw = None
    release = ""
    if policy_path.is_file():
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            release = str(policy.get("active_release_date") or "")
            entry = ((policy.get("releases") or {}).get(release) or {})
            filename = str(entry.get("enrollment_source_filename") or "")
            candidate = (
                cms_data_paths.ownership_dir(root)
                / "_sources" / "cms_snf_enrollments" / "raw" / "downloaded" / filename
            )
            raw = candidate if candidate.is_file() else None
        except (OSError, json.JSONDecodeError):
            raw = None
    if raw:
        _apply_raw_ref(snap, local_file_ref("snf_enrollments", raw, release_id=release))
        snap.release_id = release
        snap.pbjapp_latest = release
        snap.status = OpsStatus.LOCAL_RAW_ONLY.value
        snap.detail = "Policy-aligned enrollment snapshot present; validation required before promotion"
        snap.local_raw_present = True
    else:
        snap.status = OpsStatus.NOT_AVAILABLE_IN_THIS_RUNTIME.value
        snap.detail = "Policy-selected SNF enrollment snapshot unavailable"
    return snap


def _probe_chain(record: CmsSourceRecord, root: Path) -> SourceOpsSnapshot:
    snap = _base_snap(record)
    own = cms_data_paths.ownership_dir(root)
    candidates: list[Path] = []
    if own.is_dir():
        candidates.extend(own.glob("Nursing_Home_Chain_Performance_Measures_*.csv"))
        candidates.extend(own.glob("Nursing_Home_Affiliated_Entity_Performance_Measures_*.csv"))
    raw = max(candidates, key=_chain_sort_key) if candidates else None
    label = _chain_label_from_name(raw.name) if raw else None
    if raw:
        _apply_raw_ref(snap, local_file_ref("chain", raw, release_id=label))
        snap.pbjapp_latest = label or raw.name
        snap.status = OpsStatus.LOCAL_RAW_ONLY.value
        snap.detail = "Chain performance CSV present; manual acquire"
        snap.last_successful_local_processing = _mtime_iso(raw)
        snap.local_raw_present = True
    else:
        _apply_raw_ref(snap, local_file_ref("chain", own / "(none)", expected_elsewhere=True))
        snap.status = OpsStatus.NOT_AVAILABLE_IN_THIS_RUNTIME.value
        snap.detail = "Chain performance CSV not in this runtime"
    return snap


def _probe_unmodeled(record: CmsSourceRecord, detail: str) -> SourceOpsSnapshot:
    snap = _base_snap(record)
    snap.detail = detail
    snap.status = OpsStatus.UNKNOWN.value
    snap.raw_available = "NOT AVAILABLE IN THIS RUNTIME"
    snap.runtime_access = AccessMode.UNAVAILABLE.value
    return snap


def _probe_health_citations(record: CmsSourceRecord, root: Path) -> SourceOpsSnapshot:
    snap = _base_snap(record)
    cit = cms_data_paths.citations_dir(root)
    # Facility citation builder consumes this exact source family, not descriptions.
    candidates = list(cit.glob("NH_HealthCitations_*.csv")) if cit.is_dir() else []
    def _citation_key(path: Path) -> tuple[int, int]:
        m = re.search(r"_([A-Za-z]{3})(\d{4})\.csv$", path.name)
        return (int(m.group(2)), _MONTH_NAME_TO_NUM.get(m.group(1).lower(), 0)) if m else (0, 0)
    standalone = max(candidates, key=_citation_key) if candidates else None
    if standalone:
        _apply_raw_ref(
            snap, local_file_ref("citations", standalone, release_id=standalone.name)
        )
        snap.pbjapp_latest = standalone.name
        snap.release_id = standalone.name
        snap.local_raw_present = True
        snap.status = OpsStatus.LOCAL_RAW_ONLY.value
        snap.detail = (
            "Citation CSV present in Citations/. Distinct dataset r5ix-sfxw — "
            "co-extracted NH_HealthCitations_* from PI zip is a different path."
        )
        snap.last_successful_local_processing = _mtime_iso(standalone)
    else:
        _apply_raw_ref(
            snap, local_file_ref("citations", cit / "(none)", expected_elsewhere=True)
        )
        snap.status = OpsStatus.NOT_AVAILABLE_IN_THIS_RUNTIME.value
        snap.detail = "Health Citations dataset not accessible in this runtime"
    return snap


def probe_source(
    source_id: str,
    *,
    check_cms: bool = True,
    fetch_json: FetchJson | None = None,
    root: Path | None = None,
    run_zweli: bool = True,
) -> SourceOpsSnapshot:
    root = root or cms_data_paths.repo_root()
    record = get_source(source_id)
    if record is None:
        return SourceOpsSnapshot(
            source_id=source_id,
            human_name=source_id,
            source_family="unknown",
            formats=[],
            containers=[],
            cadence="unknown",
            automation_maturity="unmodeled",
            cms_dataset_id=None,
            publisher_latest=None,
            raw_available="—",
            processed=None,
            quality_reviewed=ZweliState.NOT_RUN.value,
            approved="no",
            structural_status="NOT_RUN",
            zweli_status=ZweliState.NOT_RUN.value,
            runtime_access=AccessMode.UNAVAILABLE.value,
            status=OpsStatus.ERROR.value,
            last_checked=_utc_now_iso(),
            last_successful_local_processing=None,
            detail="Unknown source_id",
            error=f"unknown source_id: {source_id}",
        )

    family = record.source_family
    if family == SourceFamily.PROVIDER_INFO:
        return _probe_provider_info(
            record, check_cms=check_cms, fetch_json=fetch_json, root=root, run_zweli=run_zweli
        )
    if family == SourceFamily.PBJ_NURSE:
        return _probe_nurse(record, root, check_cms=check_cms, fetch_json=fetch_json)
    if family == SourceFamily.PBJ_NON_NURSE:
        return _probe_quarterly_csv_family(
            record,
            root,
            raw_dir=cms_data_paths.nonnurse_raw_dir(root),
            std_dir=cms_data_paths.standardized_nonnurse_dir(root),
            raw_glob="PBJ_dailynonnurse*.csv",
            std_glob="PBJ_dailynonnurse*.csv",
        )
    if family == SourceFamily.PBJ_EIN:
        return _probe_ein(record, root)
    if family == SourceFamily.SNF_ALL_OWNERS:
        return _probe_snf_all_owners(record, root)
    if family == SourceFamily.SNF_ENROLLMENTS:
        return _probe_snf_enrollments(record, root)
    if family == SourceFamily.SNF_CHOW:
        return _probe_unmodeled(
            record,
            "SNF CHOW manual/cross-repo; raw path often absent in this runtime",
        )
    if family == SourceFamily.CHAIN_PERFORMANCE:
        return _probe_chain(record, root)
    if family == SourceFamily.HEALTH_CITATIONS:
        return _probe_health_citations(record, root)
    if family == SourceFamily.SFF_PDF_LIST:
        return _probe_unmodeled(
            record,
            "SFF PDF/list publication UNMODELED — distinct from signal.sff_status on Provider Info",
        )
    return _probe_unmodeled(record, "Unhandled source family")


def probe_all_sources(
    *,
    check_cms: bool = True,
    fetch_json: FetchJson | None = None,
    root: Path | None = None,
    run_zweli: bool = True,
) -> list[SourceOpsSnapshot]:
    return [
        probe_source(
            r.source_id,
            check_cms=check_cms,
            fetch_json=fetch_json,
            root=root,
            run_zweli=run_zweli,
        )
        for r in get_registry()
    ]


def _control_plane_ui_view(control: dict[str, Any]) -> dict[str, Any]:
    """Normalize release_control_plane.control_panel_payload() for UI overlay."""
    if control.get("active") is not None and control.get("pending_by_source") is not None:
        return control

    active: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    pending_by_source: dict[str, dict[str, Any]] = {}

    for row in control.get("datasets") or []:
        dataset_id = str(row.get("dataset_id") or "")
        if not dataset_id:
            continue

        active_rec = row.get("active")
        if isinstance(active_rec, dict):
            release_id = str(active_rec.get("active_release_id") or "")
            meta = active_rec.get("metadata") if isinstance(active_rec.get("metadata"), dict) else {}
            validation = active_rec.get("validation") if isinstance(active_rec.get("validation"), dict) else {}
            active[dataset_id] = {
                "source_id": dataset_id,
                "release_id": release_id,
                "status": str(active_rec.get("status") or "ACTIVE").upper(),
                "validation_status": validation.get("status"),
                "zweli_status": meta.get("zweli_status"),
            }

        pending_rec = row.get("pending")
        if isinstance(pending_rec, dict):
            release_id = str(pending_rec.get("release_id") or "")
            meta = pending_rec.get("metadata") if isinstance(pending_rec.get("metadata"), dict) else {}
            validation = pending_rec.get("validation") if isinstance(pending_rec.get("validation"), dict) else {}
            state = str(pending_rec.get("state") or "UNKNOWN").upper()
            normalized = {
                "source_id": dataset_id,
                "release_id": release_id,
                "state": state,
                "validation_status": validation.get("status"),
                "zweli_status": meta.get("zweli_status"),
                "requires_review": bool(pending_rec.get("requires_review"))
                or state in {"ACQUIRED", "VALIDATED"},
                "detail": str(meta.get("detail") or ""),
            }
            candidates.append(normalized)
            pending_by_source[dataset_id] = normalized

    return {
        "active": active,
        "candidates": candidates,
        "pending_by_source": pending_by_source,
    }


def promote_candidate_permitted(candidate: dict[str, Any]) -> bool:
    """Whether lifecycle permits promotion review (read-only; does not promote)."""
    return (candidate.get("state") or "").upper() == "VALIDATED"


def overlay_control_plane_on_snapshot(
    snap: SourceOpsSnapshot | dict[str, Any],
    control: dict[str, Any] | None = None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Merge authoritative control-plane active/pending state onto a probe snapshot."""
    if control is None:
        from release_control_plane import control_panel_payload

        control = control_panel_payload(Path(__file__).resolve().parent)
    payload = _control_plane_ui_view(control)
    data = snap.to_dict() if isinstance(snap, SourceOpsSnapshot) else dict(snap)
    source_id = data.get("source_id") or ""
    active = (payload.get("active") or {}).get(source_id)
    pending = (payload.get("pending_by_source") or {}).get(source_id)

    data["legacy_status"] = data.get("status")
    data["legacy_raw_available"] = data.get("raw_available")

    if active:
        data["active_release_id"] = active.get("release_id")
        data["active_release_status"] = active.get("status")
        data["release_id"] = active.get("release_id") or data.get("release_id")
        data["raw_available"] = active.get("release_id") or data.get("raw_available")
        data["pbjapp_latest"] = active.get("release_id") or data.get("pbjapp_latest")
        if active.get("validation_status") is not None:
            data["validation_status"] = active.get("validation_status")
        if active.get("zweli_status") is not None:
            data["zweli_status"] = active.get("zweli_status")
            data["quality_reviewed"] = active.get("zweli_status")
        if data.get("legacy_status") in {
            OpsStatus.NOT_AVAILABLE_IN_THIS_RUNTIME.value,
            OpsStatus.UNKNOWN.value,
        } and (
            data.get("legacy_raw_available") == "NOT AVAILABLE IN THIS RUNTIME"
            or "NOT AVAILABLE" in str(data.get("legacy_raw_available") or "")
        ):
            data["status"] = active.get("status") or OpsStatus.CURRENT.value
            data["runtime_access"] = AccessMode.REMOTE_STORE.value
            if not data.get("detail") or "UNMODELED" in (data.get("detail") or "").upper():
                data["detail"] = (
                    f"Governed active release {active.get('release_id')} "
                    f"({active.get('status')}); legacy probe unavailable in runtime"
                )
        elif active.get("status") == "ACTIVE":
            data["status"] = OpsStatus.CURRENT.value

    if pending:
        data["pending_release_id"] = pending.get("release_id")
        data["pending_release_state"] = pending.get("state")
        if pending.get("validation_status") is not None:
            data["validation_status"] = pending.get("validation_status")
        if pending.get("zweli_status") is not None and not active:
            data["zweli_status"] = pending.get("zweli_status")
            data["quality_reviewed"] = pending.get("zweli_status")

    data["display_status"] = (
        data.get("active_release_status")
        if active and data.get("active_release_status")
        else data.get("status")
    )
    return data


def snapshots_with_control_plane(
    *,
    check_cms: bool = True,
    fetch_json: FetchJson | None = None,
    root: Path | None = None,
    run_zweli: bool = True,
    control: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Probe all sources and overlay authoritative control-plane state."""
    root = root or cms_data_paths.repo_root()
    if control is None:
        from release_control_plane import control_panel_payload

        control = control_panel_payload(Path(__file__).resolve().parent)
    snaps = probe_all_sources(
        check_cms=check_cms,
        fetch_json=fetch_json,
        root=root,
        run_zweli=run_zweli,
    )
    return [overlay_control_plane_on_snapshot(s, control, root=root) for s in snaps]


def _legacy_release_review_item(s: SourceOpsSnapshot) -> Optional[dict[str, Any]]:
    reason = None
    if s.status == OpsStatus.CMS_NEWER.value:
        reason = "new_source_release"
    elif s.status == OpsStatus.PROCESSING_REQUIRED.value:
        reason = "acquired_unprocessed"
    elif s.structural_status in {"FAIL", "ERROR"}:
        reason = "structural_error"
    elif s.zweli_status == ZweliState.REQUIRES_REVIEW.value:
        reason = "zweli_requires_review"
    elif s.zweli_status == ZweliState.BLOCKED.value:
        reason = "zweli_blocked"
    elif s.status == OpsStatus.ERROR.value:
        reason = "processing_failure"
    elif s.status == OpsStatus.READY_FOR_HANDOFF.value and s.approved != "yes":
        reason = "ready_for_approval"
    if not reason:
        return None
    release_id = s.release_id or s.pbjapp_latest
    return {
        "reason": reason,
        "source_id": s.source_id,
        "human_name": s.human_name,
        "release_id": release_id,
        "status": s.status,
        "zweli_status": s.zweli_status,
        "structural_status": s.structural_status,
        "detail": s.detail,
        "zweli_report": s.zweli_report,
        "acknowledged": bool(release_id and has_acknowledgement(s.source_id, release_id)),
        "approvable": reason == "ready_for_approval"
        or (
            bool(release_id and has_acknowledgement(s.source_id, release_id))
            and s.zweli_status != ZweliState.BLOCKED.value
        ),
        "governed": False,
    }


def _governed_candidate_review_item(
    candidate: dict[str, Any],
    *,
    snap: Optional[SourceOpsSnapshot],
    root: Path | None,
) -> Optional[dict[str, Any]]:
    state = (candidate.get("state") or "").upper()
    requires_review = bool(candidate.get("requires_review"))
    if not requires_review and state not in {
        "ACQUIRED",
        "VALIDATED",
        "STRUCTURAL_FAIL",
        "ZWELI_REQUIRES_REVIEW",
        "ZWELI_BLOCKED",
        "READY",
    }:
        return None

    source_id = candidate["source_id"]
    release_id = candidate["release_id"]
    zweli_status = candidate.get("zweli_status")
    if zweli_status is None and snap is not None:
        zweli_status = snap.zweli_status
    zweli_status = zweli_status or ZweliState.NOT_RUN.value

    reason = "governed_pending_acquired"
    if state == "VALIDATED":
        reason = "governed_pending_validated"
    elif state in {"STRUCTURAL_FAIL", "ERROR"}:
        reason = "governed_structural_error"
    elif state == "ZWELI_REQUIRES_REVIEW" or zweli_status == ZweliState.REQUIRES_REVIEW.value:
        reason = "governed_zweli_requires_review"
    elif state == "ZWELI_BLOCKED" or zweli_status == ZweliState.BLOCKED.value:
        reason = "governed_zweli_blocked"

    if snap is not None and snap.release_id == release_id and snap.zweli_status:
        zweli_status = snap.zweli_status

    approvable = False
    if source_id == "cms.provider_info" and promote_candidate_permitted(candidate):
        if zweli_status in {ZweliState.PASS.value, ZweliState.NOT_RUN.value}:
            approvable = True
        elif zweli_status == ZweliState.REQUIRES_REVIEW.value and has_acknowledgement(
            source_id, release_id
        ):
            approvable = True

    detail = candidate.get("detail") or (
        f"Governed pending release {release_id} ({state}) awaiting operator review"
    )
    zweli_report = None
    if snap is not None and snap.zweli_report and snap.release_id == release_id:
        zweli_report = snap.zweli_report
    elif source_id == "cms.provider_info":
        zweli_report = load_zweli_report_for_release(source_id, release_id, root=root)

    return {
        "reason": reason,
        "source_id": source_id,
        "human_name": (snap.human_name if snap else source_id),
        "release_id": release_id,
        "status": state,
        "pending_state": state,
        "validation_status": candidate.get("validation_status"),
        "zweli_status": zweli_status,
        "structural_status": snap.structural_status if snap else "NOT_RUN",
        "detail": detail,
        "zweli_report": zweli_report,
        "acknowledged": has_acknowledgement(source_id, release_id),
        "approvable": approvable,
        "governed": True,
    }


def release_review_items(
    snapshots: list[SourceOpsSnapshot] | None = None,
    *,
    check_cms: bool = True,
    root: Path | None = None,
    control: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Items needing human attention for Release Review UI."""
    root = root or cms_data_paths.repo_root()
    if control is None:
        from release_control_plane import control_panel_payload

        control = control_panel_payload(Path(__file__).resolve().parent)
    payload = _control_plane_ui_view(control)
    snaps = snapshots or probe_all_sources(check_cms=check_cms, root=root)
    by_id = {s.source_id: s for s in snaps}
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for s in snaps:
        item = _legacy_release_review_item(s)
        if item:
            key = (item["source_id"], str(item.get("release_id") or ""))
            if key not in seen:
                seen.add(key)
                items.append(item)

    for candidate in payload.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        snap = by_id.get(candidate.get("source_id", ""))
        item = _governed_candidate_review_item(candidate, snap=snap, root=root)
        if not item:
            continue
        key = (item["source_id"], str(item.get("release_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        items.append(item)

    return items


def format_do_timestamp(value: str | None, *, suffix: str = " ET") -> str:
    """Compact human-readable timestamp for UI (does not mutate stored values)."""
    if not value or not str(value).strip():
        return "—"
    raw = str(value).strip()
    try:
        from zoneinfo import ZoneInfo

        normalized = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(ZoneInfo("America/New_York"))
        label = dt.strftime("%b %d, %I:%M %p").replace(" 0", " ")
        return f"{label}{suffix}".strip()
    except ValueError:
        return raw


def minimal_record_for_dataset(dataset_id: str) -> dict[str, Any]:
    """Fallback registry row when a control-plane dataset has no CMS source entry."""
    return {
        "source_id": dataset_id,
        "human_name": dataset_id,
        "cms_dataset_id": None,
        "cadence": "—",
        "containers": [],
        "automation_maturity": "—",
        "landing_url": None,
        "raw_artifact_resolver": "—",
        "normalized_artifact_resolver": "—",
        "acquisition_implementation": None,
        "structural_validator": None,
        "zweli_quality_profile": None,
        "cms_dataset_id_provenance": "Control-plane dataset without CMS source registry entry",
        "automation_notes": "",
        "actions_enabled": [],
    }


def _validation_status_from_control(control_row: dict[str, Any] | None) -> str | None:
    if not control_row:
        return None
    pending = control_row.get("pending") or {}
    active = control_row.get("active") or {}
    validation = pending.get("validation") if isinstance(pending, dict) else None
    if isinstance(validation, dict) and validation.get("status"):
        return str(validation.get("status"))
    meta = active.get("metadata") if isinstance(active.get("metadata"), dict) else {}
    if meta.get("validation"):
        return str(meta.get("validation"))
    return None


def _next_operator_action(
    source_id: str,
    *,
    record: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
    control_row: dict[str, Any] | None,
) -> dict[str, Any]:
    pending = (control_row or {}).get("pending") or {}
    pending_state = str(pending.get("state") or "").upper()
    active_id = ((control_row or {}).get("active") or {}).get("active_release_id")
    actions = list((record or {}).get("actions_enabled") or [])

    if pending_state == "ACQUIRED":
        return {
            "label": "Review pending release",
            "detail": f"Pending {pending.get('release_id')} is ACQUIRED — review in Release Review before promotion.",
            "endpoint": "release_review",
            "wired": True,
            "read_only": True,
        }
    if pending_state == "VALIDATED" and source_id == "cms.provider_info":
        approvable = promote_candidate_permitted(
            {
                "state": pending_state,
                "source_id": source_id,
                "release_id": pending.get("release_id"),
            }
        )
        return {
            "label": "Approve for promotion" if approvable else "Complete Zweli / acknowledgement gates",
            "detail": "VALIDATED candidate — promotion stays explicit via Release Review.",
            "endpoint": "release_review",
            "wired": True,
            "read_only": not approvable,
        }
    if source_id == "cms.provider_info" and "check_cms" in actions:
        return {
            "label": "Check CMS for newer Provider Information",
            "detail": "Compare publisher vintage against local raw/processed artifacts.",
            "endpoint": "action_pi_check",
            "wired": True,
            "method": "post",
        }
    if source_id == "cms.pbj_nurse_staffing" and "check_cms" in actions:
        return {
            "label": "Check CMS for newer nurse quarter",
            "detail": "Compare CMS Primary quarter against local nurse CSVs.",
            "endpoint": "action_nurse_check",
            "wired": True,
            "method": "post",
        }
    if active_id and not pending:
        return {
            "label": "Monitor release health",
            "detail": f"ACTIVE {active_id} — no pending candidate. Refresh health from Sources.",
            "endpoint": "sources",
            "wired": True,
            "read_only": True,
        }
    if snapshot and snapshot.get("status") == OpsStatus.PROCESSING_REQUIRED.value:
        return {
            "label": "Complete local processing",
            "detail": snapshot.get("detail") or "Raw present; processed output missing.",
            "endpoint": "source_detail",
            "endpoint_args": {"source_id": source_id},
            "wired": True,
            "read_only": True,
        }
    return {
        "label": "Inspect dataset diagnostics",
        "detail": "No automated next action is wired for this dataset stage.",
        "endpoint": "source_detail",
        "endpoint_args": {"source_id": source_id},
        "wired": True,
        "read_only": True,
    }


def build_sff_lifecycle_steps(
    *,
    control_row: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Read-only SFF lifecycle sequence for operator detail (pilot)."""
    active = (control_row or {}).get("active") or {}
    pending = (control_row or {}).get("pending") or {}
    active_id = active.get("active_release_id")
    pending_state = str(pending.get("state") or "").upper()
    pending_id = pending.get("release_id")
    validation = _validation_status_from_control(control_row)
    validated = validation == "PASS" or pending_state == "VALIDATED"

    def step(
        step_id: str,
        label: str,
        *,
        state: str,
        detail: str,
        action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": step_id,
            "label": label,
            "state": state,
            "detail": detail,
            "action": action,
        }

    steps: list[dict[str, Any]] = []
    steps.append(
        step(
            "check_cms",
            "Check CMS",
            state="not_wired",
            detail="No SFF posting check action is wired in Data Ops UI yet.",
        )
    )

    acquire_state = "completed" if active_id or pending_id else "upcoming"
    if pending_state in {"ACQUIRED", "VALIDATED"} and not active_id:
        acquire_state = "completed"
    steps.append(
        step(
            "acquire_pdf",
            "Acquire PDF",
            state=acquire_state,
            detail="Acquisition uses sff_release.stage_pdf (CLI) — not exposed as a UI button in this pass.",
        )
    )

    validate_state = "completed" if validated or active_id else ("current" if pending_state == "ACQUIRED" else "upcoming")
    steps.append(
        step(
            "validate",
            "Validate",
            state=validate_state,
            detail=f"Structural validation {'PASS' if (validated or active_id) else 'pending'} via sff_release.validate_rows.",
        )
    )

    if active_id:
        review_state = "completed"
        promote_state = "completed"
    elif pending_state == "VALIDATED":
        review_state = "completed"
        promote_state = "current"
    elif pending_state == "ACQUIRED":
        review_state = "current"
        promote_state = "blocked"
    else:
        review_state = "upcoming"
        promote_state = "upcoming"

    steps.append(
        step(
            "review",
            "Review",
            state=review_state,
            detail="Human review of pending candidate before promotion.",
            action={
                "label": "Open Release Review",
                "endpoint": "release_review",
                "wired": True,
                "read_only": True,
            }
            if pending_state in {"ACQUIRED", "VALIDATED"}
            else None,
        )
    )
    steps.append(
        step(
            "make_active",
            "Make ACTIVE",
            state=promote_state,
            detail=(
                f"ACTIVE {active_id} — promotion is explicit and already recorded."
                if active_id
                else "Explicit promotion only — no auto-promote from this page."
            ),
        )
    )
    steps.append(
        step(
            "pbj_build",
            "PBJ build",
            state="not_wired" if active_id else "upcoming",
            detail="Facility package refresh/build is separate from SFF ingest.",
            action={
                "label": "Dashboard Builder",
                "endpoint": "dashboard_builder",
                "wired": True,
                "read_only": True,
            },
        )
    )
    steps.append(
        step(
            "public_staging",
            "Public staging",
            state="not_wired",
            detail="NOT YET WIRED — no public staging path from Data Ops.",
        )
    )
    steps.append(
        step(
            "publish",
            "Publish",
            state="not_wired",
            detail="NOT YET WIRED — publication remains explicit and separate from ACTIVE.",
        )
    )
    return steps


def build_source_operator_workflow(
    source_id: str,
    *,
    record: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
    control_row: dict[str, Any] | None,
) -> dict[str, Any]:
    """Operator-facing workflow summary from existing control plane + probe overlay."""
    from release_control_plane import CAPABILITY_LABELS

    active = (control_row or {}).get("active") or {}
    pending = (control_row or {}).get("pending") or {}
    impact = (control_row or {}).get("impact") or {}
    downstream = [
        CAPABILITY_LABELS.get(item, item)
        for item in (impact.get("would_mark_stale") or [])
    ]
    zweli_status = None
    if snapshot:
        zweli_status = snapshot.get("zweli_status")
    elif isinstance(pending.get("metadata"), dict):
        zweli_status = pending.get("metadata", {}).get("zweli_status")

    workflow = {
        "source_id": source_id,
        "active_release_id": active.get("active_release_id"),
        "active_status": active.get("status") or ("ACTIVE" if active.get("active_release_id") else None),
        "pending_release_id": pending.get("release_id"),
        "pending_state": pending.get("state"),
        "validation_status": _validation_status_from_control(control_row)
        or (snapshot or {}).get("validation_status"),
        "health": (control_row or {}).get("health"),
        "health_detail": (control_row or {}).get("health_detail"),
        "zweli_status": zweli_status,
        "downstream_capabilities": downstream,
        "next_action": _next_operator_action(
            source_id, record=record, snapshot=snapshot, control_row=control_row
        ),
        "lifecycle_steps": None,
    }
    if source_id == "cms.sff_pdf_list":
        workflow["lifecycle_steps"] = build_sff_lifecycle_steps(control_row=control_row)
    return workflow


def check_provider_info_cms(
    *,
    fetch_json: FetchJson | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    import cms_provider_info_acquire as acq

    root = root or cms_data_paths.repo_root()
    cms = acq.resolve_cms_provider_info_release(fetch_json=fetch_json)
    newer = acq.cms_is_newer_than_local(cms, root)
    local = acq.latest_local_provider_info(root)
    snap = probe_source("cms.provider_info", check_cms=True, fetch_json=fetch_json, root=root)
    if newer:
        from release_control_plane import ReleaseState, control_plane_root, record_candidate

        label = str(cms.data_vintage_label or "").strip()
        release_match = re.fullmatch(r"([A-Za-z]{3,9})\s+(20\d{2})", label)
        if not release_match:
            raise RuntimeError(f"ambiguous Provider Information release identity: {label!r}")
        month = _MONTH_NAME_TO_NUM.get(release_match.group(1).lower()[:3])
        if not month:
            raise RuntimeError(f"unrecognized Provider Information month: {label!r}")
        canonical_release_id = f"{release_match.group(2)}-{month:02d}"

        record_candidate(
            "cms.provider_info", canonical_release_id, ReleaseState.DETECTED,
            metadata={"publisher_label": label, "publisher_modified": cms.modified, "publisher_released": cms.released},
            root=control_plane_root(root),
        )
    return {
        "action": "check_cms",
        "cms": {
            "dataset_id": cms.dataset_id,
            "data_vintage_label": cms.data_vintage_label,
            "distribution_filename": cms.distribution_filename,
            "released": cms.released,
            "modified": cms.modified,
            "next_update_date": cms.next_update_date,
        },
        "local": (
            {"year": local[0], "month": local[1], "path": str(local[2])} if local else None
        ),
        "cms_is_newer": newer,
        "dry_run": acq.acquire_and_process(root=root, fetch_json=fetch_json, dry_run=True),
        "snapshot": snap.to_dict(),
    }


def acquire_provider_info(
    *,
    dry_run: bool = False,
    force: bool = False,
    fetch_json: FetchJson | None = None,
    fetch_bytes: Callable[[str], bytes] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    import cms_provider_info_acquire as acq

    root = root or cms_data_paths.repo_root()
    report = acq.acquire_and_process(
        root=root,
        fetch_json=fetch_json,
        fetch_bytes=fetch_bytes,
        dry_run=dry_run,
        force=force,
    )
    snap = probe_source(
        "cms.provider_info",
        check_cms=not dry_run,
        fetch_json=fetch_json,
        root=root,
    )
    if not dry_run and snap.canonical_source_path and snap.release_id:
        from release_control_plane import ReleaseState, control_plane_root, record_candidate

        record_candidate(
            "cms.provider_info", snap.release_id, ReleaseState.ACQUIRED,
            source_path=snap.canonical_source_path,
            metadata={"structural_status": snap.structural_status, "zweli_status": snap.zweli_status},
            root=control_plane_root(root),
        )
    return {
        "action": "acquire_process",
        "acquire_report": report,
        "snapshot": snap.to_dict(),
    }


def check_nurse_cms(
    *,
    fetch_json: FetchJson | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    import cms_pbj_nurse_acquire as nurse_acq

    root = root or cms_data_paths.repo_root()
    cms = nurse_acq.resolve_cms_nurse_release(fetch_json=fetch_json)
    newer = nurse_acq.cms_is_newer_than_local(cms, root)
    local = nurse_acq.latest_local_nurse(root)
    snap = probe_source(
        "cms.pbj_nurse_staffing", check_cms=True, fetch_json=fetch_json, root=root
    )
    dry = nurse_acq.acquire_and_process(root=root, fetch_json=fetch_json, dry_run=True)
    if newer:
        from release_control_plane import ReleaseState, control_plane_root, record_candidate

        record_candidate(
            "cms.pbj_nurse_staffing", cms.quarter_label, ReleaseState.DETECTED,
            metadata={"publisher_dataset_id": cms.dataset_id},
            root=control_plane_root(root),
        )
    return {
        "action": "check_cms",
        "cms": {
            "dataset_id": cms.dataset_id,
            "quarter_label": cms.quarter_label,
            "distribution_filename": cms.distribution_filename,
            "distribution_url": cms.distribution_url,
            "title": cms.title,
        },
        "local": (
            {"year": local[0], "quarter": local[1], "path": str(local[2])} if local else None
        ),
        "cms_is_newer": newer,
        "dry_run": dry,
        "snapshot": snap.to_dict(),
    }


def acquire_nurse(
    *,
    dry_run: bool = False,
    force: bool = False,
    fetch_json: FetchJson | None = None,
    fetch_bytes: Callable[[str], bytes] | None = None,
    root: Path | None = None,
    skip_standardize: bool = False,
) -> dict[str, Any]:
    import cms_pbj_nurse_acquire as nurse_acq

    root = root or cms_data_paths.repo_root()
    report = nurse_acq.acquire_and_process(
        root=root,
        fetch_json=fetch_json,
        fetch_bytes=fetch_bytes,
        dry_run=dry_run,
        force=force,
        skip_standardize=skip_standardize,
    )
    snap = probe_source(
        "cms.pbj_nurse_staffing",
        check_cms=not dry_run,
        fetch_json=fetch_json,
        root=root,
    )
    if not dry_run and snap.canonical_source_path and snap.release_id:
        from release_control_plane import ReleaseState, control_plane_root, record_candidate

        record_candidate(
            "cms.pbj_nurse_staffing", snap.release_id, ReleaseState.ACQUIRED,
            source_path=snap.canonical_source_path,
            metadata={"structural_status": snap.structural_status},
            root=control_plane_root(root),
        )
    return {
        "action": "acquire_process",
        "acquire_report": report,
        "snapshot": snap.to_dict(),
    }


def recommended_next_automation() -> dict[str, str]:
    return {
        "source_id": "cms.pbj_non_nurse_staffing",
        "human_name": "PBJ non-nurse staffing",
        "why": (
            "Same quarterly CMS pattern as nurse; detection + standardize exist, but "
            "acquire CLI is broken-legacy (missing ingest_cms_nonnurse_quarter.py)."
        ),
        "first_broken_layer": (
            "acquisition — manage_cms_sources nonnurse ingest references a missing "
            "script; repair/replace with data-api resources acquire like nurse"
        ),
    }


def derived_signals_payload() -> list[dict[str, Any]]:
    return [s.to_dict() for s in get_derived_signals()]


def load_zweli_report_for_release(
    source_id: str,
    release_id: str,
    *,
    root: Path | None = None,
) -> Optional[dict[str, Any]]:
    """Load stored Zweli report if present and matching source/release."""
    root = root or cms_data_paths.repo_root()
    if source_id != "cms.provider_info":
        return None
    path = (
        cms_data_paths.provider_release_manifest_dir(release_id, root) / "zweli_report.json"
    )
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("source_id") != source_id or data.get("release_id") != release_id:
        return None
    if not data.get("state"):
        return None
    return data


def provider_info_pbjapp_root(root: Path | None = None) -> Path:
    """PBJapp data root for canonical Provider Info / ownership artifacts."""
    if root is not None:
        return Path(root).resolve()
    configured = (os.environ.get("PBJ_REPO_ROOT") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    sibling = (_ROOT.parent / "PBJapp").resolve()
    if sibling.is_dir():
        return sibling
    return cms_data_paths.repo_root()


def _parse_provider_release_id(release_id: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"(\d{4})-(\d{2})", (release_id or "").strip())
    if not match:
        raise ValueError(f"invalid provider release_id: {release_id!r}")
    year, month = int(match.group(1)), int(match.group(2))
    if month < 1 or month > 12:
        raise ValueError(f"invalid provider release month: {release_id!r}")
    month_abbr = (
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    )[month - 1]
    return year, month, month_abbr


def build_provider_info_promotion_bundle(
    release_id: str,
    *,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Canonical ACTIVE shape: normalized primary + nh_ownership source_set."""
    from data_ops_approval import ApprovalError

    year, month, month_abbr = _parse_provider_release_id(release_id)
    base = provider_info_pbjapp_root(data_root)
    norm = (
        cms_data_paths.provider_info_normalized_dir(base)
        / f"ProviderInfoNorm_{year}_{month:02d}.csv"
    )
    ownership = cms_data_paths.ownership_dir(base) / f"NH_Ownership_{month_abbr}{year}.csv"
    manifest = (
        cms_data_paths.provider_release_manifest_dir(release_id, base)
        / "release_manifest.json"
    )
    if not norm.is_file() or norm.stat().st_size <= 0:
        raise ApprovalError(f"canonical normalized Provider Info missing: {norm}")
    if not ownership.is_file() or ownership.stat().st_size <= 0:
        raise ApprovalError(f"canonical NH Ownership missing: {ownership}")
    metadata: dict[str, Any] = {
        "source_set": [
            {"role": "provider_info", "source_path": str(norm)},
            {"role": "nh_ownership", "source_path": str(ownership)},
        ],
    }
    if manifest.is_file():
        metadata["validation_evidence"] = str(manifest)
    return {
        "source_path": norm,
        "source_hash": sha256_file(norm),
        "release_date": release_id,
        "metadata": metadata,
    }


def resolve_zweli_state_for_release(
    source_id: str,
    release_id: str,
    *,
    root: Path | None = None,
) -> ZweliState:
    """Server-authoritative Zweli state for approval (never trust the browser).

    Prefer the stored report for ``source_id``+``release_id``; otherwise recompute
    via canonical probe. NOT_RUN is returned when Zweli could not compare — it is
    optional evidence, not an activation blocker.
    """
    from data_ops_approval import ApprovalError

    root = root or cms_data_paths.repo_root()
    release_id = (release_id or "").strip()
    source_id = (source_id or "").strip()
    if not source_id or not release_id:
        raise ApprovalError("source_id and release_id are required for approval")

    stored = load_zweli_report_for_release(source_id, release_id, root=root)
    if stored:
        try:
            return ZweliState(stored["state"])
        except ValueError as exc:
            raise ApprovalError(
                f"Stored Zweli report has invalid state for {source_id} {release_id}"
            ) from exc

    snap = probe_source(source_id, check_cms=False, root=root, run_zweli=True)
    if snap.release_id != release_id:
        raise ApprovalError(
            f"Zweli report missing/mismatched for {source_id} {release_id} "
            f"(probe release={snap.release_id!r}) — fail closed"
        )
    if snap.zweli_report:
        if snap.zweli_report.get("source_id") not in (None, source_id):
            raise ApprovalError("Zweli report source_id mismatch — fail closed")
        if snap.zweli_report.get("release_id") not in (None, release_id):
            raise ApprovalError("Zweli report release_id mismatch — fail closed")
    try:
        return ZweliState(snap.zweli_status or ZweliState.NOT_RUN.value)
    except ValueError as exc:
        raise ApprovalError(f"Invalid Zweli state {snap.zweli_status!r}") from exc


def approve_release_authoritative(
    source_id: str,
    release_id: str,
    *,
    note: str = "",
    root: Path | None = None,
    audit_path: Path | None = None,
) -> Any:
    """Approve using server-resolved Zweli state only (form status ignored)."""
    from data_ops_approval import ApprovalError, approve_release
    from release_control_plane import ReleaseState, control_plane_root, promote_candidate, record_candidate

    state = resolve_zweli_state_for_release(source_id, release_id, root=root)
    snap = probe_source(source_id, check_cms=False, root=root, run_zweli=False)
    if snap.release_id != release_id:
        raise ApprovalError(
            f"release {source_id} {release_id} has no matching canonical local source"
        )
    if (snap.structural_status or "").upper() not in {"PASS", "UNKNOWN"}:
        raise ApprovalError(
            f"structural validation must pass before approval ({snap.structural_status})"
        )
    if source_id == "cms.provider_info":
        bundle = build_provider_info_promotion_bundle(release_id)
        source_path = bundle["source_path"]
        promotion_metadata = dict(bundle["metadata"])
        promotion_metadata["zweli_state"] = state.value
        promotion_metadata["approval_note"] = note
    else:
        if not snap.canonical_source_path:
            raise ApprovalError(
                f"release {source_id} {release_id} has no matching canonical local source"
            )
        source_path = snap.canonical_source_path
        promotion_metadata = {"zweli_state": state.value, "approval_note": note}
    entry = approve_release(
        source_id,
        release_id,
        zweli_state=state,
        note=note,
        audit_path=audit_path,
    )
    control_root = control_plane_root(root)
    record_candidate(
        source_id,
        release_id,
        ReleaseState.VALIDATED,
        source_path=source_path,
        validation={"status": "PASS", "validated_at": entry.timestamp},
        metadata=promotion_metadata,
        root=control_root,
    )
    promote_candidate(source_id, root=control_root)
    return entry


def prepare_provider_info_validated_candidate(
    release_id: str,
    *,
    root: Path | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Validate canonical August/July-style bundle and record VALIDATED (no ACTIVE)."""
    from data_ops_approval import ApprovalError
    from release_control_plane import ReleaseState, control_plane_root, record_candidate

    import cms_provider_info_acquire as acq

    control_root = control_plane_root(root)
    pbj_root = provider_info_pbjapp_root(data_root)
    year, month, month_abbr = _parse_provider_release_id(release_id)
    raw = cms_data_paths.provider_info_dir(pbj_root) / f"NH_ProviderInfo_{month_abbr}{year}.csv"
    if not raw.is_file():
        raise ApprovalError(f"raw Provider Info missing in PBJapp: {raw}")
    prior_path = None
    if month > 1:
        py, pm, pabbr = year, month - 1, None
        pabbr = _parse_provider_release_id(f"{py:04d}-{pm:02d}")[2]
        prior_path = cms_data_paths.provider_info_dir(pbj_root) / f"NH_ProviderInfo_{pabbr}{py}.csv"
        if not prior_path.is_file():
            prior_path = None
    elif year > 2000:
        py, pm = year - 1, 12
        pabbr = _parse_provider_release_id(f"{py:04d}-{pm:02d}")[2]
        prior_path = cms_data_paths.provider_info_dir(pbj_root) / f"NH_ProviderInfo_{pabbr}{py}.csv"
        if not prior_path.is_file():
            prior_path = None
    validation = acq.validate_raw_provider_info_csv(raw, prior_path=prior_path)
    if validation.get("row_count", 0) < 1000:
        raise ApprovalError("structural validation failed: row count too low")
    bundle = build_provider_info_promotion_bundle(release_id, data_root=pbj_root)
    zweli_state = resolve_zweli_state_for_release(
        "cms.provider_info", release_id, root=control_root
    )
    metadata = dict(bundle["metadata"])
    metadata["structural_status"] = "PASS"
    metadata["zweli_status"] = zweli_state.value
    metadata["acquisition_manifest"] = (
        f"provider_info/_manifests/{release_id}/release_manifest.json"
    )
    candidate = record_candidate(
        "cms.provider_info",
        release_id,
        ReleaseState.VALIDATED,
        source_path=bundle["source_path"],
        validation={"status": "PASS", "validated_at": datetime.now(timezone.utc).isoformat()},
        metadata=metadata,
        root=control_root,
    )
    impact = __import__("release_control_plane", fromlist=["what_would_change"]).what_would_change(
        "cms.provider_info"
    )
    return {
        "candidate": candidate,
        "validation": validation,
        "bundle": {
            "source_path": str(bundle["source_path"]),
            "source_hash": bundle["source_hash"],
            "source_set": metadata.get("source_set"),
        },
        "zweli_status": zweli_state.value,
        "downstream_impact": impact,
    }
