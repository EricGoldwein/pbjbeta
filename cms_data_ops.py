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
from active_release_registry import get_active_release, registry_path, sha256_file  # noqa: E402
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


def _probe_health_citations(
    record: CmsSourceRecord,
    root: Path,
    *,
    check_cms: bool = False,
    fetch_json: FetchJson | None = None,
    theme_publication: Any | None = None,
) -> SourceOpsSnapshot:
    snap = _base_snap(record)
    cit = cms_data_paths.citations_dir(root)
    candidates = list(cit.glob("NH_HealthCitations_*.csv")) if cit.is_dir() else []

    def _citation_key(path: Path) -> tuple[int, int]:
        m = re.search(r"_([A-Za-z]{3})(\d{4})\.csv$", path.name)
        return (int(m.group(2)), _MONTH_NAME_TO_NUM.get(m.group(1).lower(), 0)) if m else (0, 0)

    standalone = max(candidates, key=_citation_key) if candidates else None
    if standalone:
        release_id = _release_id_from_citation_basename(standalone.name)
        _apply_raw_ref(
            snap, local_file_ref("citations", standalone, release_id=release_id or standalone.name)
        )
        snap.pbjapp_latest = format_release_month_label(release_id) or standalone.name
        snap.release_id = release_id or standalone.name
        snap.local_raw_present = True
        snap.status = OpsStatus.LOCAL_RAW_ONLY.value
        snap.detail = "Citation CSV cached locally (immutable evidence after acquisition)."
        snap.last_successful_local_processing = _mtime_iso(standalone)
    else:
        _apply_raw_ref(
            snap, local_file_ref("citations", cit / "(none)", expected_elsewhere=True)
        )
        snap.status = OpsStatus.NOT_AVAILABLE_IN_THIS_RUNTIME.value
        snap.detail = "Health Citations dataset not accessible in this runtime"
    if check_cms:
        if theme_publication is None:
            try:
                from cms_theme_publication import get_latest_nh_theme_publication

                theme_publication = get_latest_nh_theme_publication(fetch_json=fetch_json)
            except Exception:
                theme_publication = None
        try:
            from health_citations_acquire import check_health_citations_cms

            check = check_health_citations_cms(
                fetch_json=fetch_json,
                root=root,
                theme_publication=theme_publication,
            )
            cms = check["cms"]
            snap.publisher_latest = cms["data_vintage_label"]
            snap.cms_latest = cms["data_vintage_label"]
            if check.get("bundle_provenance_ok"):
                snap.detail = check.get("bundle_detail") or snap.detail
            if check.get("cms_is_newer"):
                snap.status = OpsStatus.CMS_NEWER.value
            elif snap.local_raw_present and check.get("local_artifact_ready"):
                snap.status = OpsStatus.LOCAL_RAW_ONLY.value
            elif check.get("active_release_id") and not check.get("cms_is_newer"):
                snap.status = OpsStatus.CURRENT.value
        except Exception as exc:  # noqa: BLE001
            if theme_publication is not None:
                from active_release_registry import get_active_release, registry_path
                from cms_theme_publication import publication_availability_for_source

                active_release_id = (get_active_release("cms.health_citations", registry_path()) or {}).get(
                    "active_release_id"
                )
                theme_fields = publication_availability_for_source(
                    "cms.health_citations",
                    active_release_id=active_release_id,
                    publication=theme_publication,
                ) or {}
                pub_id = theme_fields.get("publisher_latest_release_id")
                if pub_id:
                    label = format_release_month_label(pub_id) or pub_id
                    snap.publisher_latest = label
                    snap.cms_latest = label
                    if active_release_id == pub_id:
                        snap.status = OpsStatus.CURRENT.value
                        snap.detail = "CMS latest matches ACTIVE release (theme publication discovery)."
                else:
                    snap.error = str(exc)
                    snap.detail = f"CMS Health Citations probe failed: {exc}"
            else:
                snap.error = str(exc)
                snap.detail = f"CMS Health Citations probe failed: {exc}"
    return snap


def _probe_sff_pdf_list(
    record: CmsSourceRecord,
    root: Path,
    *,
    check_cms: bool = False,
) -> SourceOpsSnapshot:
    snap = _base_snap(record)
    control_root = Path(__file__).resolve().parent
    active = get_active_release("cms.sff_pdf_list", registry_path(control_root)) or {}
    active_id = active.get("active_release_id")
    if active_id:
        staged_pdf = control_root / "sff" / "releases" / active_id / f"cms_sff_posting_{active_id}.pdf"
        if staged_pdf.is_file():
            _apply_raw_ref(snap, local_file_ref("sff", staged_pdf, release_id=active_id))
            snap.local_raw_present = True
            snap.release_id = active_id
            snap.pbjapp_latest = format_release_month_label(active_id) or active_id
            snap.status = OpsStatus.LOCAL_RAW_ONLY.value
            snap.detail = "Staged SFF posting PDF present for ACTIVE release."
            snap.last_successful_local_processing = _mtime_iso(staged_pdf)
        else:
            snap.status = OpsStatus.NOT_AVAILABLE_IN_THIS_RUNTIME.value
            snap.detail = f"ACTIVE {active_id} — staged PDF not present in this runtime."
    else:
        snap.status = OpsStatus.NOT_AVAILABLE_IN_THIS_RUNTIME.value
        snap.detail = "No ACTIVE SFF posting release."
    if check_cms:
        try:
            from sff_release import check_sff_cms

            check = check_sff_cms(root=control_root, record_detection=False)
            cms = check["cms"]
            snap.publisher_latest = cms.get("posting_label")
            snap.cms_latest = cms.get("posting_label")
            if check.get("cms_is_newer"):
                snap.status = OpsStatus.CMS_NEWER.value
                snap.detail = (
                    f"CMS posting {cms.get('posting_label')} is newer than ACTIVE {active_id or '—'}."
                )
            elif active_id and not check.get("cms_is_newer"):
                snap.status = OpsStatus.CURRENT.value
                snap.detail = f"CMS posting matches ACTIVE {format_release_month_label(active_id) or active_id}."
        except Exception as exc:  # noqa: BLE001 — probe must not crash callers
            snap.error = str(exc)
            snap.detail = f"CMS SFF posting probe failed: {exc}"
    return snap


def probe_source(
    source_id: str,
    *,
    check_cms: bool = True,
    fetch_json: FetchJson | None = None,
    root: Path | None = None,
    run_zweli: bool = True,
    theme_publication: Any | None = None,
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
        return _probe_health_citations(
            record,
            root,
            check_cms=check_cms,
            fetch_json=fetch_json,
            theme_publication=theme_publication,
        )
    if family == SourceFamily.SFF_PDF_LIST:
        return _probe_sff_pdf_list(record, root, check_cms=check_cms)
    if family == SourceFamily.SURVEY_SUMMARY:
        from survey_summary import probe_snapshot
        return probe_snapshot(_base_snap(record))
    return _probe_unmodeled(record, "Unhandled source family")


def probe_all_sources(
    *,
    check_cms: bool = True,
    fetch_json: FetchJson | None = None,
    root: Path | None = None,
    run_zweli: bool = True,
    theme_publication: Any | None = None,
) -> list[SourceOpsSnapshot]:
    return [
        probe_source(
            r.source_id,
            check_cms=check_cms,
            fetch_json=fetch_json,
            root=root,
            run_zweli=run_zweli,
            theme_publication=theme_publication,
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
            pending_same = (
                pending
                and str(pending.get("release_id") or "") == str(active.get("release_id") or "")
                and str(pending.get("state") or "").upper() == "ACQUIRED"
            )
            if pending_same:
                data["status"] = "PENDING VALIDATION"
            else:
                data["status"] = OpsStatus.CURRENT.value

    if pending:
        data["pending_release_id"] = pending.get("release_id")
        data["pending_release_state"] = pending.get("state")
        if pending.get("validation_status") is not None:
            data["validation_status"] = pending.get("validation_status")
        if pending.get("zweli_status") is not None and not active:
            data["zweli_status"] = pending.get("zweli_status")
            data["quality_reviewed"] = pending.get("zweli_status")

    pending_same_active = (
        pending
        and active
        and str(pending.get("release_id") or "") == str(active.get("release_id") or "")
        and str(pending.get("state") or "").upper() == "ACQUIRED"
    )
    if pending_same_active:
        data["display_status"] = "PENDING VALIDATION"
    else:
        data["display_status"] = (
            data.get("active_release_status")
            if active and data.get("active_release_status")
            else data.get("status")
        )
    rec = get_source(source_id)
    zweli_applicable = bool(rec and rec.zweli_quality_profile)
    data["zweli_applicable"] = zweli_applicable
    if not zweli_applicable:
        data["zweli_status"] = None
        data["quality_reviewed"] = None
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
    from release_review_policy import zweli_applies

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
    zweli_applicable = zweli_applies(s.source_id)
    return {
        "reason": reason,
        "source_id": s.source_id,
        "human_name": s.human_name,
        "release_id": release_id,
        "release_label": format_release_month_label(release_id) if release_id else None,
        "human_state": reason.replace("_", " ").title(),
        "primary_action_label": "Approve" if reason == "ready_for_approval" else "Review",
        "primary_action_kind": "activate" if reason == "ready_for_approval" else "review",
        "status": s.status,
        "zweli_applicable": zweli_applicable,
        "zweli_status": s.zweli_status if zweli_applicable else None,
        "structural_status": s.structural_status,
        "detail": s.detail,
        "evidence_lines": [s.detail] if s.detail else [],
        "zweli_report": s.zweli_report if zweli_applicable else None,
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
    active_release_id: str | None = None,
    theme_publication: Any | None = None,
) -> Optional[dict[str, Any]]:
    from release_review_policy import evaluate_governed_candidate_review

    snap_dict = snap.to_dict() if snap is not None else None
    return evaluate_governed_candidate_review(
        candidate,
        snap=snap_dict,
        active_release_id=active_release_id,
        root=root,
        theme_publication=theme_publication,
    )


def release_review_items(
    snapshots: list[SourceOpsSnapshot] | None = None,
    *,
    check_cms: bool = True,
    root: Path | None = None,
    control: dict[str, Any] | None = None,
    theme_publication: Any | None = None,
    focus_source_id: str | None = None,
    focus_release_id: str | None = None,
) -> list[dict[str, Any]]:
    """Items needing genuine human review decisions for Release Review UI."""
    from release_review_policy import (
        PAIR_SOURCE_ID,
        build_ownership_pair_review_item,
        filter_release_review_focus,
        zweli_applies,
    )
    from ownership_pairing import ENROLLMENTS, OWNERS

    root = root or cms_data_paths.repo_root()
    if control is None:
        from release_control_plane import control_panel_payload

        control = control_panel_payload(Path(__file__).resolve().parent)
    payload = _control_plane_ui_view(control)
    snaps = snapshots or probe_all_sources(check_cms=check_cms, root=root)
    by_id = {s.source_id: s for s in snaps}
    active_by_source = payload.get("active") or {}
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    pair_item = build_ownership_pair_review_item(root=root, control=control)
    if pair_item:
        key = (pair_item["source_id"], str(pair_item.get("release_id") or ""))
        seen.add(key)
        items.append(pair_item)

    for row in control.get("datasets") or []:
        dataset_id = row.get("dataset_id")
        pending = row.get("pending")
        if not dataset_id or not pending:
            continue
        candidate = {
            "source_id": dataset_id,
            "release_id": pending.get("release_id"),
            "state": pending.get("state"),
            "validation_status": (pending.get("validation") or {}).get("status"),
            "zweli_status": pending.get("zweli_status"),
            "validation": pending.get("validation"),
            "metadata": pending.get("metadata"),
        }
        if dataset_id in {OWNERS, ENROLLMENTS} and pair_item:
            continue
        active_release_id = ((row.get("active") or {}).get("active_release_id"))
        snap = by_id.get(dataset_id)
        item = _governed_candidate_review_item(
            candidate,
            snap=snap,
            root=root,
            active_release_id=active_release_id,
            theme_publication=theme_publication,
        )
        if not item:
            continue
        key = (item["source_id"], str(item.get("release_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        items.append(item)

    from cms_source_registry import is_review_only_source

    governed_sources = {item["source_id"] for item in items if item.get("governed")}
    for s in snaps:
        if is_review_only_source(s.source_id):
            continue  # Reference candidates have no legacy activation path.
        if s.source_id in governed_sources:
            continue
        if s.source_id in {OWNERS, ENROLLMENTS} and pair_item:
            continue
        active = active_by_source.get(s.source_id) or {}
        if active.get("status") == "ACTIVE" and not (payload.get("pending_by_source") or {}).get(
            s.source_id
        ):
            continue
        item = _legacy_release_review_item(s)
        if not item:
            continue
        if not item.get("zweli_applicable", True) and item.get("zweli_status") == ZweliState.NOT_RUN.value:
            if item.get("reason") not in {"structural_error", "processing_failure"}:
                continue
        key = (item["source_id"], str(item.get("release_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        items.append(item)

    items.sort(
        key=lambda item: (
            0 if item.get("approvable") else 1,
            0 if item.get("source_id") == PAIR_SOURCE_ID else 1,
            item.get("human_name") or item.get("source_id") or "",
        )
    )
    return filter_release_review_focus(
        items,
        source_id=focus_source_id,
        release_id=focus_release_id,
    )


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


def format_release_month_label(release_id: str | None) -> str | None:
    """Turn ``2026-08`` into ``Aug 2026`` for operator-facing copy."""
    if not release_id:
        return None
    match = re.fullmatch(r"(\d{4})-(\d{2})", str(release_id).strip())
    if not match:
        return str(release_id)
    year, month = int(match.group(1)), int(match.group(2))
    if month < 1 or month > 12:
        return str(release_id)
    abbr = (
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    )[month - 1]
    return f"{abbr} {year}"


def _release_id_from_citation_basename(name: str) -> str | None:
    match = re.search(r"_([A-Za-z]{3})(\d{4})\.csv$", name or "", re.I)
    if not match:
        return None
    month = _MONTH_NAME_TO_NUM.get(match.group(1).lower()[:3])
    if not month:
        return None
    return f"{int(match.group(2)):04d}-{month:02d}"


def build_release_availability_context(
    source_id: str,
    *,
    control_row: dict[str, Any] | None,
    check_row: dict[str, Any] | None = None,
    snapshot: dict[str, Any] | None = None,
    record: dict[str, Any] | None = None,
    root: Path | None = None,
    theme_publication: Any | None = None,
) -> dict[str, Any]:
    """Reusable operator copy: ACTIVE vs publisher/latest vs pending."""
    from cms_source_registry import THEME_PUBLICATION_SOURCES
    from cms_theme_publication import publication_availability_for_source
    from release_source_catalog import SOURCES, UpdateMechanism

    root = root or cms_data_paths.repo_root()
    active = (control_row or {}).get("active") or {}
    pending = (control_row or {}).get("pending") or {}
    active_id = active.get("active_release_id")
    pending_id = pending.get("release_id")
    pending_state = pending.get("state")
    catalog = next((item for item in SOURCES if item.dataset_id == source_id), None)

    publisher_latest_id: str | None = None
    new_available = False
    availability_source = "none"
    theme_fields: dict[str, Any] = {}

    if source_id in THEME_PUBLICATION_SOURCES and theme_publication is not None:
        theme_fields = publication_availability_for_source(
            source_id,
            active_release_id=active_id,
            publication=theme_publication,
        ) or {}
        if theme_fields:
            publisher_latest_id = theme_fields.get("publisher_latest_release_id")
            new_available = bool(theme_fields.get("new_release_available"))
            availability_source = theme_fields.get("availability_source") or "theme_publication"

    if availability_source == "none" and catalog and catalog.mechanism == UpdateMechanism.DERIVED:
        from release_check import derived_state

        derived = derived_state(source_id, catalog.upstream, root=root)
        new_available = bool(derived.get("new_release_available"))
        upstream_active = derived.get("upstream_active") or {}
        if catalog.upstream:
            publisher_latest_id = upstream_active.get(catalog.upstream[0])
        availability_source = "derived_upstream"
        theme_fields = {**theme_fields, "derived_upstream_status": derived.get("status")}
    elif availability_source == "none" and check_row is not None:
        new_available = check_row.get("new_release_available") is True
        publisher_latest_id = check_row.get("release_id") or check_row.get("pending_release")
        availability_source = "release_check"

    if active_id and publisher_latest_id and str(active_id) == str(publisher_latest_id):
        new_available = False
    if (
        pending_id
        and pending_state not in (None, "ACTIVE")
        and active_id
        and str(pending_id) == str(active_id)
    ):
        new_available = False
    if publisher_latest_id is None and snapshot:
        pub = snapshot.get("publisher_latest") or snapshot.get("cms_latest")
        if pub:
            label_match = re.fullmatch(r"([A-Za-z]{3,9})\s+(20\d{2})", str(pub).strip())
            if label_match:
                month = _MONTH_NAME_TO_NUM.get(label_match.group(1).lower()[:3])
                if month:
                    publisher_latest_id = f"{label_match.group(2)}-{month:02d}"
            else:
                publisher_latest_id = str(pub)
            if availability_source == "none":
                availability_source = "cms_probe"

    local_release_id = None
    if snapshot:
        raw_release = snapshot.get("release_id")
        if raw_release and re.fullmatch(r"\d{4}-\d{2}", str(raw_release)):
            local_release_id = str(raw_release)
        elif raw_release:
            local_release_id = _release_id_from_citation_basename(str(raw_release))

    active_label = format_release_month_label(active_id)
    publisher_label = format_release_month_label(publisher_latest_id)
    if publisher_label is None and publisher_latest_id:
        publisher_label = str(publisher_latest_id)

    summary = "Current"
    if new_available and active_id and publisher_latest_id and active_id != publisher_latest_id:
        summary = "New release available"
    elif new_available and not active_id:
        summary = "Release missing"
    elif pending_id and pending_state not in (None, "ACTIVE"):
        if active_id and str(pending_id) == str(active_id) and pending_state == "ACQUIRED":
            summary = "Pending validation"
        else:
            summary = f"Pending {pending_state.lower()}"
    elif active_id and publisher_latest_id and str(active_id) == str(publisher_latest_id):
        summary = "Current"

    bundle_provenance_ok = False
    local_artifact_ready = bool(local_release_id)
    if source_id == "cms.health_citations" and publisher_latest_id:
        target_release = publisher_latest_id
        local_artifact_ready = bool(
            (snapshot or {}).get("local_raw_present")
            or citations_artifact_exists(target_release, root=root)
            or (local_release_id and local_release_id >= target_release)
        )
        if local_artifact_ready:
            try:
                from health_citations_acquire import verify_bundle_provenance

                verify_bundle_provenance(target_release, root=root)
                bundle_provenance_ok = True
            except Exception:
                bundle_provenance_ok = False

    result = {
        "source_id": source_id,
        "human_name": (record or {}).get("human_name") or source_id,
        "active_release_id": active_id,
        "active_release_label": active_label,
        "publisher_latest_release_id": publisher_latest_id,
        "publisher_latest_label": publisher_label,
        "pending_release_id": pending_id,
        "pending_state": pending_state,
        "local_release_id": local_release_id,
        "local_artifact_ready": local_artifact_ready,
        "bundle_provenance_ok": bundle_provenance_ok,
        "new_release_available": new_available,
        "availability_summary": summary,
        "availability_source": availability_source,
        "cms_publication_id": theme_fields.get("cms_publication_id"),
        "cms_publication_date": theme_fields.get("cms_publication_date"),
        "processing_modified_date": theme_fields.get("processing_modified_date"),
        "product_release_id": theme_fields.get("product_release_id") or publisher_latest_id,
        "cms_dataset_id": theme_fields.get("cms_dataset_id"),
        "in_latest_publication": theme_fields.get("in_latest_publication"),
        "unchanged_in_latest_publication": theme_fields.get("unchanged_in_latest_publication")
        if theme_fields.get("unchanged_in_latest_publication") is not None
        else bool(
            active_id
            and publisher_latest_id
            and str(active_id) == str(publisher_latest_id)
            and not pending_id
        ),
    }
    if catalog and catalog.mechanism == UpdateMechanism.DERIVED:
        upstream_active = {}
        if availability_source == "derived_upstream":
            from release_check import derived_state

            derived = derived_state(source_id, catalog.upstream, root=root)
            upstream_active = derived.get("upstream_active") or {}
            result["derived_upstream_status"] = derived.get("status")
            result["provenance_missing"] = derived.get("provenance_missing")
        result["upstream_active"] = upstream_active
        from operator_freshness import inventory_fields_for_source

        result.update(inventory_fields_for_source(source_id, availability=result, check_row=check_row))
    elif catalog:
        from operator_freshness import inventory_fields_for_source

        result.update(inventory_fields_for_source(source_id, availability=result, check_row=check_row))
    return result


def citations_artifact_exists(release_id: str, *, root: Path | None = None) -> bool:
    from health_citations_acquire import citations_artifact_path

    return citations_artifact_path(release_id, root=root).is_file()


def build_needs_attention_queue(
    *,
    control: dict[str, Any],
    check_by_dataset: dict[str, dict[str, Any]],
    snapshots: list[dict[str, Any]] | None = None,
    root: Path | None = None,
    theme_publication: Any | None = None,
) -> list[dict[str, Any]]:
    """Primary operator queue: datasets with an actionable newer release."""
    from release_source_catalog import SOURCES, UpdateMechanism
    from release_control_plane import stale_derived_consumers

    snap_by_id = {item.get("source_id"): item for item in (snapshots or []) if item.get("source_id")}
    catalog_by_id = {item.dataset_id: item for item in SOURCES}
    stale_map = stale_derived_consumers(root)
    items: list[dict[str, Any]] = []
    for row in control.get("datasets") or []:
        dataset_id = row.get("dataset_id")
        if not dataset_id:
            continue
        if dataset_id.startswith("pbj.benchmarks.") or dataset_id == "pbj.peer_distribution":
            continue
        check_row = check_by_dataset.get(dataset_id) or {}
        record = get_source(dataset_id)
        record_dict = record.to_dict() if record else minimal_record_for_dataset(dataset_id)
        availability = build_release_availability_context(
            dataset_id,
            control_row=row,
            check_row=check_row,
            snapshot=snap_by_id.get(dataset_id),
            record=record_dict,
            root=root,
            theme_publication=theme_publication,
        )
        if dataset_id in {"cms.snf_all_owners", "cms.snf_enrollments"}:
            pending = row.get("pending")
            if not pending and not availability.get("new_release_available"):
                continue
        pending = row.get("pending")
        pending_state = str((pending or {}).get("state") or "").upper()
        health = str(row.get("health") or "").upper()
        from provenance_freshness import downstream_stale_capabilities_for_source

        stale_caps = downstream_stale_capabilities_for_source(dataset_id, root=root)
        downstream_stale = bool(stale_caps)
        workflow = build_source_operator_workflow(
            dataset_id,
            record=record_dict,
            snapshot=snap_by_id.get(dataset_id),
            control_row=row,
            release_availability=availability,
            theme_publication=theme_publication,
            root=root,
        )
        next_action = workflow.get("next_action") or {}
        needs = (
            availability.get("new_release_available")
            or pending
            or downstream_stale
            or health not in {"PASS", "CURRENT", ""}
            or pending_state in {"ACQUIRED", "VALIDATED", "DETECTED"}
        )
        if not needs:
            continue
        if dataset_id == "cms.health_citations":
            consumer_only_stale = (
                bool(stale_caps)
                and not availability.get("new_release_available")
                and not pending
                and set(stale_caps) <= {"facility.citations"}
            )
            if consumer_only_stale:
                continue
        if (
            not availability.get("new_release_available")
            and not pending
            and not downstream_stale
            and health in {"PASS", "CURRENT", ""}
            and next_action.get("label") in {"Monitor release health", "Inspect dataset diagnostics"}
        ):
            continue
        item_payload = {
            **availability,
            "health": row.get("health"),
            "downstream_stale": downstream_stale,
            "next_action": next_action,
            "mechanism": (catalog_by_id.get(dataset_id).mechanism.value if catalog_by_id.get(dataset_id) else None),
        }
        if dataset_id == "cms.pbj_nurse_staffing":
            from operator_freshness import audit_nurse_staffing_candidate_state

            item_payload["candidate_audit"] = audit_nurse_staffing_candidate_state(
                control_row=row,
                root=root,
            )
        items.append(item_payload)
    items.sort(
        key=lambda item: (
            0 if item.get("new_release_available") else 1,
            item.get("human_name") or item.get("source_id") or "",
        )
    )
    paired = _ownership_pair_attention_item(control, check_by_dataset, snap_by_id, root=root)
    downstream = _ownership_downstream_attention_item(root=root)
    citations = _citation_packages_attention_item(root=root)
    quarter_map = _provider_quarter_mapping_attention_item(root=root)
    prefix: list[dict[str, Any]] = []
    if paired:
        prefix.append(paired)
    if downstream:
        prefix.append(downstream)
    if citations:
        prefix.append(citations)
    if quarter_map:
        prefix.append(quarter_map)
    if prefix:
        skip = {"cms.snf_all_owners", "cms.snf_enrollments", "cms.health_citations"}
        if downstream:
            skip.add("ownership.downstream")
        if citations:
            skip.add("consumer.citation_packages")
        if quarter_map:
            skip.add("cms.provider_info.quarter_map")
        others = [i for i in items if i.get("source_id") not in skip]
        others.sort(
            key=lambda item: (
                0 if item.get("new_release_available") else 1,
                item.get("human_name") or item.get("source_id") or "",
            )
        )
        items = prefix + others
    return [_finalize_attention_item(item) for item in items]


def _ownership_pair_attention_item(
    control: dict[str, Any],
    check_by_dataset: dict[str, dict[str, Any]],
    snap_by_id: dict[str, dict[str, Any]],
    *,
    root: Path | None,
) -> dict[str, Any] | None:
    """Single queue row for SNF owners + enrollments when a pair needs review."""
    from ownership_pairing import ENROLLMENTS, OWNERS, PAIR_SOURCE_ID, pair_lifecycle_action, pairing_status

    pair = pairing_status(root)
    pending = pair.get("pending") or {}
    if not pending.get("owners_release") and not pending.get("enrollment_release"):
        return None
    owners_row = next((r for r in control.get("datasets") or [] if r.get("dataset_id") == OWNERS), None)
    enroll_row = next((r for r in control.get("datasets") or [] if r.get("dataset_id") == ENROLLMENTS), None)
    release_id = pending.get("owners_release") or pending.get("enrollment_release")
    from ownership_pairing import format_ownership_release_label

    release_label = format_ownership_release_label(release_id)
    owners_state = str(pending.get("owners_state") or "").upper()
    enrollment_state = str(pending.get("enrollment_state") or "").upper()
    action = pair_lifecycle_action(pair)
    candidate_states = [
        owners_state if source_id == OWNERS else enrollment_state
        for source_id in pair.get("candidate_source_ids") or []
    ]
    if candidate_states and all(state == "VALIDATED" for state in candidate_states):
        concise_state = f"{release_label} validated"
    elif owners_state == "ACQUIRED" or enrollment_state == "ACQUIRED":
        concise_state = f"{release_label} acquired"
    else:
        concise_state = f"{release_label} pending"
    next_action = {
        **action,
        "label": "Review candidate" if action.get("kind") == "activate_pair" else action.get("label"),
        "endpoint": "source_detail_panel",
        "endpoint_args": {"source_id": PAIR_SOURCE_ID},
        "page_endpoint": "source_detail",
        "page_endpoint_args": {"source_id": PAIR_SOURCE_ID},
        "opens_panel": True,
        "wired": True,
        "read_only": True,
    }
    return _finalize_attention_item(
        {
            "source_id": PAIR_SOURCE_ID,
            "human_name": "SNF All Owners (PECOS) / Enrollments",
            "active_release_id": (pair.get("active") or {}).get("owners_release"),
            "active_release_label": format_ownership_release_label((pair.get("active") or {}).get("owners_release")),
            "publisher_latest_release_id": None,
            "publisher_latest_label": None,
            "pending_release_id": release_id,
            "pending_state": owners_state or pending.get("enrollment_state"),
            "local_release_id": None,
            "new_release_available": pair.get("review_state") == "READY FOR REVIEW",
            "availability_summary": concise_state,
            "availability_source": "ownership_pairing",
            "health": (owners_row or enroll_row or {}).get("health"),
            "next_action": next_action,
            "mechanism": "external recurring release",
            "concise_state": concise_state,
            "release_line": release_label or str(release_id or "—"),
            "panel_source_id": PAIR_SOURCE_ID,
        }
    )


def _provider_quarter_mapping_attention_item(*, root: Path | None) -> dict[str, Any] | None:
    from provider_quarter_mapping import audit_active_provider_quarter_mapping

    audit = audit_active_provider_quarter_mapping(root=root)
    if not audit.get("needs_attention"):
        return None
    release_id = audit.get("release_id") or ""
    can_sync = bool(audit.get("can_sync"))
    return _finalize_attention_item(
        {
            "source_id": "cms.provider_info.quarter_map",
            "human_name": "Provider Information quarters",
            "active_release_id": release_id,
            "active_release_label": format_release_month_label(release_id) or release_id,
            "publisher_latest_release_id": release_id,
            "publisher_latest_label": format_release_month_label(release_id) or release_id,
            "pending_release_id": None,
            "pending_state": None,
            "local_release_id": None,
            "new_release_available": True,
            "availability_summary": "quarter map missing",
            "availability_source": "provider_quarter_mapping",
            "health": None,
            "downstream_stale": True,
            "next_action": {
                "label": audit.get("action_label") or "Apply extracted quarter map",
                "detail": audit.get("detail") or "",
                "endpoint": "action_provider_quarter_map_sync" if can_sync else "action_pi_check",
                "wired": True,
                "method": "post",
                "read_only": False,
            },
            "mechanism": "extracted interval → dashboard quarter map",
            "concise_state": "quarter map missing",
            "release_line": format_release_month_label(release_id) or str(release_id or "—"),
            "panel_source_id": "cms.provider_info",
        }
    )


def _citation_packages_attention_item(*, root: Path | None) -> dict[str, Any] | None:
    from operator_freshness import count_stale_citation_packages

    stale_count, checked = count_stale_citation_packages(root=root)
    if stale_count <= 0:
        return None
    return _finalize_attention_item(
        {
            "source_id": "consumer.citation_packages",
            "human_name": "Citation packages",
            "active_release_id": None,
            "active_release_label": None,
            "publisher_latest_release_id": None,
            "publisher_latest_label": None,
            "pending_release_id": None,
            "pending_state": None,
            "local_release_id": None,
            "new_release_available": False,
            "availability_summary": f"{stale_count} stale",
            "availability_source": "citation_package_gate",
            "health": None,
            "downstream_stale": True,
            "next_action": {
                "label": "Rebuild facility packages",
                "detail": (
                    f"{stale_count} of {checked} facility citation tables lag the ACTIVE national "
                    "Health Citations file — refresh local facility slices (no CMS re-download or deploy)."
                ),
                "endpoint": "action_citation_packages_rebuild",
                "wired": True,
                "method": "post",
                "read_only": False,
            },
            "mechanism": "consumer packaging",
            "concise_state": f"{stale_count} stale",
            "release_line": f"{checked} checked",
            "panel_source_id": "cms.health_citations",
        }
    )


def _ownership_downstream_attention_item(*, root: Path | None) -> dict[str, Any] | None:
    from ownership_downstream_rebuild import (
        OWNERSHIP_DOWNSTREAM_SOURCE_ID,
        OwnershipRebuildError,
        audit_ownership_downstream_stale,
    )

    try:
        audit = audit_ownership_downstream_stale(root=root)
    except OwnershipRebuildError:
        return None
    if not audit.get("is_stale"):
        return None
    release_label = audit.get("release_label") or audit.get("release_id") or "—"
    caps = audit.get("stale_capabilities") or []
    cap_text = ", ".join(caps[:2]) + ("…" if len(caps) > 2 else "")
    return _finalize_attention_item(
        {
            "source_id": OWNERSHIP_DOWNSTREAM_SOURCE_ID,
            "human_name": "Ownership data",
            "active_release_id": audit.get("release_id"),
            "active_release_label": release_label,
            "publisher_latest_release_id": audit.get("release_id"),
            "publisher_latest_label": release_label,
            "pending_release_id": None,
            "pending_state": None,
            "local_release_id": None,
            "new_release_available": False,
            "availability_summary": "downstream stale",
            "availability_source": "ownership_downstream_rebuild",
            "health": None,
            "downstream_stale": True,
            "next_action": {
                "label": "Rebuild downstream",
                "detail": (
                    f"Ownership pair {release_label} is ACTIVE; derived outputs lag "
                    f"({cap_text or 'bridge/policy'})."
                ),
                "endpoint": "action_ownership_downstream_rebuild",
                "wired": True,
                "method": "post",
                "read_only": False,
            },
            "mechanism": "derived from ACTIVE ownership pair",
            "concise_state": "downstream stale",
            "release_line": release_label,
            "panel_source_id": "cms.snf_all_owners",
        }
    )


def _finalize_attention_item(item: dict[str, Any]) -> dict[str, Any]:
    """Add action-first display fields for Sources queue cards."""
    if item.get("concise_state"):
        return item
    pending_state = str(item.get("pending_state") or "").upper()
    pending_id = item.get("pending_release_id")
    active_label = item.get("active_release_label") or item.get("active_release_id")
    publisher_label = item.get("publisher_latest_label") or item.get("publisher_latest_release_id")
    concise = item.get("availability_summary") or "—"
    release_line = active_label or publisher_label or "—"
    audit = item.get("candidate_audit") or {}
    if audit.get("same_quarter") and audit.get("is_redundant_reacquisition"):
        release_line = format_release_month_label(pending_id) or str(pending_id or release_line)
        concise = "re-acquired same quarter"
    elif pending_state == "ACQUIRED" and pending_id:
        if str(pending_id) == str(item.get("active_release_id") or ""):
            release_line = format_release_month_label(pending_id) or str(pending_id)
            concise = f"{release_line} · pending validation"
        elif str(pending_id).count("-") == 2:
            parts = str(pending_id).split("-")
            month = int(parts[1])
            abbr = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")[month - 1]
            release_line = f"{abbr} {int(parts[2])}"
            concise = f"{release_line} acquired"
        else:
            release_line = format_release_month_label(pending_id) or str(pending_id)
            concise = f"{release_line} acquired"
    elif item.get("new_release_available") and publisher_label:
        concise = f"{publisher_label} available"
        release_line = publisher_label
    elif active_label and not item.get("new_release_available") and not pending_id:
        concise = f"{active_label} active"
        release_line = active_label
    elif item.get("downstream_stale") and active_label:
        concise = "downstream stale"
        release_line = active_label
    elif pending_state == "VALIDATED" and pending_id:
        concise = f"{format_release_month_label(pending_id) or pending_id} ready to promote"
        release_line = format_release_month_label(pending_id) or str(pending_id)
    elif pending_state and pending_id:
        concise = f"{pending_id} {pending_state.lower()}"
        release_line = format_release_month_label(pending_id) or str(pending_id)
    item = dict(item)
    item["concise_state"] = concise
    item["release_line"] = release_line
    item.setdefault("panel_source_id", item.get("source_id"))
    return item


def _hc_rebuild_orchestration_hint() -> str:
    from release_control_plane import DERIVED_ARTIFACT_PIPELINES

    pipes = DERIVED_ARTIFACT_PIPELINES.get("cms.health_citations", ())
    if not pipes:
        return "facility packaging (ACTIVE cms.health_citations gate)"
    return "; ".join(str(pipe.get("rebuild") or pipe.get("artifact")) for pipe in pipes)


def build_operator_reference_summary(
    source_id: str,
    *,
    workflow: dict[str, Any],
) -> dict[str, Any]:
    """Compact operator headline for post-activation source detail (Health Citations reference)."""
    from release_review_policy import zweli_applies

    release_availability = workflow.get("release_availability") or {}
    provenance = workflow.get("provenance_freshness") or {}
    downstream_rows = provenance.get("downstream_artifacts") or []
    downstream_stale = [row for row in downstream_rows if row.get("freshness") == "STALE"]
    pending_state = str(workflow.get("pending_state") or "").upper()
    new_release = bool(release_availability.get("new_release_available"))

    if new_release or pending_state in {"ACQUIRED", "VALIDATED", "DETECTED"}:
        status_label = "NEEDS ATTENTION"
        status_tone = "attention"
    elif downstream_stale:
        status_label = "NEEDS ATTENTION"
        status_tone = "attention"
    else:
        status_label = "Up to date"
        status_tone = "current"

    return {
        "status_label": status_label,
        "status_tone": status_tone,
        "active_label": release_availability.get("active_release_label")
        or format_release_month_label(workflow.get("active_release_id")),
        "cms_latest_label": release_availability.get("publisher_latest_label"),
        "downstream_stale": downstream_stale,
        "downstream_current": [row for row in downstream_rows if row.get("freshness") == "CURRENT"],
        "show_release_review": pending_state == "VALIDATED",
        "show_zweli": zweli_applies(source_id) and bool(workflow.get("zweli_status")),
        "rebuild_orchestration_hint": _hc_rebuild_orchestration_hint()
        if source_id == "cms.health_citations"
        else None,
    }


def _next_operator_action(
    source_id: str,
    *,
    record: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
    control_row: dict[str, Any] | None,
    release_availability: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    pending = (control_row or {}).get("pending") or {}
    pending_state = str(pending.get("state") or "").upper()
    active_id = ((control_row or {}).get("active") or {}).get("active_release_id")
    actions = list((record or {}).get("actions_enabled") or [])

    if (
        pending_state == "DETECTED"
        and (
            pending.get("release_id") != active_id
            or bool((pending.get("metadata") or {}).get("publisher_revision_changed"))
        )
        and source_id
        in {
            "cms.provider_info",
            "cms.pbj_nurse_staffing",
            "cms.pbj_non_nurse_staffing",
            "cms.snf_all_owners",
            "cms.snf_enrollments",
        }
    ):
        release_label = format_release_month_label(pending.get("release_id")) or pending.get("release_id")
        revision = bool((pending.get("metadata") or {}).get("publisher_revision_changed"))
        return {
            "label": f"Acquire revised {release_label}" if revision else f"Acquire {release_label}",
            "detail": (
                "Acquire the revised publisher artifact without overwriting ACTIVE; activation still requires review."
                if revision
                else "Acquire and process only this DETECTED source; activation still requires review."
            ),
            "endpoint": "action_source_acquire",
            "endpoint_args": {"source_id": source_id},
            "wired": True,
            "method": "post",
            "read_only": False,
            "busy_submit": True,
            "busy_label": f"Acquiring {release_label}…",
            "busy_detail": "Using the existing production handler for this source only.",
        }

    if release_availability is None:
        release_availability = build_release_availability_context(
            source_id,
            control_row=control_row,
            snapshot=snapshot,
            record=record,
            root=root,
        )

    from release_control_plane import stale_derived_consumers
    from provenance_freshness import downstream_stale_capabilities_for_source

    stale_map = stale_derived_consumers(root)
    downstream_stale = downstream_stale_capabilities_for_source(source_id, root=root)

    if source_id == "cms.health_citations":
        target_label = release_availability.get("publisher_latest_label") or "next release"
        pub_id = release_availability.get("publisher_latest_release_id")
        active_label = format_release_month_label(active_id) or active_id
        pub_label = release_availability.get("publisher_latest_label") or active_label
        if pending_state == "VALIDATED":
            release_label = format_release_month_label(pending.get("release_id")) or pending.get("release_id")
            from release_review_policy import release_review_query

            return {
                "label": f"Activate {release_label}",
                "detail": f"{release_label} is VALIDATED — explicit human activation in Release Review.",
                "endpoint": "release_review",
                "endpoint_args": release_review_query(source_id, pending.get("release_id")),
                "wired": True,
                "read_only": True,
            }
        if pending_state == "ACQUIRED":
            release_label = format_release_month_label(pending.get("release_id")) or pending.get("release_id")
            return {
                "label": f"Validate {release_label}",
                "detail": "Bundle artifact acquired — run structural validation before promotion.",
                "endpoint": "action_citations_validate",
                "endpoint_args": {"release_id": pending.get("release_id")},
                "wired": True,
                "method": "post",
            }
        if active_id and not pending and not release_availability.get("new_release_available"):
            if downstream_stale:
                stale_labels = []
                for cap in downstream_stale:
                    from release_control_plane import CONSUMER_OPERATOR_LABELS

                    stale_labels.append(CONSUMER_OPERATOR_LABELS.get(cap, cap))
                return {
                    "label": "Rebuild facility packages",
                    "detail": (
                        f"Health Citations · {active_label} active · CMS latest {pub_label}. "
                        f"National file is current; facility citation tables are stale "
                        f"({', '.join(stale_labels[:2])}{'…' if len(stale_labels) > 2 else ''}). "
                        "Rebuilds local facility citation slices from ACTIVE national file — no deploy."
                    ),
                    "endpoint": "action_citation_packages_rebuild",
                    "wired": True,
                    "method": "post",
                    "read_only": False,
                }
            return {
                "label": "Check CMS",
                "detail": f"Up to date · {active_label} active · CMS latest {pub_label}",
                "endpoint": "source_detail",
                "endpoint_args": {"source_id": source_id},
                "wired": True,
                "method": "get",
                "read_only": True,
            }
        if release_availability.get("new_release_available"):
            if release_availability.get("bundle_provenance_ok") and release_availability.get("local_artifact_ready"):
                return {
                    "label": f"Validate {target_label}",
                    "detail": (
                        f"CMS latest {target_label}; local artifact matches Provider Info bundle manifest. "
                        "Adopt and validate without re-download."
                    ),
                    "endpoint": "action_citations_validate",
                    "endpoint_args": {"release_id": pub_id},
                    "wired": True,
                    "method": "post",
                }
            return {
                "label": f"Acquire {target_label}",
                "detail": f"CMS latest {target_label}; download from official CMS endpoint required.",
                "endpoint": "source_detail",
                "endpoint_args": {"source_id": source_id},
                "wired": False,
                "read_only": True,
                "not_wired_label": "Not yet wired",
            }

    if source_id == "cms.provider_info" and active_id and not pending:
        active_label = format_release_month_label(active_id) or active_id
        pub_label = release_availability.get("publisher_latest_label")
        from pbj320_stage_provider_info import (
            audit_provider_info_pbj320_destination,
            evaluate_pi_stage_publish_eligibility,
            load_stage_manifest,
        )

        dest_audit = audit_provider_info_pbj320_destination(root=root)
        stage_manifest = load_stage_manifest(active_id, root=root)
        publish_eligibility = evaluate_pi_stage_publish_eligibility(stage_manifest, root=root)
        if dest_audit.get("canonical_current") and not publish_eligibility.get("publishable"):
            if stage_manifest and str(stage_manifest.get("status") or "") == "STAGED":
                stale_detail = str(publish_eligibility.get("detail") or "rebuild required")
                return {
                    "label": "Stage again against current production",
                    "detail": (
                        f"Provider Information · {active_label} · existing Stage manifest is out of date "
                        f"({stale_detail}). Rebuild against current origin/master baseline."
                    ),
                    "endpoint": "action_pi_stage_pbj320",
                    "wired": True,
                    "method": "post",
                    "read_only": False,
                    "busy_submit": True,
                    "busy_label": "Staging Provider Information…",
                    "busy_detail": (
                        "Isolated baseline worktree + PI overlay; captures publication_base_sha and "
                        "shared-destination fingerprints."
                    ),
                }
            pending_files = 4
            return {
                "label": "Stage for PBJ320",
                "detail": (
                    f"Provider Information · {active_label} ACTIVE · "
                    f"prepare {pending_files} pbj-root destination artifacts (working tree only; no publish)."
                ),
                "endpoint": "action_pi_stage_pbj320",
                "wired": True,
                "method": "post",
                "read_only": False,
                "busy_submit": True,
                "busy_label": "Staging Provider Information…",
                "busy_detail": (
                    "Running validation and build gates (Norm sync, combined rebuild, "
                    "pre-publication checks). This may take several minutes."
                ),
            }
        if publish_eligibility.get("publishable"):
            from pbj320_publish_provider_info import load_publication_record

            pub = load_publication_record(active_id, root=root)
            if pub and pub.get("production_verified"):
                return {
                    "label": "Production verified",
                    "detail": (
                        f"Provider Information {active_label} verified on production · "
                        f"commit {str(pub.get('commit_sha') or '')[:12]}…"
                    ),
                    "endpoint": "pi_stage_manifest",
                    "endpoint_args": {"release_id": active_id},
                    "wired": True,
                    "method": "get",
                    "read_only": True,
                }
            if pub and pub.get("push_succeeded"):
                return {
                    "label": "Verify production",
                    "detail": (
                        f"Provider Information {active_label} pushed · commit "
                        f"{str(pub.get('commit_sha') or '')[:12]}… · run read-only production checks."
                    ),
                    "endpoint": "pi_stage_manifest",
                    "endpoint_args": {"release_id": active_id},
                    "wired": True,
                    "method": "get",
                    "read_only": True,
                }
            return {
                "label": "Publish to PBJ320",
                "detail": (
                    f"Provider Information · {active_label} STAGED · "
                    f"review manifest and confirm selective commit to pbj-root."
                ),
                "endpoint": "pi_stage_manifest",
                "endpoint_args": {"release_id": active_id},
                "wired": True,
                "method": "get",
                "read_only": True,
            }
        if not release_availability.get("new_release_available") and stale_map.get(source_id):
            return {
                "label": "Rebuild downstream",
                "detail": (
                    f"Provider Information · {active_label} active · CMS latest {pub_label or active_label}. "
                    f"Derived consumers stale ({', '.join(stale_map[source_id][:3])}{'…' if len(stale_map[source_id]) > 3 else ''})."
                ),
                "endpoint": "source_detail",
                "endpoint_args": {"source_id": source_id},
                "wired": False,
                "read_only": True,
                "not_wired_label": "Not yet wired",
            }
        if not release_availability.get("new_release_available"):
            return {
                "label": "Check CMS",
                "detail": f"Up to date · {active_label} active · CMS latest {pub_label or active_label}",
                "endpoint": "action_pi_check",
                "wired": True,
                "method": "post",
                "read_only": False,
            }

    if source_id == "cms.nh_ownership" and release_availability.get("new_release_available"):
        target_label = release_availability.get("publisher_latest_label") or "next release"
        return {
            "label": f"Acquire {target_label}",
            "detail": f"CMS latest {target_label}; acquisition lifecycle not wired in Data Ops yet.",
            "endpoint": "source_detail",
            "endpoint_args": {"source_id": source_id},
            "wired": False,
            "read_only": True,
            "not_wired_label": "Not yet wired",
        }

    if pending_state in {"ACQUIRED", "VALIDATED"} and source_id in {"cms.snf_all_owners", "cms.snf_enrollments"}:
        from ownership_pairing import PAIR_SOURCE_ID, pair_lifecycle_action, pairing_status

        action = pair_lifecycle_action(pairing_status(root))
        return {
            "label": "Review candidate" if action.get("kind") == "activate_pair" else (action.get("label") or "Validate pair"),
            "detail": action.get("detail") or "Owners and enrollment releases must move together.",
            "endpoint": action.get("endpoint") or "source_detail_panel",
            "endpoint_args": action.get("endpoint_args") or {"source_id": PAIR_SOURCE_ID},
            "wired": True,
            "read_only": True,
            "opens_panel": True,
            "page_endpoint": "source_detail",
            "page_endpoint_args": {"source_id": PAIR_SOURCE_ID},
        }
    if source_id in {"cms.snf_all_owners", "cms.snf_enrollments"} and active_id and not pending:
        from ownership_downstream_rebuild import audit_ownership_downstream_stale

        audit = audit_ownership_downstream_stale()
        active_label = format_release_month_label(active_id) or active_id
        pub_label = (release_availability or {}).get("publisher_latest_label") or active_label
        if audit.get("is_stale"):
            caps = audit.get("stale_capabilities") or []
            return {
                "label": "Rebuild downstream",
                "detail": (
                    f"Ownership · {active_label} active · CMS latest {pub_label}. "
                    f"Derived outputs stale ({', '.join(caps[:2])}{'…' if len(caps) > 2 else ''})."
                ),
                "endpoint": "action_ownership_downstream_rebuild",
                "wired": True,
                "method": "post",
                "read_only": False,
            }
        if not release_availability.get("new_release_available"):
            return {
                "label": "Check CMS",
                "detail": f"Up to date · {active_label} active · CMS latest {pub_label}",
                "endpoint": "source_detail",
                "endpoint_args": {"source_id": source_id},
                "wired": True,
                "method": "get",
                "read_only": True,
            }
    if source_id == "cms.sff_pdf_list" and active_id and not pending:
        active_label = format_release_month_label(active_id) or active_id
        pub_label = (release_availability or {}).get("publisher_latest_label")
        if pub_label and not release_availability.get("new_release_available"):
            return {
                "label": "Check CMS",
                "detail": f"Up to date · ACTIVE {active_label} · CMS latest {pub_label}",
                "endpoint": "action_sff_check",
                "wired": True,
                "method": "post",
                "read_only": True,
                "return_to": "/sources",
            }
        if pub_label and release_availability.get("new_release_available"):
            return {
                "label": "Check CMS",
                "detail": f"New CMS posting · {pub_label}",
                "endpoint": "action_sff_check",
                "wired": True,
                "method": "post",
                "read_only": True,
                "return_to": "/sources",
            }
        return {
            "label": "Check CMS",
            "detail": f"ACTIVE {active_label} — compare against CMS SFF posting.",
            "endpoint": "action_sff_check",
            "wired": True,
            "method": "post",
            "read_only": True,
            "return_to": "/sources",
        }

    if pending_state == "ACQUIRED":
        release_label = format_release_month_label(pending.get("release_id")) or pending.get("release_id")
        if source_id == "cms.pbj_nurse_staffing":
            from operator_freshness import audit_nurse_staffing_candidate_state

            audit = audit_nurse_staffing_candidate_state(control_row=control_row)
            if audit.get("is_redundant_reacquisition"):
                return {
                    "label": "Dismiss re-acquisition",
                    "detail": audit.get("summary") or "Redundant same-quarter candidate.",
                    "endpoint": "action_nurse_dismiss_candidate",
                    "wired": True,
                    "method": "post",
                    "read_only": False,
                }
            if active_id and str(pending.get("release_id") or "") == str(active_id):
                return {
                    "label": "Validate re-acquisition",
                    "detail": audit.get("summary") or (
                        f"Same quarter {release_label} — validation workflow not wired in Data Ops yet."
                    ),
                    "endpoint": "source_detail",
                    "endpoint_args": {"source_id": source_id},
                    "wired": False,
                    "read_only": True,
                    "not_wired_label": "Validation not wired",
                }
            return {
                "label": f"Validate {release_label or 'pending release'}",
                "detail": f"Pending {pending.get('release_id')} is ACQUIRED — run structural validation before review.",
                "endpoint": "source_detail",
                "endpoint_args": {"source_id": source_id},
                "wired": False,
                "read_only": True,
                "not_wired_label": "Not yet wired",
            }
        from release_review_policy import release_review_query

        return {
            "label": f"Review {release_label or 'pending release'}",
            "detail": f"Pending {pending.get('release_id')} is ACQUIRED — validate on Sources before activation review.",
            "endpoint": "source_detail",
            "endpoint_args": {"source_id": source_id},
            "wired": True,
            "read_only": True,
        }
    if pending_state == "VALIDATED" and source_id == "cms.provider_info":
        from release_review_policy import release_review_query

        approvable = promote_candidate_permitted(
            {
                "state": pending_state,
                "source_id": source_id,
                "release_id": pending.get("release_id"),
            }
        )
        release_label = format_release_month_label(pending.get("release_id")) or pending.get("release_id")
        return {
            "label": f"Activate {release_label}" if approvable else "Complete quality gates",
            "detail": "VALIDATED candidate — promotion stays explicit via Release Review.",
            "endpoint": "release_review",
            "endpoint_args": release_review_query(source_id, pending.get("release_id")),
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
    if active_id and not pending and source_id not in {"cms.sff_pdf_list"}:
        active_label = format_release_month_label(active_id) or active_id
        pub_label = (release_availability or {}).get("publisher_latest_label")
        if pub_label and not release_availability.get("new_release_available"):
            return {
                "label": "Check CMS",
                "detail": f"Up to date · CMS latest {pub_label}",
                "endpoint": f"action_{source_id.split('.')[-1]}_check" if source_id == "cms.provider_info" else "source_detail",
                "endpoint_args": {"source_id": source_id} if source_id != "cms.provider_info" else {},
                "wired": source_id in {"cms.provider_info", "cms.pbj_nurse_staffing"},
                "method": "post" if source_id in {"cms.provider_info", "cms.pbj_nurse_staffing"} else None,
                "read_only": source_id not in {"cms.provider_info", "cms.pbj_nurse_staffing"},
            }
        if pub_label and release_availability.get("new_release_available"):
            return {
                "label": "Check CMS",
                "detail": f"New release · {pub_label}",
                "endpoint": "action_pi_check" if source_id == "cms.provider_info" else "source_detail",
                "endpoint_args": {"source_id": source_id} if source_id != "cms.provider_info" else {},
                "wired": source_id == "cms.provider_info",
                "method": "post" if source_id == "cms.provider_info" else None,
                "read_only": source_id != "cms.provider_info",
            }
        return {
            "label": "Check CMS",
            "detail": f"ACTIVE {active_label} — compare against CMS publication index.",
            "endpoint": "action_pi_check" if source_id == "cms.provider_info" else "source_detail",
            "endpoint_args": {"source_id": source_id} if source_id != "cms.provider_info" else {},
            "wired": source_id in {"cms.provider_info", "cms.pbj_nurse_staffing", "cms.health_citations"},
            "method": "post" if source_id in {"cms.provider_info", "cms.pbj_nurse_staffing"} else None,
            "read_only": source_id not in {"cms.provider_info", "cms.pbj_nurse_staffing", "cms.health_citations"},
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
    """SFF lifecycle sequence, including governed staging of DETECTED candidates."""
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
            state="current" if active_id else "upcoming",
            detail=(
                f"Compare CMS SFF posting against ACTIVE {active_id}."
                if active_id
                else "Discover the latest CMS-posted SFF PDF before acquisition."
            ),
        )
    )

    acquire_state = "current" if pending_state == "DETECTED" else ("completed" if active_id or pending_id else "upcoming")
    if pending_state in {"ACQUIRED", "VALIDATED"} and not active_id:
        acquire_state = "completed"
    steps.append(
        step(
            "acquire_pdf",
            "Stage detected SFF PDF",
            state=acquire_state,
            detail=(
                "Download the recorded cms.gov PDF, verify its embedded posting month, parse, normalize and validate."
                if pending_state == "DETECTED"
                else "Acquisition uses the governed sff_release.stage_pdf path."
            ),
            action={
                "label": "Stage detected SFF PDF",
                "endpoint": "action_sff_stage_detected",
                "wired": True,
                "method": "post",
                "read_only": False,
                "busy_submit": True,
                "busy_label": "Staging SFF PDF…",
                "busy_detail": "Downloading the recorded CMS PDF, parsing four tables and validating the candidate.",
            }
            if pending_state == "DETECTED"
            else None,
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
            detail="Facility package build/deploy is explicit in Dashboard Builder.",
            action={
                "label": "Dashboard Builder",
                "endpoint": "dashboard_builder",
                "wired": True,
                "read_only": False,
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
    release_availability: dict[str, Any] | None = None,
    theme_publication: Any | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Operator-facing workflow summary from existing control plane + probe overlay."""
    from provenance_freshness import build_source_provenance_freshness, downstream_stale_capabilities_for_source
    from release_control_plane import CAPABILITY_LABELS
    from release_review_policy import zweli_applies

    root = root or cms_data_paths.repo_root()

    active = (control_row or {}).get("active") or {}
    pending = (control_row or {}).get("pending") or {}
    impact = (control_row or {}).get("impact") or {}
    stale_caps = downstream_stale_capabilities_for_source(source_id, root=root)
    downstream = [
        CAPABILITY_LABELS.get(item, item)
        for item in stale_caps
    ] or [
        CAPABILITY_LABELS.get(item, item)
        for item in (impact.get("would_mark_stale") or [])
    ]
    zweli_status = None
    if zweli_applies(source_id):
        if snapshot:
            zweli_status = snapshot.get("zweli_status")
        elif isinstance(pending.get("metadata"), dict):
            zweli_status = pending.get("metadata", {}).get("zweli_status")

    if release_availability is None:
        release_availability = build_release_availability_context(
            source_id,
            control_row=control_row,
            snapshot=snapshot,
            record=record,
            root=root,
            theme_publication=theme_publication,
        )

    provenance_freshness = build_source_provenance_freshness(
        source_id,
        root=root,
        theme_publication=theme_publication,
    )

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
        "release_availability": release_availability,
        "provenance_freshness": provenance_freshness,
        "next_action": _next_operator_action(
            source_id,
            record=record,
            snapshot=snapshot,
            control_row=control_row,
            release_availability=release_availability,
            root=root,
        ),
        "lifecycle_steps": None,
    }
    workflow["operator_reference"] = build_operator_reference_summary(source_id, workflow=workflow)
    from operator_freshness import audit_nurse_staffing_candidate_state, build_freshness_layers

    if source_id == "cms.pbj_nurse_staffing":
        workflow["candidate_audit"] = audit_nurse_staffing_candidate_state(control_row=control_row, root=root)
    workflow["freshness_layers"] = build_freshness_layers(source_id, workflow=workflow, root=root)
    if source_id == "cms.provider_info":
        from pbj320_stage_provider_info import ProviderInfoStageError, audit_provider_info_pbj320_destination, load_stage_manifest

        active_id = str(active.get("active_release_id") or "")
        try:
            workflow["pbj320_stage_audit"] = audit_provider_info_pbj320_destination(root=root)
        except ProviderInfoStageError as exc:
            workflow["pbj320_stage_audit"] = {"destination_staged": False, "error": str(exc)}
        workflow["pbj320_stage_manifest"] = load_stage_manifest(active_id, root=root) if active_id else None
    if source_id == "cms.sff_pdf_list":
        workflow["lifecycle_steps"] = build_sff_lifecycle_steps(control_row=control_row)
    return workflow


def check_provider_info_cms(
    *,
    fetch_json: FetchJson | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    import cms_provider_info_acquire as acq
    from active_release_registry import get_active_release, registry_path
    from cms_theme_publication import get_latest_nh_theme_publication, publication_availability_for_source
    from release_control_plane import ReleaseState, control_plane_root, record_candidate

    root = root or cms_data_paths.repo_root()
    cms = acq.resolve_cms_provider_info_release(fetch_json=fetch_json)
    newer = acq.cms_is_newer_than_local(cms, root)
    local = acq.latest_local_provider_info(root)
    snap = probe_source("cms.provider_info", check_cms=True, fetch_json=fetch_json, root=root)

    control_root = control_plane_root(root)
    active_provider = get_active_release("cms.provider_info", registry_path(control_root)) or {}
    active_ownership = get_active_release("cms.nh_ownership", registry_path(control_root)) or {}
    provider_active_id = str(active_provider.get("active_release_id") or "") or None
    ownership_active_id = (
        str(active_ownership.get("active_release_id") or "")
        or provider_active_id
    )
    theme_publication = get_latest_nh_theme_publication(
        fetch_json=fetch_json,
        force_refresh=True,
    )
    provider_theme = publication_availability_for_source(
        "cms.provider_info",
        active_release_id=provider_active_id,
        publication=theme_publication,
    )
    ownership_theme = publication_availability_for_source(
        "cms.nh_ownership",
        active_release_id=ownership_active_id,
        publication=theme_publication,
    )
    theme_source_set = [
        {
            "source_id": source_id,
            "role": role,
            "cms_dataset_id": (availability or {}).get("cms_dataset_id"),
            "release_id": (availability or {}).get("product_release_id"),
            "manifest_filename": (availability or {}).get("manifest_filename"),
            "manifest_filesize": (availability or {}).get("manifest_filesize"),
        }
        for source_id, role, availability in (
            ("cms.provider_info", "provider_info", provider_theme),
            ("cms.nh_ownership", "nh_ownership", ownership_theme),
        )
        if availability and availability.get("in_latest_publication")
    ]
    if newer:
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
            metadata={
                "publisher_label": label,
                "publisher_modified": cms.modified,
                "publisher_released": cms.released,
                "theme_publication_id": getattr(theme_publication, "publication_id", None),
                "theme_publication_date": getattr(theme_publication, "publication_date", None),
                "source_set": theme_source_set,
            },
            root=control_root,
        )
    if ownership_theme and ownership_theme.get("new_release_available"):
        ownership_release_id = str(ownership_theme.get("product_release_id") or "")
        if not ownership_release_id:
            raise RuntimeError("CMS theme publication has no release identity for cms.nh_ownership")
        provider_release_id = str(
            (provider_theme or {}).get("product_release_id")
            or getattr(cms, "data_vintage_label", "")
            or ownership_release_id
        )
        record_candidate(
            "cms.nh_ownership",
            ownership_release_id,
            ReleaseState.DETECTED,
            metadata={
                "candidate_kind": "DERIVED",
                "upstream_source_id": "cms.provider_info",
                "upstream_release_id": provider_release_id,
                "cms_dataset_id": ownership_theme.get("cms_dataset_id"),
                "theme_publication_id": ownership_theme.get("cms_publication_id"),
                "theme_publication_date": ownership_theme.get("cms_publication_date"),
                "manifest_filename": ownership_theme.get("manifest_filename"),
                "manifest_filesize": ownership_theme.get("manifest_filesize"),
            },
            root=control_root,
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
        "theme_publication": {
            "publication_id": getattr(theme_publication, "publication_id", None),
            "publication_date": getattr(theme_publication, "publication_date", None),
            "source_set": theme_source_set,
        } if theme_publication is not None else None,
        "nh_ownership": ownership_theme,
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


def check_sff_cms(
    *,
    fetch_bytes: Callable[[str], bytes] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    from sff_release import check_sff_cms as _check

    control_root = Path(__file__).resolve().parent
    result = _check(fetch_bytes=fetch_bytes, root=control_root)
    snap = probe_source("cms.sff_pdf_list", check_cms=False, root=root or cms_data_paths.repo_root())
    return {
        **result,
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
            {
                "role": "provider_info",
                "source_id": "cms.provider_info",
                "source_path": str(norm),
                "hash": sha256_file(norm),
            },
            {
                "role": "nh_ownership",
                "source_id": "cms.nh_ownership",
                "cms_dataset_id": "y2hd-n93e",
                "source_path": str(ownership),
                "hash": sha256_file(ownership),
            },
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
    from release_review_policy import (
        load_governed_candidate,
        structural_status_from_candidate,
        zweli_applies,
    )

    root = root or cms_data_paths.repo_root()
    release_id = (release_id or "").strip()
    source_id = (source_id or "").strip()
    from cms_source_registry import is_review_only_source

    if is_review_only_source(source_id):
        raise ApprovalError(f"{source_id} is review-only; no activation is permitted")
    pending = load_governed_candidate(source_id, release_id, root=root)
    if not pending or str(pending.get("state") or "").upper() != "VALIDATED":
        raise ApprovalError(f"No VALIDATED candidate {source_id} {release_id} for activation")

    structural = structural_status_from_candidate(pending)
    if structural != "PASS":
        raise ApprovalError(
            f"structural validation must pass before approval ({structural})"
        )

    # Fail before Zweli resolution, audit writes, source-path rebinding, or any
    # ACTIVE registry mutation.  Provider/nurse candidates remain VALIDATED and
    # reviewable when a required derivative cannot be regenerated safely.
    if source_id in {"cms.provider_info", "cms.pbj_nurse_staffing"}:
        from derived_provenance import (
            DerivativeActivationBlocked,
            assert_activation_derivatives_ready,
        )

        try:
            assert_activation_derivatives_ready(source_id, root=root)
        except DerivativeActivationBlocked as exc:
            raise ApprovalError(str(exc)) from exc

    if zweli_applies(source_id):
        state = resolve_zweli_state_for_release(source_id, release_id, root=root)
    else:
        state = ZweliState.NOT_RUN

    if source_id == "cms.provider_info":
        bundle = build_provider_info_promotion_bundle(release_id)
        source_path = bundle["source_path"]
        promotion_metadata = dict(bundle["metadata"])
        promotion_metadata["zweli_state"] = state.value
        promotion_metadata["approval_note"] = note
    else:
        source_uri = pending.get("source_uri")
        if not source_uri:
            raise ApprovalError(
                f"release {source_id} {release_id} has no candidate source_uri"
            )
        local = str(source_uri).replace("file:///", "").replace("file://", "")
        source_path = Path(local)
        if not source_path.is_file():
            raise ApprovalError(
                f"release {source_id} {release_id} candidate artifact missing: {source_path}"
            )
        promotion_metadata = {
            "zweli_state": state.value,
            "approval_note": note,
            "structural_status": structural,
        }
        if isinstance(pending.get("metadata"), dict):
            promotion_metadata.update(
                {
                    k: v
                    for k, v in pending["metadata"].items()
                    if k not in promotion_metadata
                }
            )
    entry = approve_release(
        source_id,
        release_id,
        zweli_state=state,
        note=note,
        audit_path=audit_path,
    )
    control_root = control_plane_root(root)
    validation = pending.get("validation") if isinstance(pending.get("validation"), dict) else {}
    record_candidate(
        source_id,
        release_id,
        ReleaseState.VALIDATED,
        source_path=source_path,
        validation={
            "status": "PASS",
            "validated_at": validation.get("validated_at") or entry.timestamp,
        },
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
