"""PBJ Data Ops status probes and Provider Info action wrappers.

Control plane over canonical services — UI must call these modules, not
duplicate ETL. Provider Info check/acquire → scripts/cms_provider_info_acquire.py.
"""

from __future__ import annotations

import json
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

        # Zweli — comparable prior = previous calendar month for monthly PI
        if run_zweli and raw_path and raw_path.is_file() and local:
            snaps = acq.list_local_provider_info_snapshots(root)
            ey, em = expected_prior_month(local[0], local[1])
            expected_rel = f"{ey:04d}-{em:02d}"
            baseline_path = None
            for y, m, p in snaps:
                if (y, m) == (ey, em):
                    baseline_path = p
                    break
            if baseline_path is not None:
                report = run_provider_info_zweli(
                    raw_path,
                    release_id or "unknown",
                    baseline_csv=baseline_path,
                    baseline_release=expected_rel,
                    baseline_availability=BaselineAvailability.PRESENT,
                    expected_baseline_release=expected_rel,
                )
            else:
                report = run_provider_info_zweli(
                    raw_path,
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
    raw = _latest_by_glob(raw_dir, raw_glob)
    std = _latest_by_glob(std_dir, std_glob)
    label = None
    if std:
        label = _quarter_label_from_name(std.name)
    elif raw:
        label = _quarter_label_from_name(raw.name)
    snap.pbjapp_latest = label
    snap.processed = label if std else None
    if raw:
        _apply_raw_ref(snap, local_file_ref("raw", raw, release_id=label))
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
    raw = _latest_by_glob(own, "SNF_All_Owners*.csv")
    if raw:
        _apply_raw_ref(snap, local_file_ref("snf_all_owners", raw, release_id=raw.name))
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
    # Prefer standalone-looking files; still report co-extracted NH_* if only those exist
    standalone = _latest_by_glob(cit, "*Citation*.csv") if cit.is_dir() else None
    if standalone is None and cit.is_dir():
        standalone = _latest_by_glob(cit, "*.csv")
    if standalone:
        _apply_raw_ref(
            snap, local_file_ref("citations", standalone, release_id=standalone.name)
        )
        snap.pbjapp_latest = standalone.name
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
        return _probe_unmodeled(
            record,
            "SNF Enrollments is a separate CMS source from All Owners; unmodeled on main",
        )
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


def release_review_items(
    snapshots: list[SourceOpsSnapshot] | None = None,
    *,
    check_cms: bool = True,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Items needing human attention for Release Review UI."""
    snaps = snapshots or probe_all_sources(check_cms=check_cms, root=root)
    items: list[dict[str, Any]] = []
    for s in snaps:
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
        if reason:
            items.append(
                {
                    "reason": reason,
                    "source_id": s.source_id,
                    "human_name": s.human_name,
                    "release_id": s.release_id or s.pbjapp_latest,
                    "status": s.status,
                    "zweli_status": s.zweli_status,
                    "structural_status": s.structural_status,
                    "detail": s.detail,
                    "zweli_report": s.zweli_report,
                    "acknowledged": bool(
                        s.release_id
                        and has_acknowledgement(s.source_id, s.release_id)
                    ),
                }
            )
    return items


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


def resolve_zweli_state_for_release(
    source_id: str,
    release_id: str,
    *,
    root: Path | None = None,
) -> ZweliState:
    """Server-authoritative Zweli state for approval (never trust the browser).

    Prefer the stored report for ``source_id``+``release_id``; otherwise recompute
    via canonical probe. Fail closed (raise) when state cannot be established.
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

    # Recompute via canonical probe (Provider Info profile in V0).
    snap = probe_source(source_id, check_cms=False, root=root, run_zweli=True)
    if snap.release_id != release_id:
        raise ApprovalError(
            f"Zweli report missing/mismatched for {source_id} {release_id} "
            f"(probe release={snap.release_id!r}) — fail closed"
        )
    if not snap.zweli_report or snap.zweli_status == ZweliState.NOT_RUN.value:
        raise ApprovalError(
            f"Zweli NOT_RUN / missing for {source_id} {release_id} — fail closed"
        )
    if snap.zweli_report.get("source_id") not in (None, source_id):
        raise ApprovalError("Zweli report source_id mismatch — fail closed")
    if snap.zweli_report.get("release_id") not in (None, release_id):
        raise ApprovalError("Zweli report release_id mismatch — fail closed")
    try:
        return ZweliState(snap.zweli_status)
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
    from data_ops_approval import approve_release

    state = resolve_zweli_state_for_release(source_id, release_id, root=root)
    return approve_release(
        source_id,
        release_id,
        zweli_state=state,
        note=note,
        audit_path=audit_path,
    )
