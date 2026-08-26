"""PBJ Data Ops status probes and Provider Info action wrappers (v0).

Provider Info check/acquire call scripts/cms_provider_info_acquire.py only.
Other families are local read-only probes — no new ingestion.
"""

from __future__ import annotations

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
    CmsSourceRecord,
    OpsStatus,
    SourceFamily,
    get_registry,
    get_source,
)

FetchJson = Callable[[str], Any]


@dataclass
class SourceOpsSnapshot:
    source_id: str
    human_name: str
    source_family: str
    formats: list[str]
    cadence: str
    automation_level: str
    cms_dataset_id: Optional[str]
    cms_latest: Optional[str]
    pbjapp_latest: Optional[str]
    status: str
    last_checked: str
    last_successful_local_processing: Optional[str]
    local_raw_present: bool = False
    local_derived_present: bool = False
    detail: str = ""
    actions_enabled: list[str] = field(default_factory=list)
    error: Optional[str] = None

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
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
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


def _probe_provider_info(
    record: CmsSourceRecord,
    *,
    check_cms: bool,
    fetch_json: FetchJson | None,
    root: Path,
) -> SourceOpsSnapshot:
    import cms_provider_info_acquire as acq
    import cms_provider_release_lib as cpr

    checked = _utc_now_iso()
    snap = SourceOpsSnapshot(
        source_id=record.source_id,
        human_name=record.human_name,
        source_family=record.source_family.value,
        formats=[f.value for f in record.formats],
        cadence=record.cadence.value,
        automation_level=record.automation_level.value,
        cms_dataset_id=record.cms_dataset_id,
        cms_latest=None,
        pbjapp_latest=None,
        status=OpsStatus.UNKNOWN.value,
        last_checked=checked,
        last_successful_local_processing=None,
        actions_enabled=list(record.actions_enabled),
    )

    try:
        handoff_ready = False
        local = acq.latest_local_provider_info(root)
        if local:
            ly, lm, lpath = local
            snap.pbjapp_latest = f"{acq._MONTH_ABBR[lm]} {ly}"
            snap.local_raw_present = True
            norm = (
                cms_data_paths.provider_info_normalized_dir(root)
                / f"ProviderInfoNorm_{ly}_{lm:02d}.csv"
            )
            snap.local_derived_present = norm.is_file() and norm.stat().st_size > 0
            key = cpr.release_key(ly, lm)
            acq_path = (
                cms_data_paths.provider_release_manifest_dir(key.label, root) / "acquisition.json"
            )
            handoff_path = (
                cms_data_paths.provider_release_manifest_dir(key.label, root)
                / "pbj_root_handoff.json"
            )
            if acq_path.is_file():
                import json

                try:
                    acq_data = json.loads(acq_path.read_text(encoding="utf-8"))
                    snap.last_successful_local_processing = acq_data.get(
                        "acquired_at"
                    ) or _mtime_iso(acq_path)
                except (OSError, json.JSONDecodeError):
                    snap.last_successful_local_processing = _mtime_iso(acq_path)
            elif snap.local_derived_present:
                snap.last_successful_local_processing = _mtime_iso(norm)
            else:
                snap.last_successful_local_processing = _mtime_iso(lpath)

            if handoff_path.is_file():
                import json

                try:
                    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
                    promo = handoff.get("provider_promotion") or {}
                    handoff_ready = bool(promo.get("ready_for_pbj_commit")) or bool(
                        (handoff.get("pbj_root_sync") or {}).get("sha256")
                    )
                except (OSError, json.JSONDecodeError):
                    handoff_ready = True

        cms = None
        if check_cms:
            cms = acq.resolve_cms_provider_info_release(fetch_json=fetch_json)
            snap.cms_latest = cms.data_vintage_label

        if cms is not None and acq.cms_is_newer_than_local(cms, root):
            snap.status = OpsStatus.CMS_NEWER.value
            snap.detail = "CMS metastore vintage newer than local NH_ProviderInfo CSV"
            return snap

        if not snap.local_raw_present:
            snap.status = OpsStatus.UNKNOWN.value
            snap.detail = "No real (non-LFS) local NH_ProviderInfo CSV found"
            return snap

        if not snap.local_derived_present:
            snap.status = OpsStatus.PROCESSING_REQUIRED.value
            snap.detail = "Raw Provider Info present; normalized output missing"
            return snap

        if (
            handoff_ready
            and cms is not None
            and not acq.cms_is_newer_than_local(cms, root)
        ):
            snap.status = OpsStatus.READY_FOR_HANDOFF.value
            snap.detail = "Local matches CMS; Norm + handoff artifact present"
            return snap

        if cms is not None and not acq.cms_is_newer_than_local(cms, root):
            snap.status = OpsStatus.CURRENT.value
            snap.detail = "Local Provider Info vintage matches CMS"
            return snap

        if snap.local_derived_present:
            snap.status = OpsStatus.UNKNOWN.value
            snap.detail = "Local processed snapshot present; CMS not checked"
            return snap

        snap.status = OpsStatus.UNKNOWN.value
        return snap
    except Exception as exc:  # noqa: BLE001 — surface as ERROR status for ops UI
        snap.status = OpsStatus.ERROR.value
        snap.error = str(exc)
        snap.detail = f"Provider Info probe failed: {exc}"
        return snap


def _probe_nurse(record: CmsSourceRecord, root: Path) -> SourceOpsSnapshot:
    checked = _utc_now_iso()
    raw_dir = cms_data_paths.nurse_raw_dir(root)
    std_dir = cms_data_paths.standardized_nurse_dir(root)
    raw = _latest_by_glob(raw_dir, "PBJ_dailynurse*.csv")
    std = _latest_by_glob(std_dir, "PBJ_dailynurse*.csv")
    label = None
    if std:
        label = _quarter_label_from_name(std.name)
    elif raw:
        label = _quarter_label_from_name(raw.name)

    if raw and not std:
        status = OpsStatus.PROCESSING_REQUIRED.value
        detail = "Raw nurse CSV present; standardized output missing"
    elif not raw and not std:
        status = OpsStatus.UNKNOWN.value
        detail = "No local nurse PBJ files found"
    elif std and not raw:
        status = OpsStatus.UNKNOWN.value
        detail = "Standardized nurse files present; CMS currency unknown (no dataset ID)"
    else:
        status = OpsStatus.UNKNOWN.value
        detail = "Local nurse files present; CMS currency unknown (no dataset ID)"

    return SourceOpsSnapshot(
        source_id=record.source_id,
        human_name=record.human_name,
        source_family=record.source_family.value,
        formats=[f.value for f in record.formats],
        cadence=record.cadence.value,
        automation_level=record.automation_level.value,
        cms_dataset_id=None,
        cms_latest=None,
        pbjapp_latest=label,
        status=status,
        last_checked=checked,
        last_successful_local_processing=_mtime_iso(std) if std else None,
        local_raw_present=bool(raw),
        local_derived_present=bool(std),
        detail=detail,
        actions_enabled=[],
    )


def _probe_nonnurse(record: CmsSourceRecord, root: Path) -> SourceOpsSnapshot:
    checked = _utc_now_iso()
    raw_dir = cms_data_paths.nonnurse_raw_dir(root)
    std_dir = cms_data_paths.standardized_nonnurse_dir(root)
    raw = _latest_by_glob(raw_dir, "PBJ_dailynonnurse*.csv")
    if raw is None:
        raw = _latest_by_glob(raw_dir, "PBJ_dailyNonnurse*.csv")
    std = _latest_by_glob(std_dir, "PBJ_dailynonnurse*.csv")
    label = None
    if std:
        label = _quarter_label_from_name(std.name)
    elif raw:
        label = _quarter_label_from_name(raw.name)

    if raw and not std:
        status, detail = (
            OpsStatus.PROCESSING_REQUIRED.value,
            "Raw non-nurse CSV present; standardized output missing",
        )
    elif not raw and not std:
        status, detail = OpsStatus.UNKNOWN.value, "No local non-nurse PBJ files found"
    else:
        status, detail = (
            OpsStatus.UNKNOWN.value,
            "Local non-nurse files present; CMS currency unknown (no dataset ID). "
            "Acquire CLI broken-legacy (missing ingest script).",
        )

    return SourceOpsSnapshot(
        source_id=record.source_id,
        human_name=record.human_name,
        source_family=record.source_family.value,
        formats=[f.value for f in record.formats],
        cadence=record.cadence.value,
        automation_level=record.automation_level.value,
        cms_dataset_id=None,
        cms_latest=None,
        pbjapp_latest=label,
        status=status,
        last_checked=checked,
        last_successful_local_processing=_mtime_iso(std) if std else None,
        local_raw_present=bool(raw),
        local_derived_present=bool(std),
        detail=detail,
        actions_enabled=[],
    )


def _probe_ein(record: CmsSourceRecord, root: Path) -> SourceOpsSnapshot:
    checked = _utc_now_iso()
    mono = cms_data_paths.ein_monolithic_dir(root)
    quarters = cms_data_paths.ein_quarters_dir(root)
    extracted = cms_data_paths.ein_extracted_dir(root)
    mono_zips = list(mono.glob("*.zip")) if mono.is_dir() else []
    q_zips = list(quarters.glob("*.zip")) if quarters.is_dir() else []
    # Also check legacy EIN/*.zip at root of EIN
    ein_root = cms_data_paths.ein_root(root)
    root_zips = list(ein_root.glob("*.zip")) if ein_root.is_dir() else []
    raw_present = bool(mono_zips or q_zips or root_zips)
    derived = _latest_by_glob(extracted, "CY*.csv") if extracted.is_dir() else None

    label = None
    if q_zips:
        labels = [_quarter_label_from_name(z.name) for z in q_zips]
        labels = [x for x in labels if x]
        if labels:
            label = max(labels)
    if label is None and mono_zips:
        label = "monolithic PUF present"
    elif label is None and root_zips:
        label = root_zips[0].name

    if raw_present and not derived:
        status = OpsStatus.LOCAL_RAW_ONLY.value
        detail = "EIN zip(s) present; extracted national CSV not found (broken-legacy ingest)"
    elif not raw_present:
        status = OpsStatus.UNKNOWN.value
        detail = "No local EIN zips found"
    else:
        status = OpsStatus.UNKNOWN.value
        detail = "Local EIN artifacts present; CMS currency unknown (no dataset ID)"

    return SourceOpsSnapshot(
        source_id=record.source_id,
        human_name=record.human_name,
        source_family=record.source_family.value,
        formats=[f.value for f in record.formats],
        cadence=record.cadence.value,
        automation_level=record.automation_level.value,
        cms_dataset_id=None,
        cms_latest=None,
        pbjapp_latest=label,
        status=status,
        last_checked=checked,
        last_successful_local_processing=_mtime_iso(derived) if derived else None,
        local_raw_present=raw_present,
        local_derived_present=bool(derived),
        detail=detail,
        actions_enabled=[],
    )


def _probe_snf_all_owners(record: CmsSourceRecord, root: Path) -> SourceOpsSnapshot:
    checked = _utc_now_iso()
    own = cms_data_paths.ownership_dir(root)
    raw = _latest_by_glob(own, "SNF_All_Owners*.csv")
    if raw:
        status = OpsStatus.LOCAL_RAW_ONLY.value
        detail = "SNF_All_Owners CSV present; no normalize/index scripts on main"
        label = raw.name
    else:
        status = OpsStatus.UNKNOWN.value
        detail = "No SNF_All_Owners*.csv in ownership/"
        label = None
    return SourceOpsSnapshot(
        source_id=record.source_id,
        human_name=record.human_name,
        source_family=record.source_family.value,
        formats=[f.value for f in record.formats],
        cadence=record.cadence.value,
        automation_level=record.automation_level.value,
        cms_dataset_id=None,
        cms_latest=None,
        pbjapp_latest=label,
        status=status,
        last_checked=checked,
        last_successful_local_processing=_mtime_iso(raw) if raw else None,
        local_raw_present=bool(raw),
        local_derived_present=False,
        detail=detail,
        actions_enabled=[],
    )


def _probe_placeholder(record: CmsSourceRecord, detail: str) -> SourceOpsSnapshot:
    return SourceOpsSnapshot(
        source_id=record.source_id,
        human_name=record.human_name,
        source_family=record.source_family.value,
        formats=[f.value for f in record.formats],
        cadence=record.cadence.value,
        automation_level=record.automation_level.value,
        cms_dataset_id=record.cms_dataset_id,
        cms_latest=None,
        pbjapp_latest=None,
        status=OpsStatus.UNKNOWN.value,
        last_checked=_utc_now_iso(),
        last_successful_local_processing=None,
        detail=detail,
        actions_enabled=[],
    )


def _probe_chain(record: CmsSourceRecord, root: Path) -> SourceOpsSnapshot:
    checked = _utc_now_iso()
    own = cms_data_paths.ownership_dir(root)
    candidates: list[Path] = []
    if own.is_dir():
        candidates.extend(own.glob("Nursing_Home_Chain_Performance_Measures_*.csv"))
        candidates.extend(own.glob("Nursing_Home_Affiliated_Entity_Performance_Measures_*.csv"))
    raw = max(candidates, key=_chain_sort_key) if candidates else None
    label = _chain_label_from_name(raw.name) if raw else None
    if raw:
        status = OpsStatus.LOCAL_RAW_ONLY.value
        detail = "Chain performance CSV present; manual drop / detection only"
    else:
        status = OpsStatus.UNKNOWN.value
        detail = "No chain performance CSV in ownership/"
    return SourceOpsSnapshot(
        source_id=record.source_id,
        human_name=record.human_name,
        source_family=record.source_family.value,
        formats=[f.value for f in record.formats],
        cadence=record.cadence.value,
        automation_level=record.automation_level.value,
        cms_dataset_id=None,
        cms_latest=None,
        pbjapp_latest=label or (raw.name if raw else None),
        status=status,
        last_checked=checked,
        last_successful_local_processing=_mtime_iso(raw) if raw else None,
        local_raw_present=bool(raw),
        local_derived_present=False,
        detail=detail,
        actions_enabled=[],
    )


def _probe_sff(record: CmsSourceRecord, root: Path, *, check_cms: bool, fetch_json: FetchJson | None) -> SourceOpsSnapshot:
    """SFF tracks Provider Info column; PDF list not ingested on main."""
    pi = _probe_provider_info(
        get_source("cms.provider_info"),  # type: ignore[arg-type]
        check_cms=check_cms,
        fetch_json=fetch_json,
        root=root,
    )
    return SourceOpsSnapshot(
        source_id=record.source_id,
        human_name=record.human_name,
        source_family=record.source_family.value,
        formats=[f.value for f in record.formats],
        cadence=record.cadence.value,
        automation_level=record.automation_level.value,
        cms_dataset_id=None,
        cms_latest=pi.cms_latest,
        pbjapp_latest=pi.pbjapp_latest,
        status=pi.status if pi.status != OpsStatus.READY_FOR_HANDOFF.value else OpsStatus.CURRENT.value,
        last_checked=pi.last_checked,
        last_successful_local_processing=pi.last_successful_local_processing,
        local_raw_present=pi.local_raw_present,
        local_derived_present=pi.local_derived_present,
        detail=(
            "SFF via Provider Info Special Focus Status column. "
            "PDF container supported in registry; no separate PDF ingest on main. "
            + (pi.detail or "")
        ),
        actions_enabled=[],
        error=pi.error,
    )


def probe_source(
    source_id: str,
    *,
    check_cms: bool = True,
    fetch_json: FetchJson | None = None,
    root: Path | None = None,
) -> SourceOpsSnapshot:
    root = root or cms_data_paths.repo_root()
    record = get_source(source_id)
    if record is None:
        return SourceOpsSnapshot(
            source_id=source_id,
            human_name=source_id,
            source_family="unknown",
            formats=[],
            cadence="unknown",
            automation_level="fully_manual",
            cms_dataset_id=None,
            cms_latest=None,
            pbjapp_latest=None,
            status=OpsStatus.ERROR.value,
            last_checked=_utc_now_iso(),
            last_successful_local_processing=None,
            detail="Unknown source_id",
            error=f"unknown source_id: {source_id}",
        )

    family = record.source_family
    if family == SourceFamily.PROVIDER_INFO:
        return _probe_provider_info(record, check_cms=check_cms, fetch_json=fetch_json, root=root)
    if family == SourceFamily.PBJ_NURSE:
        return _probe_nurse(record, root)
    if family == SourceFamily.PBJ_NON_NURSE:
        return _probe_nonnurse(record, root)
    if family == SourceFamily.PBJ_EIN:
        return _probe_ein(record, root)
    if family == SourceFamily.SNF_ALL_OWNERS:
        return _probe_snf_all_owners(record, root)
    if family == SourceFamily.SNF_ENROLLMENTS:
        return _probe_placeholder(
            record,
            "No standalone SNF Enrollments drop verified on main",
        )
    if family == SourceFamily.SNF_CHOW:
        chow_doc = root / "ownership" / "_sources" / "cms_chow"
        detail = (
            f"Documented path exists locally: {chow_doc}"
            if chow_doc.is_dir()
            else "Documented ownership/_sources/cms_chow/ missing on main; fully manual"
        )
        return _probe_placeholder(record, detail)
    if family == SourceFamily.CHAIN_PERFORMANCE:
        return _probe_chain(record, root)
    if family == SourceFamily.SFF:
        return _probe_sff(record, root, check_cms=check_cms, fetch_json=fetch_json)
    return _probe_placeholder(record, "Unhandled source family")


def probe_all_sources(
    *,
    check_cms: bool = True,
    fetch_json: FetchJson | None = None,
    root: Path | None = None,
) -> list[SourceOpsSnapshot]:
    return [
        probe_source(r.source_id, check_cms=check_cms, fetch_json=fetch_json, root=root)
        for r in get_registry()
    ]


def check_provider_info_cms(
    *,
    fetch_json: FetchJson | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Refresh/check CMS for Provider Info via canonical acquire resolve (no download)."""
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
        "dry_run": acq.acquire_and_process(
            root=root, fetch_json=fetch_json, dry_run=True
        ),
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
    """Acquire/process current Provider Info via PR #63 machinery only."""
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


def recommended_next_automation() -> dict[str, str]:
    """Heuristic: next family to automate after Provider Info pilot."""
    return {
        "source_id": "cms.pbj_nurse_staffing",
        "human_name": "PBJ nurse staffing",
        "why": (
            "Highest-volume core product input with existing detection + "
            "standardization on main, a documented CMS landing URL, and no "
            "broken missing-script acquire path (unlike non-nurse/EIN). "
            "Automating CMS quarter discovery/download would close the largest "
            "manual gap without first repairing broken-legacy CLI stubs."
        ),
    }
