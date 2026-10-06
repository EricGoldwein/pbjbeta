"""Keep Provider Information processing months mapped to PBJ quarters.

Dashboards resolve stars/charts from processing month, not the CMS ``quarter``
CSV label. New monthly extracts must update the shipped interval JSON (and
fail closed when neither the extract nor the manual map can resolve a month).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cms_data_paths
from active_release_registry import get_active_release, registry_path
from prov_info_quarter_map import get_manual_quarter_from_processing_month, get_quarter_from_processing_month

_MONTH_ABBR = (
    "",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def processing_months_from_provider_csv(path: Path) -> set[str]:
    """Unique YYYY-MM processing months in a facility provider slice."""
    months: set[str] = set()
    if not path.is_file():
        return months
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for rec in csv.DictReader(handle):
            key = _month_key(str(rec.get("processing_month") or rec.get("processing_date") or ""))
            if key:
                months.add(key)
    return months


def _month_key(raw: str) -> str | None:
    s = str(raw or "").strip()
    if len(s) >= 7 and s[4] == "-":
        return s[:7]
    token = s.split()[0] if s else ""
    for fmt in ("%m/%d/%Y", "%Y/%m/%d", "%m/%d/%y", "%Y%m%d"):
        try:
            dt = datetime.strptime(token, fmt)
            return f"{dt.year:04d}-{dt.month:02d}"
        except ValueError:
            continue
    return None


def _parse_yyyy_mm(value: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"(20\d{2})-(\d{2})", str(value or "").strip())
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    if month < 1 or month > 12:
        return None
    return year, month


def interval_csv_path(release_id: str, *, root: Path | None = None) -> Path:
    parsed = _parse_yyyy_mm(release_id)
    if not parsed:
        raise ValueError(f"invalid provider processing month: {release_id!r}")
    year, month = parsed
    name = f"NH_DataCollectionIntervals_{_MONTH_ABBR[month]}{year}.csv"
    return cms_data_paths.provider_info_dir(root) / name


def _quarter_from_mdy(mdy: str) -> str:
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", str(mdy or "").strip())
    if not m:
        return ""
    month, year = int(m.group(1)), int(m.group(3))
    q = ((month - 1) // 3) + 1
    return f"Q{q} {year}"


def quarter_from_interval_csv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    from scripts.cms_provider_release_lib import staffing_interval_from_interval_csv

    meta = staffing_interval_from_interval_csv(path)
    q = _quarter_from_mdy(meta.get("staffing_level_from") or "")
    return {**meta, "staffing_level_quarter": q}


def interval_json_targets(root: Path | None = None) -> list[Path]:
    ops_root = Path(__file__).resolve().parent
    canonical = cms_data_paths.repo_root()
    explicit = Path(root).resolve() if root is not None else None
    if explicit is not None and explicit not in {canonical, ops_root}:
        return [explicit / "static" / "data" / "interval_quarter_mapping.json"]
    out: list[Path] = []
    for path in (
        canonical / "static" / "data" / "interval_quarter_mapping.json",
        ops_root / "static" / "data" / "interval_quarter_mapping.json",
    ):
        if path not in out and (path.is_file() or path.parent.is_dir()):
            out.append(path)
    return out


def mapping_row_from_extract(release_id: str, *, root: Path | None = None) -> dict[str, Any] | None:
    parsed = _parse_yyyy_mm(release_id)
    if not parsed:
        return None
    year, month = parsed
    path = interval_csv_path(release_id, root=root)
    meta = quarter_from_interval_csv(path)
    quarter = str(meta.get("staffing_level_quarter") or get_manual_quarter_from_processing_month(release_id) or "")
    if not quarter:
        return None
    abbr = _MONTH_ABBR[month]
    return {
        "processing_month": f"{month:02d}-{year}",
        "provider_info_csv_name": f"NH_ProviderInfo_{abbr}{year}.csv",
        "provider_info_download_url": f"/api/provider-info/download?file=NH_ProviderInfo_{abbr}{year}.csv",
        "interval_csv_name": path.name if path.is_file() else "",
        "staffing_level_from": meta.get("staffing_level_from") or "",
        "staffing_level_through": meta.get("staffing_level_through") or "",
        "interval_staffing_level_quarter": quarter,
        "manual_processing_quarter": get_manual_quarter_from_processing_month(release_id) or "",
        "used_case_mix_quarter": quarter,
        "turnover_from": meta.get("turnover_from") or "",
        "turnover_through": meta.get("turnover_through") or "",
        "turnover_quarter": "",
        "pbj_quarter_url": (
            "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing/data/"
            + quarter.lower().replace(" ", "-")
        ),
    }


def json_has_processing_month(path: Path, release_id: str) -> bool:
    parsed = _parse_yyyy_mm(release_id)
    if not parsed or not path.is_file():
        return False
    year, month = parsed
    key = f"{month:02d}-{year}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return False
    return any(str(r.get("processing_month") or "").strip() == key for r in rows if isinstance(r, dict))


def upsert_interval_json_row(path: Path, row: dict[str, Any]) -> bool:
    key = str(row.get("processing_month") or "").strip()
    if not key:
        return False
    data: dict[str, Any] = {"rows": []}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict) and isinstance(loaded.get("rows"), list):
            data = loaded
    rows = [r for r in (data.get("rows") or []) if isinstance(r, dict) and str(r.get("processing_month") or "").strip() != key]
    rows.insert(0, row)
    data["rows"] = rows
    data["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    try:
        import prov_info_quarter_map as quarter_map

        quarter_map._INTERVAL_JSON_ROWS = None
    except Exception:
        pass
    return True


def sync_interval_mapping_from_extract(
    release_id: str | None = None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Write the extracted interval quarter into shipped interval_quarter_mapping.json files."""
    root = root or cms_data_paths.repo_root()
    if not release_id:
        active = get_active_release("cms.provider_info", registry_path(root)) or {}
        release_id = str(active.get("active_release_id") or "")
    if not release_id:
        return {"ok": False, "detail": "No ACTIVE Provider Information release.", "updated": []}
    row = mapping_row_from_extract(release_id, root=root)
    if not row:
        interval = interval_csv_path(release_id, root=root)
        if not interval.is_file():
            return {
                "ok": False,
                "release_id": release_id,
                "detail": (
                    f"Provider Information {release_id} has no interval extract on disk, "
                    "so it cannot be mapped to a PBJ quarter."
                ),
                "updated": [],
            }
        return {
            "ok": False,
            "release_id": release_id,
            "detail": f"Could not derive a PBJ quarter from {interval.name}.",
            "updated": [],
        }
    updated: list[str] = []
    for path in interval_json_targets(root):
        if upsert_interval_json_row(path, row):
            updated.append(str(path))
    return {
        "ok": True,
        "release_id": release_id,
        "quarter": row.get("interval_staffing_level_quarter"),
        "updated": updated,
        "detail": (
            f"Mapped Provider Information {release_id} to "
            f"{row.get('interval_staffing_level_quarter')} from the interval extract."
        ),
    }


def audit_active_provider_quarter_mapping(*, root: Path | None = None) -> dict[str, Any]:
    """Operator status: can ACTIVE Provider Info resolve to a PBJ quarter, and is JSON shipped?"""
    root = root or cms_data_paths.repo_root()
    active = get_active_release("cms.provider_info", registry_path(root)) or {}
    release_id = str(active.get("active_release_id") or "")
    if not release_id:
        return {"needs_attention": False, "release_id": "", "resolved_quarter": None}
    resolved = get_quarter_from_processing_month(release_id)
    interval = interval_csv_path(release_id, root=root)
    json_paths = [p for p in interval_json_targets(root) if p.is_file()]
    json_missing = [str(p) for p in json_paths if not json_has_processing_month(p, release_id)]
    can_sync = mapping_row_from_extract(release_id, root=root) is not None
    needs = (not resolved) or bool(json_missing)
    if not resolved:
        detail = (
            f"Provider Information {release_id} is ACTIVE but is not mapped to a PBJ quarter. "
            "Stars and provider charts on facility dashboards will be empty until the interval extract is applied."
        )
        label = "Apply extracted quarter map" if can_sync else "Extract Provider Information intervals"
    elif json_missing:
        detail = (
            f"Provider Information {release_id} maps to {resolved}, but the shipped quarter table "
            "does not include that month. Apply the extract so local dashboards pick it up."
        )
        label = "Apply extracted quarter map"
    else:
        detail = f"Provider Information {release_id} maps to {resolved}."
        label = ""
    return {
        "needs_attention": needs,
        "release_id": release_id,
        "resolved_quarter": resolved,
        "interval_present": interval.is_file(),
        "can_sync": can_sync and needs,
        "json_missing": json_missing,
        "detail": detail,
        "action_label": label,
    }


PROVE_COMMAND = "python scripts/check_provider_quarter_flow.py --prove"


def _pbjapp_root_for_bundles(explicit: Path | None) -> Path:
    """Facility bundles live in PBJapp, not the Data Ops control-plane repo."""
    if explicit is not None:
        return Path(explicit)
    env = (os.environ.get("PBJ_REPO_ROOT") or "").strip().strip('"')
    if env:
        return Path(env).expanduser().resolve()
    sibling = Path(__file__).resolve().parent.parent / "PBJapp"
    if (sibling / "create_vercel_deployment.py").is_file():
        return sibling.resolve()
    return cms_data_paths.repo_root()


def operator_quarter_flow(ccn: str | None = None, *, root: Path | None = None) -> dict[str, Any]:
    """One connected status for extract → map → facility history → prove."""
    audit = audit_active_provider_quarter_mapping(root=root)
    history: dict[str, Any] = {"ready": True, "detail": "No facility selected."}
    if ccn:
        ccn_n = str(ccn).zfill(6)
        # Facility bundles under PBJ_DATA_ROOT — never PBJapp deployments stub.
        deploy = cms_data_paths.facility_deploy_dir(
            ccn_n, cms_data_paths.optional_repo_root(root)
        )
        path = deploy / f"facility_{ccn_n}_provider_info_data.csv"
        if not deploy.is_dir() or not path.is_file():
            history = {
                "ready": False,
                "detail": "No local dashboard bundle yet. Build it after the quarter map is current.",
            }
        else:
            months = processing_months_from_provider_csv(path)
            if len(months) >= 6:
                history = {
                    "ready": True,
                    "detail": f"Provider Information history has {len(months)} months.",
                }
            else:
                history = {
                    "ready": False,
                    "detail": (
                        f"Provider Information for this facility only has {len(months)} month(s) of stars history. "
                        "Rebuild the local dashboard so it pulls every extracted month, not just the latest file."
                    ),
                }
    mapped = not audit.get("needs_attention")
    ready = mapped and history.get("ready") is True
    next_step = "Nothing pending."
    if not mapped:
        next_step = audit.get("action_label") or "Apply extracted quarter map"
    elif not history.get("ready"):
        next_step = "Rebuild the local dashboard"
    return {
        **audit,
        "history": history,
        "ready": ready,
        "next_step": next_step,
        "prove_command": PROVE_COMMAND,
        "steps": [
            "Acquire / Process Provider Information (writes the interval extract).",
            "Apply extracted quarter map so stars/charts know which PBJ quarter that month is.",
            "Rebuild each local dashboard so it keeps full Provider Information history, not one month.",
            f"Prove with: {PROVE_COMMAND}",
        ],
    }
