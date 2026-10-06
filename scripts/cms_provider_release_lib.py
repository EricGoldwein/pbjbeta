#!/usr/bin/env python3
"""Reproducible CMS monthly provider-information release extraction and manifests."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import cms_data_paths
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
# Logical destination keys → filename prefix (month/year appended at runtime).
_ACTIVE_CSV_DEST: tuple[tuple[str, str], ...] = (
    ("NH_ProviderInfo_", "provider_info"),
    ("NH_DataCollectionIntervals_", "provider_info"),
    ("NH_Ownership_", "ownership"),
    ("NH_HealthCitations_", "citations"),
    ("NH_CitationDescriptions_", "citations"),
)
_CLASSIFICATION_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^NH_ProviderInfo_", re.I), "provider_info"),
    (re.compile(r"^NH_DataCollectionIntervals_", re.I), "intervals"),
    (re.compile(r"^NH_Ownership_", re.I), "ownership"),
    (re.compile(r"^NH_HealthCitations_", re.I), "citations"),
    (re.compile(r"^NH_CitationDescriptions_", re.I), "citations"),
    (re.compile(r"^NH_FireSafetyCitations_", re.I), "citations"),
    (re.compile(r"^NH_Chain", re.I), "chain"),
    (re.compile(r"Chain.*Performance", re.I), "chain"),
    (re.compile(r"QM|Quality_Meas|QualityMsr", re.I), "quality_measures"),
    (re.compile(r"StateUSAverages", re.I), "quality_measures"),
    (re.compile(r"HlthInspecCutpoints", re.I), "surveys"),
    (re.compile(r"Survey", re.I), "surveys"),
    (re.compile(r"Penalt", re.I), "penalties"),
    (re.compile(r"VBP", re.I), "vbp"),
    (re.compile(r"SNF.*QRP|QRP|Quality_Reporting_Program", re.I), "snf_qrp"),
    (re.compile(r"Swing", re.I), "swing_bed"),
    (re.compile(r"Data_Dictionary", re.I), "documentation"),
    (re.compile(r"^readme\.", re.I), "documentation"),
)
@dataclass(frozen=True)
class ReleaseKey:
    year: int
    month: int
    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"
    @property
    def month_abbr(self) -> str:
        if self.month < 1 or self.month > 12:
            raise ValueError(f"invalid month: {self.month}")
        return _MONTH_ABBR[self.month]
def release_key(year: int, month: int) -> ReleaseKey:
    return ReleaseKey(int(year), int(month))
def prior_release_key(key: ReleaseKey) -> Optional[ReleaseKey]:
    if key.month > 1:
        return ReleaseKey(key.year, key.month - 1)
    return None
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
def normalize_column_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())
def schema_fingerprint(columns: list[str]) -> str:
    normalized = sorted({normalize_column_name(c) for c in columns if str(c).strip()})
    payload = "\n".join(normalized).encode("utf-8")
    return _sha256_bytes(payload)
def classify_member_basename(basename: str) -> tuple[Optional[str], str]:
    """Return (dataset_family, dataset_type). dataset_type mirrors family for CSV members."""
    for pattern, family in _CLASSIFICATION_RULES:
        if pattern.search(basename):
            return family, family
    if basename.lower().endswith(".csv"):
        return None, "unknown_csv"
    ext = Path(basename).suffix.lower()
    if ext in {".txt", ".pdf", ".xlsx", ".xls"}:
        return None, f"non_csv{ext}"
    return None, "unknown"
def is_active_extract_basename(key: ReleaseKey, basename: str) -> bool:
    return basename in expected_active_csv_names(key)
def _ingestion_status_for_member(
    key: ReleaseKey,
    basename: str,
    family: Optional[str],
    *,
    prior_member: Optional[dict[str, Any]],
) -> str:
    if is_active_extract_basename(key, basename):
        return "ingested"
    if family is not None:
        return "retained_unmodeled"
    if not basename.lower().endswith(".csv"):
        return "retained_unmodeled"
    if prior_member:
        return "retained_unmodeled"
    return "unmapped_new_source"
def _member_coverage(entry: dict[str, Any]) -> dict[str, Any]:
    cov: dict[str, Any] = {}
    if "row_count" in entry:
        cov["row_count"] = entry["row_count"]
    if "column_count" in entry:
        cov["column_count"] = entry["column_count"]
    pk = entry.get("primary_key_columns") or []
    if pk:
        cov["primary_key_columns"] = pk
    cols = entry.get("columns") or []
    ccn_cols = [c for c in cols if "CCN" in str(c)]
    if ccn_cols:
        cov["ccn_columns"] = ccn_cols
    return cov
def _canonical_member_path(member_name: str) -> str:
    return Path(member_name).name
def _normalized_outputs_for_member(
    key: ReleaseKey,
    basename: str,
    logical_dest: str,
    root: Path | None,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    if logical_dest == "provider_info" and basename.startswith("NH_ProviderInfo_"):
        norm = (
            cms_data_paths.provider_info_normalized_dir(root)
            / f"ProviderInfoNorm_{key.year}_{key.month:02d}.csv"
        )
        if norm.is_file():
            outputs.append(
                {
                    "artifact": "ProviderInfoNorm",
                    "path": str(norm),
                    "sha256": _sha256_file(norm),
                    "size_bytes": norm.stat().st_size,
                }
            )
    dest_path = _extracted_path_for_basename(basename, logical_dest, root)
    if dest_path and dest_path.is_file():
        outputs.append(
            {
                "artifact": basename,
                "path": str(dest_path),
                "sha256": _sha256_file(dest_path),
                "size_bytes": dest_path.stat().st_size,
            }
        )
    return outputs
def _extracted_path_for_basename(basename: str, logical_dest: str, root: Path | None) -> Optional[Path]:
    try:
        dest_dir = _dest_dir(logical_dest, root)
    except KeyError:
        return None
    path = dest_dir / basename
    return path if path.is_file() else None
def build_source_members(
    key: ReleaseKey,
    inner_inventory: list[dict[str, Any]],
    extracted: list[dict[str, Any]],
    *,
    prior_manifest: Optional[dict[str, Any]] = None,
    retained_extracted: Optional[dict[str, dict[str, Any]]] = None,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    prior_by_basename = {
        m.get("basename"): m
        for m in (prior_manifest or {}).get("source_members", [])
        if m.get("basename")
    }
    extracted_by_basename = {e["basename"]: e for e in extracted}
    logical_by_basename = expected_active_csv_names(key)
    members: list[dict[str, Any]] = []
    for entry in inner_inventory:
        basename = entry.get("basename") or _canonical_member_path(entry.get("member_name", ""))
        family, dataset_type = classify_member_basename(basename)
        prior_member = prior_by_basename.get(basename)
        ingestion_status = _ingestion_status_for_member(
            key, basename, family, prior_member=prior_member
        )
        columns = [str(c) for c in (entry.get("columns") or [])]
        fingerprint = schema_fingerprint(columns) if columns else ""
        logical_dest = logical_by_basename.get(basename, "")
        norm_outputs: list[dict[str, Any]] = []
        if ingestion_status == "ingested" and logical_dest:
            norm_outputs = _normalized_outputs_for_member(key, basename, logical_dest, root)
        policy = classification_policy(family, basename, ingestion_status)
        retained_rec = (retained_extracted or {}).get(basename)
        member: dict[str, Any] = {
            "canonical_member_path": _canonical_member_path(entry.get("member_name", basename)),
            "member_name": entry.get("member_name", basename),
            "basename": basename,
            "dataset_family": family or "unclassified",
            "dataset_type": dataset_type,
            "classification_policy": policy,
            "source_sha256": entry.get("sha256", ""),
            "size_bytes": entry.get("size_bytes", 0),
            "schema_fingerprint": fingerprint,
            "columns": columns,
            "row_count": entry.get("row_count"),
            "coverage": _member_coverage(entry),
            "ingestion_status": ingestion_status,
            "normalized_outputs": norm_outputs,
        }
        if entry.get("content_type"):
            member["content_type"] = entry["content_type"]
        if entry.get("text_fingerprint"):
            member["text_fingerprint"] = entry["text_fingerprint"]
        if retained_rec:
            base = root or cms_data_paths.repo_root()
            rpath = Path(str(retained_rec.get("retained_path", "")))
            if rpath.is_file():
                try:
                    member["retained_artifact"] = {
                        "path": str(rpath.relative_to(base)),
                        "sha256": retained_rec.get("sha256"),
                        "size_bytes": retained_rec.get("size_bytes"),
                        "action": retained_rec.get("action"),
                    }
                except ValueError:
                    member["retained_artifact"] = {
                        "path": str(rpath),
                        "sha256": retained_rec.get("sha256"),
                        "size_bytes": retained_rec.get("size_bytes"),
                        "action": retained_rec.get("action"),
                    }
        if entry.get("inventory_error"):
            member["inventory_error"] = entry["inventory_error"]
        if basename in extracted_by_basename:
            member["extract_action"] = extracted_by_basename[basename].get("action")
        members.append(member)
    return members
def outer_yearly_archive_path(key: ReleaseKey, root: Path | None = None) -> Path:
    pi = cms_data_paths.provider_info_dir(root)
    outer = pi / f"nursing_homes_including_rehab_services_{key.year}.zip"
    if outer.is_file():
        return outer
    if key.year <= 2019:
        legacy = pi / f"nh_archive_{key.year}.zip"
        if legacy.is_file():
            return legacy
    return outer
def inner_archive_member_name(key: ReleaseKey) -> str:
    if key.year <= 2019:
        return f"nh_archive_{key.month:02d}_{key.year}.zip"
    return f"nursing_homes_including_rehab_services_{key.month:02d}_{key.year}.zip"
def expected_active_csv_names(key: ReleaseKey) -> dict[str, str]:
    """basename -> logical destination key (provider_info, ownership, citations)."""
    mon = key.month_abbr
    suffix = f"{mon}{key.year}.csv"
    return {prefix + suffix: dest for prefix, dest in _ACTIVE_CSV_DEST}
def _dest_dir(logical: str, root: Path | None) -> Path:
    if logical == "provider_info":
        return cms_data_paths.provider_info_dir(root)
    if logical == "ownership":
        return cms_data_paths.ownership_dir(root)
    if logical == "citations":
        return cms_data_paths.citations_dir(root)
    raise KeyError(logical)
def manifest_dir(key: ReleaseKey, root: Path | None = None) -> Path:
    return cms_data_paths.provider_info_dir(root) / "_manifests" / key.label


def retained_release_dir(key: ReleaseKey, root: Path | None = None) -> Path:
    """On-disk copies of classified ``retained_unmodeled`` inner-archive members."""
    return cms_data_paths.provider_info_dir(root) / "_retained" / key.label


def classification_policy(
    family: Optional[str],
    basename: str,
    ingestion_status: str,
) -> str:
    """Explicit promotion policy bucket (not inferred from filename alone)."""
    if ingestion_status == "ingested":
        return "operational_ingested"
    if family == "documentation":
        return "documentation_reference"
    if family in {"quality_measures", "snf_qrp", "vbp", "swing_bed"}:
        return "adjacent_cms_program"
    if family == "surveys" and "Cutpoints" in basename:
        return "documentation_reference"
    if family in {"penalties"}:
        return "adjacent_cms_program"
    if ingestion_status == "retained_unmodeled":
        return "operational_retained"
    if ingestion_status == "unmapped_new_source":
        return "unclassified"
    return "unclassified"
def read_inner_zip_bytes(key: ReleaseKey, root: Path | None = None) -> tuple[Path, str, bytes]:
    outer = outer_yearly_archive_path(key, root)
    if not outer.is_file():
        raise FileNotFoundError(f"yearly CMS archive not found: {outer}")
    inner_name = inner_archive_member_name(key)
    with zipfile.ZipFile(outer, "r") as z_outer:
        try:
            inner_bytes = z_outer.read(inner_name)
        except KeyError as exc:
            raise FileNotFoundError(f"{inner_name} not found inside {outer.name}") from exc
    return outer, inner_name, inner_bytes
def inventory_inner_zip(inner_bytes: bytes) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as z_inner:
        for info in sorted(z_inner.infolist(), key=lambda x: x.filename.lower()):
            name = Path(info.filename).name
            entry: dict[str, Any] = {
                "member_name": info.filename,
                "basename": name,
                "size_bytes": info.file_size,
                "extension": Path(name).suffix.lower(),
            }
            if name.lower().endswith(".csv"):
                raw = z_inner.read(info.filename)
                entry["sha256"] = _sha256_bytes(raw)
                try:
                    text = raw.decode("utf-8", errors="replace")
                    reader = csv.reader(io.StringIO(text))
                    header = next(reader, [])
                    row_count = sum(1 for _ in reader)
                    entry["row_count"] = row_count
                    entry["column_count"] = len(header)
                    entry["columns"] = header
                    entry["schema_fingerprint"] = schema_fingerprint([str(c) for c in header])
                    pk = [c for c in header if "CCN" in c]
                    if pk:
                        entry["primary_key_columns"] = pk
                except Exception as exc:
                    entry["inventory_error"] = str(exc)
            else:
                raw = z_inner.read(info.filename)
                entry["sha256"] = _sha256_bytes(raw)
                ext = name.lower()
                if ext.endswith(".txt"):
                    entry["content_type"] = "text/plain"
                    try:
                        text = raw.decode("utf-8", errors="replace")
                        entry["text_fingerprint"] = _sha256_bytes(text.strip().encode("utf-8"))
                    except Exception as exc:
                        entry["inventory_error"] = str(exc)
                elif ext.endswith(".pdf"):
                    entry["content_type"] = "application/pdf"
            out.append(entry)
    return out


def extract_retained_unmodeled_members(
    key: ReleaseKey,
    inner_bytes: bytes,
    inner_inventory: list[dict[str, Any]],
    *,
    prior_manifest: Optional[dict[str, Any]] = None,
    root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Copy classified ``retained_unmodeled`` members from the inner archive to
    ``provider_info/_retained/YYYY-MM/`` (idempotent on matching SHA-256).
    """
    prior_by = {
        m.get("basename"): m
        for m in (prior_manifest or {}).get("source_members", [])
        if m.get("basename")
    }
    out_dir = retained_release_dir(key, root)
    extracted: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as z_inner:
        available = {Path(n).name: n for n in z_inner.namelist()}
        for entry in inner_inventory:
            basename = str(entry.get("basename") or "")
            if not basename or basename not in available:
                continue
            family, _ = classify_member_basename(basename)
            status = _ingestion_status_for_member(
                key,
                basename,
                family,
                prior_member=prior_by.get(basename),
            )
            if status != "retained_unmodeled":
                continue
            member_path = available[basename]
            raw = z_inner.read(member_path)
            digest = _sha256_bytes(raw)
            policy = classification_policy(family, basename, status)
            if policy == "documentation_reference" and basename.lower().endswith((".pdf", ".txt")):
                rel_dest = Path("documentation") / basename
            else:
                rel_dest = Path(basename)
            dest = out_dir / rel_dest
            action = "dry_run"
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.is_file() and _sha256_file(dest) == digest:
                    action = "skip_identical"
                else:
                    dest.write_bytes(raw)
                    action = "written"
            rec: dict[str, Any] = {
                "basename": basename,
                "relative_path": str(dest.relative_to(out_dir.parent.parent) if not dry_run and dest.is_file() else out_dir / rel_dest),
                "retained_path": str(dest),
                "sha256": digest,
                "size_bytes": len(raw),
                "action": action,
                "classification_policy": policy,
            }
            if entry.get("row_count") is not None:
                rec["row_count"] = entry["row_count"]
            if entry.get("schema_fingerprint"):
                rec["schema_fingerprint"] = entry["schema_fingerprint"]
            extracted[basename] = rec
    return extracted
def inventory_release(key: ReleaseKey, root: Path | None = None) -> dict[str, Any]:
    """Dry-run inventory of a monthly inner archive (no extraction)."""
    outer, inner_name, inner_bytes = read_inner_zip_bytes(key, root)
    inner_inventory = inventory_inner_zip(inner_bytes)
    prior = find_prior_successful_manifest(key, root)
    source_members = build_source_members(
        key, inner_inventory, [], prior_manifest=prior, root=root
    )
    return {
        "release_key": key.label,
        "inventory_at": datetime.now(timezone.utc).isoformat(),
        "outer_archive": {
            "path": str(outer),
            "basename": outer.name,
            "size_bytes": outer.stat().st_size,
            "sha256": _sha256_file(outer),
        },
        "inner_archive": {
            "member_name": inner_name,
            "size_bytes": len(inner_bytes),
            "sha256": _sha256_bytes(inner_bytes),
        },
        "source_members": source_members,
        "member_count": len(source_members),
    }
def _csv_stats_from_path(path: Path) -> dict[str, Any]:
    df_head = pd.read_csv(path, nrows=0, low_memory=False)
    cols = [str(c) for c in df_head.columns]
    ccn_col = next((c for c in cols if "CCN" in c), None)
    row_count = sum(1 for _ in path.open(encoding="utf-8", errors="replace")) - 1
    stats: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "row_count": row_count,
        "column_count": len(cols),
        "columns": cols,
        "schema_fingerprint": schema_fingerprint(cols),
    }
    if ccn_col and row_count > 0:
        df = pd.read_csv(path, usecols=[ccn_col], dtype=str, low_memory=False)
        ccn = df[ccn_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
        stats["unique_ccn"] = int(ccn.nunique())
        stats["duplicate_ccn_rows"] = int(ccn.duplicated().sum())
        stats["primary_key_column"] = ccn_col
    return stats
def extract_active_csvs(
    key: ReleaseKey,
    *,
    root: Path | None = None,
    dry_run: bool = False,
    force_identical: bool = False,
) -> dict[str, Any]:
    """
    Extract active pipeline CSVs from the nested monthly archive.
    Refuses to overwrite an existing file with different content unless
    ``force_identical`` is True (used only in tests).
    """
    outer, inner_name, inner_bytes = read_inner_zip_bytes(key, root)
    expected = expected_active_csv_names(key)
    missing = []
    extracted: list[dict[str, Any]] = []
    skipped: list[str] = []
    with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as z_inner:
        available = {Path(n).name: n for n in z_inner.namelist()}
        for basename, logical in expected.items():
            if basename not in available:
                missing.append(basename)
                continue
            member = available[basename]
            raw = z_inner.read(member)
            digest = _sha256_bytes(raw)
            dest_dir = _dest_dir(logical, root)
            dest_dir.mkdir(parents=True, exist_ok=True)
            out_path = dest_dir / basename
            if out_path.is_file():
                existing = _sha256_file(out_path)
                if existing == digest:
                    skipped.append(basename)
                    extracted.append(
                        {
                            "basename": basename,
                            "destination": str(out_path),
                            "logical_dest": logical,
                            "action": "skip_identical",
                            "sha256": digest,
                        }
                    )
                    continue
                if not force_identical:
                    raise FileExistsError(
                        f"refusing to overwrite non-identical file: {out_path} "
                        f"(existing sha256={existing}, archive sha256={digest})"
                    )
            if dry_run:
                extracted.append(
                    {
                        "basename": basename,
                        "destination": str(out_path),
                        "logical_dest": logical,
                        "action": "dry_run",
                        "sha256": digest,
                    }
                )
                continue
            out_path.write_bytes(raw)
            stats = _csv_stats_from_path(out_path)
            extracted.append(
                {
                    "basename": basename,
                    "destination": str(out_path),
                    "logical_dest": logical,
                    "action": "written",
                    "sha256": digest,
                    **{k: v for k, v in stats.items() if k not in ("path",)},
                }
            )
    if missing:
        raise FileNotFoundError(f"expected members missing from inner archive: {missing}")
    inner_inventory = inventory_inner_zip(inner_bytes)
    prior_manifest = find_prior_successful_manifest(key, root)
    retained_extracted = extract_retained_unmodeled_members(
        key,
        inner_bytes,
        inner_inventory,
        prior_manifest=prior_manifest,
        root=root,
        dry_run=dry_run,
    )
    manifest = build_release_manifest(
        key,
        outer_path=outer,
        inner_member=inner_name,
        inner_bytes=inner_bytes,
        inner_inventory=inner_inventory,
        extracted=extracted,
        skipped=skipped,
        prior_manifest=prior_manifest,
        retained_extracted=retained_extracted,
        root=root,
    )
    diff = build_release_diff(key, manifest, prior_manifest, root=root)
    blocked, reasons = compute_promotion_blocked(manifest, diff, prior_manifest=prior_manifest)
    quarter_map: dict[str, Any] = {}
    if not dry_run:
        ops_root = Path(__file__).resolve().parents[1]
        if str(ops_root) not in sys.path:
            sys.path.insert(0, str(ops_root))
        try:
            from provider_quarter_mapping import sync_interval_mapping_from_extract

            quarter_map = sync_interval_mapping_from_extract(key.label, root=root)
        except Exception as exc:  # noqa: BLE001
            quarter_map = {"ok": False, "detail": str(exc), "updated": []}
        from prov_info_quarter_map import get_quarter_from_processing_month

        if not get_quarter_from_processing_month(key.label):
            blocked = True
            reasons = sorted(
                set(
                    list(reasons)
                    + [
                        (
                            f"Provider Information {key.label} has no PBJ quarter map "
                            "(interval extract or manual processing-month table)."
                        )
                    ]
                )
            )
    manifest["quarter_mapping"] = quarter_map
    manifest["promotion_blocked"] = blocked
    manifest["promotion_blocked_reasons"] = reasons
    diff["promotion_blocked"] = blocked
    diff["promotion_blocked_reasons"] = reasons
    if not dry_run:
        write_manifest(key, manifest, root=root)
        write_release_diff(key, diff, root=root)
    manifest["release_diff"] = diff
    return manifest
def staffing_interval_from_interval_csv(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, dtype=str)
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        code = str(row.get("Measure Code") or "").strip()
        if code in {"STAFFING_LEVELS", "STAFFING"}:
            out["staffing_level_from"] = str(row.get("Data Collection Period From Date") or "").strip()
            out["staffing_level_through"] = str(row.get("Data Collection Period Through Date") or "").strip()
        elif code == "STAFFING_TURNOVER":
            out["turnover_from"] = str(row.get("Data Collection Period From Date") or "").strip()
            out["turnover_through"] = str(row.get("Data Collection Period Through Date") or "").strip()
    proc = df["Processing Date"].dropna().astype(str).iloc[0] if "Processing Date" in df.columns else ""
    out["processing_date"] = proc
    return out
def _quarter_label_from_mdy(mdy: str) -> str:
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", mdy.strip())
    if not m:
        return ""
    month, _, year = int(m.group(1)), m.group(2), int(m.group(3))
    q = ((month - 1) // 3) + 1
    return f"Q{q} {year}"
def build_release_manifest(
    key: ReleaseKey,
    *,
    outer_path: Path,
    inner_member: str,
    inner_bytes: bytes,
    inner_inventory: list[dict[str, Any]],
    extracted: list[dict[str, Any]],
    skipped: list[str],
    prior_manifest: Optional[dict[str, Any]] = None,
    retained_extracted: Optional[dict[str, dict[str, Any]]] = None,
    root: Path | None = None,
) -> dict[str, Any]:
    interval_name = f"NH_DataCollectionIntervals_{key.month_abbr}{key.year}.csv"
    interval_path = cms_data_paths.provider_info_dir(root) / interval_name
    interval_meta: dict[str, Any] = {}
    staffing_q = ""
    if interval_path.is_file():
        interval_meta = staffing_interval_from_interval_csv(interval_path)
        staffing_q = _quarter_label_from_mdy(interval_meta.get("staffing_level_from", ""))
    base = root or cms_data_paths.repo_root()
    source_members = build_source_members(
        key,
        inner_inventory,
        extracted,
        prior_manifest=prior_manifest,
        retained_extracted=retained_extracted,
        root=root,
    )
    return {
        "release_key": key.label,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "provider_processing_month": key.label,
        "staffing_case_mix_reporting_period": {
            "from": interval_meta.get("staffing_level_from", ""),
            "through": interval_meta.get("staffing_level_through", ""),
            "quarter_label": staffing_q,
        },
        "source_note": (
            "CMS provider-information processing month is not a PBJ calendar quarter. "
            f"Processing month {key.label} maps to staffing/case-mix period {staffing_q or 'unknown'}."
        ),
        "outer_archive": {
            "relative_path": str(outer_path.relative_to(base)),
            "basename": outer_path.name,
            "size_bytes": outer_path.stat().st_size,
            "sha256": _sha256_file(outer_path),
        },
        "inner_archive": {
            "member_name": inner_member,
            "size_bytes": len(inner_bytes),
            "sha256": _sha256_bytes(inner_bytes),
        },
        "source_members": source_members,
        "retained_extracted": list(retained_extracted.values()) if retained_extracted else [],
        "inner_inventory": inner_inventory,
        "extracted_active_files": extracted,
        "skipped_identical": skipped,
        "retained_in_yearly_archive_only": [
            e["basename"] for e in inner_inventory if e["basename"] not in expected_active_csv_names(key)
        ],
    }
def update_manifest_normalized_outputs(key: ReleaseKey, root: Path | None = None) -> None:
    """Refresh normalized_outputs on ingested members after normalize step."""
    manifest = load_manifest(key, root)
    if not manifest:
        return
    logical_by_basename = expected_active_csv_names(key)
    for member in manifest.get("source_members", []):
        basename = member.get("basename", "")
        logical = logical_by_basename.get(basename, "")
        if member.get("ingestion_status") == "ingested" and logical:
            member["normalized_outputs"] = _normalized_outputs_for_member(
                key, basename, logical, root
            )
    write_manifest(key, manifest, root=root)
def write_manifest(key: ReleaseKey, manifest: dict[str, Any], root: Path | None = None) -> Path:
    mdir = manifest_dir(key, root)
    mdir.mkdir(parents=True, exist_ok=True)
    out = mdir / "release_manifest.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")
    return out
def write_pbj_root_handoff(key: ReleaseKey, root: Path | None = None) -> Path:
    """Write tracked handoff manifest for pbj-root provider-norm sync."""
    root = root or cms_data_paths.repo_root()
    from prov_info_quarter_map import get_quarter_from_processing_month
    norm_path = (
        cms_data_paths.provider_info_normalized_dir(root)
        / f"ProviderInfoNorm_{key.year}_{key.month:02d}.csv"
    )
    norm_sha = ""
    norm_rows = 0
    if norm_path.is_file():
        norm_sha = _sha256_file(norm_path)
        with norm_path.open(encoding="utf-8", newline="") as f:
            norm_rows = max(0, sum(1 for _ in csv.reader(f)) - 1)
    interval_path = (
        cms_data_paths.provider_info_dir(root)
        / f"NH_DataCollectionIntervals_{key.month_abbr}{key.year}.csv"
    )
    staffing_period: dict[str, str] = {}
    if interval_path.is_file():
        interval = staffing_interval_from_interval_csv(interval_path)
        staffing_period = {
            "from": interval.get("staffing_level_from", ""),
            "through": interval.get("staffing_level_through", ""),
        }
    manifest = load_manifest(key, root) or {}
    diff = manifest.get("release_diff") or load_release_diff(key, root) or {}
    dropped_ccns: list[str] = []
    ccn_cov = diff.get("ccn_coverage") if isinstance(diff.get("ccn_coverage"), dict) else {}
    if ccn_cov.get("dropped_ccns"):
        dropped_ccns = list(ccn_cov["dropped_ccns"])
    else:
        prior_key = prior_release_key(key)
        if prior_key:
            prior_pi = (
                cms_data_paths.provider_info_dir(root)
                / f"NH_ProviderInfo_{prior_key.month_abbr}{prior_key.year}.csv"
            )
            curr_pi = (
                cms_data_paths.provider_info_dir(root)
                / f"NH_ProviderInfo_{key.month_abbr}{key.year}.csv"
            )
            if prior_pi.is_file() and curr_pi.is_file():
                cov = compare_provider_month_ccn_coverage(prior_pi, curr_pi)
                dropped_ccns = list(cov.get("dropped_ccns") or [])
    own_name = f"NH_Ownership_{key.month_abbr}{key.year}.csv"
    own_rows = 0
    own_path = cms_data_paths.ownership_dir(root) / own_name
    if own_path.is_file():
        with own_path.open(encoding="utf-8", newline="") as f:
            own_rows = max(0, sum(1 for _ in csv.reader(f)) - 1)
    prior_own_rows = None
    prior_key = prior_release_key(key)
    if prior_key:
        prior_own = (
            cms_data_paths.ownership_dir(root)
            / f"NH_Ownership_{prior_key.month_abbr}{prior_key.year}.csv"
        )
        if prior_own.is_file():
            with prior_own.open(encoding="utf-8", newline="") as f:
                prior_own_rows = max(0, sum(1 for _ in csv.reader(f)) - 1)
    nh_path = cms_data_paths.provider_info_dir(root) / f"NH_ProviderInfo_{key.month_abbr}{key.year}.csv"
    nh_present = nh_path.is_file()
    nh_sha = _sha256_file(nh_path) if nh_present else ""
    nh_rows = 0
    if nh_present:
        with nh_path.open(encoding="utf-8", newline="") as f:
            nh_rows = max(0, sum(1 for _ in csv.reader(f)) - 1)
    handoff: dict[str, Any] = {
        "release_key": key.label,
        "provider_processing_month": key.label,
        "staffing_case_mix_quarter": get_quarter_from_processing_month(key.label),
        "staffing_period": staffing_period,
        "pbj_root_sync": {
            "source_file": str(norm_path.relative_to(root)),
            "destination_file": f"provider_info/ProviderInfoNorm_{key.year}_{key.month:02d}.csv",
            "sha256": norm_sha,
            "row_count": norm_rows,
        },
        "pbj_root_nh_snapshot_sync": {
            "source_file": str(nh_path.relative_to(root)) if nh_present else None,
            "destination_file": f"provider_info/NH_ProviderInfo_{key.month_abbr}{key.year}.csv",
            "sha256": nh_sha or None,
            "row_count": nh_rows if nh_present else None,
            "gitignored_in_pbj_root": True,
            "required_for_parity": True,
            "note": (
                "NH snapshot is local/gitignored for parity gates; Norm is the public handoff "
                "artifact. Cross-repo copy is a separate explicit step — not part of acquire."
            ),
        },
        "pbj_root_combined_latest_note": (
            "Rebuild provider_info_combined_latest.csv separately in pbj-root after Norm export; "
            "do not copy full PBJapp provider_info_combined.csv."
        ),
        "sync_command": (
            f"# Provider Info pilot: handoff artifact only — no auto cross-repo write. "
            f"When ready for public handoff, copy "
            f"ProviderInfoNorm_{key.year}_{key.month:02d}.csv per pbj_root_sync and run "
            f"pbj-root verify_provider_release_handoff.py --release-key {key.label}"
        ),
        "gates_to_run_in_pbj_root": [
            "python scripts/backfill_provider_norm_urban.py",
            "python scripts/validate_provider_norm_snapshot.py",
            "python scripts/simulate_render_deploy_gates.py",
            f"python scripts/verify_provider_release_handoff.py --release-key {key.label}",
        ],
        "derived_rebuilds_in_pbj_root": [
            "python scripts/build_state_page_aggregates.py",
            "python generate_search_index.py",
        ],
        "manual_before_pbj_commit": [
            "Rebuild provider_info_combined_latest.csv",
            "python scripts/validate_release.py",
        ],
        "provider_promotion": {
            "nh_snapshot": f"provider_info/NH_ProviderInfo_{key.month_abbr}{key.year}.csv",
            "nh_present_in_pbjapp": nh_present,
            "ready_for_pbj_commit": norm_path.is_file() and nh_present,
            "validate_mode": "NH parity" if nh_present else "self-check only",
            "note": (
                "Norm-only sync is a bug. Both NH (local) and Norm (committed) must exist in PBJapp "
                "before provider-release. Render deploys Norm; NH drives parity gates and search_index."
            ),
        },
        "pbj_metrics_rebuild_when_synced": [
            "python scripts/patch_state_quarterly_lpn.py",
            "python scripts/patch_state_quarterly_medians.py",
            "python scripts/verify_pbjapp_sync.py --source <PBJapp>",
        ],
        "pbj_metrics_verify_scope": (
            "verify_pbjapp_sync.py checks facility/state/national quarterly CSVs only — "
            "not provider Norm/NH or handoff parity"
        ),
        "snf_owners_state_lists": {
            "source_glob": "ownership/SNF_All_Owners*.csv",
            "sync_command": (
                "# Separate ownership family — explicit future handoff; not part of Provider Info acquire"
            ),
            "rebuild_scripts": [
                "python scripts/build_snf_owners_index.py",
                "python scripts/build_snf_owners_ccn_index.py",
                "python scripts/validate_ownership_linkage.py",
            ],
            "note": (
                "State /owners/* pages use SNF_All_Owners + build_snf_owners_index.py "
                "(not monthly NH_Ownership_* from provider zip)."
            ),
        },
        "dropped_ccns_vs_prior_month": dropped_ccns,
    }
    if prior_key and prior_own_rows is not None:
        handoff[f"ownership_rows_{prior_key.month_abbr.lower()}{prior_key.year}"] = prior_own_rows
    handoff[f"ownership_rows_{key.month_abbr.lower()}{key.year}"] = own_rows
    out = manifest_dir(key, root) / "pbj_root_handoff.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(handoff, f, indent=2, sort_keys=False)
        f.write("\n")
    return out
def load_manifest(key: ReleaseKey, root: Path | None = None) -> Optional[dict[str, Any]]:
    path = manifest_dir(key, root) / "release_manifest.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)
def load_release_diff(key: ReleaseKey, root: Path | None = None) -> Optional[dict[str, Any]]:
    path = manifest_dir(key, root) / "release_diff.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)
def write_release_diff(key: ReleaseKey, diff: dict[str, Any], root: Path | None = None) -> Path:
    mdir = manifest_dir(key, root)
    mdir.mkdir(parents=True, exist_ok=True)
    out = mdir / "release_diff.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(diff, f, indent=2, sort_keys=False)
        f.write("\n")
    return out
def find_prior_successful_manifest(key: ReleaseKey, root: Path | None = None) -> Optional[dict[str, Any]]:
    """Walk back month-by-month until a prior release_manifest.json is found."""
    cursor = prior_release_key(key)
    while cursor is not None:
        prior = load_manifest(cursor, root)
        if prior is not None:
            return prior
        cursor = prior_release_key(cursor)
    return None
def _member_index(manifest: Optional[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not manifest:
        return {}
    members = manifest.get("source_members") or []
    if members:
        return {
            m.get("basename"): m
            for m in members
            if m.get("basename")
        }
    out: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("inner_inventory") or []:
        basename = entry.get("basename")
        if basename:
            out[basename] = {
                "basename": basename,
                "ingestion_status": "retained_unmodeled",
                "schema_fingerprint": entry.get("schema_fingerprint", ""),
                "source_sha256": entry.get("sha256", ""),
            }
    return out
def _compact_diff_intervals(prior_path: Path, current_path: Path) -> dict[str, Any]:
    prior = staffing_interval_from_interval_csv(prior_path)
    curr = staffing_interval_from_interval_csv(current_path)
    changed_codes = [
        code
        for code in sorted(set(prior) | set(curr))
        if prior.get(code) != curr.get(code)
    ]
    return {
        "prior": prior,
        "current": curr,
        "changed_fields": changed_codes,
    }
def _compact_diff_ccn_coverage(prior_path: Path, current_path: Path) -> dict[str, Any]:
    cov = compare_provider_month_ccn_coverage(prior_path, current_path)
    return {
        "prior_unique_ccn": cov["prior_unique_ccn"],
        "current_unique_ccn": cov["current_unique_ccn"],
        "added_ccns_count": len(cov["added_ccns"]),
        "dropped_ccns_count": len(cov["dropped_ccns"]),
        "added_ccns_sample": cov["added_ccns"][:20],
        "dropped_ccns_sample": cov["dropped_ccns"][:20],
    }
def _stable_citation_keys(path: Path) -> set[str]:
    usecols = ["CMS Certification Number (CCN)", "Survey Date", "Deficiency Prefix", "Deficiency Tag Number"]
    df_head = pd.read_csv(path, nrows=0, low_memory=False)
    cols = [c for c in usecols if c in df_head.columns]
    if len(cols) < 2:
        return set()
    df = pd.read_csv(path, usecols=cols, dtype=str, low_memory=False)
    ccn = df[cols[0]].astype(str).str.zfill(6)
    parts = [ccn]
    for c in cols[1:]:
        parts.append(df[c].astype(str).str.strip())
    return set("|".join(row) for row in zip(*parts))
def _compact_diff_citations(prior_path: Path, current_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["ccn_coverage"] = _compact_diff_ccn_coverage(prior_path, current_path)
    try:
        prior_keys = _stable_citation_keys(prior_path)
        curr_keys = _stable_citation_keys(current_path)
        new_keys = sorted(curr_keys - prior_keys)
        removed_keys = sorted(prior_keys - curr_keys)
        out["stable_key_diff"] = {
            "new_events_count": len(new_keys),
            "removed_events_count": len(removed_keys),
            "new_events_sample": new_keys[:20],
            "removed_events_sample": removed_keys[:20],
        }
    except Exception as exc:
        out["stable_key_diff_error"] = str(exc)
    return out
def _stable_ownership_keys(path: Path) -> set[str]:
    df_head = pd.read_csv(path, nrows=0, low_memory=False)
    cols = list(df_head.columns)
    ccn_col = next((c for c in cols if "CCN" in c), None)
    owner_col = next((c for c in cols if "Owner" in c), None)
    if not ccn_col or not owner_col:
        return set()
    df = pd.read_csv(path, usecols=[ccn_col, owner_col], dtype=str, low_memory=False)
    ccn = df[ccn_col].astype(str).str.zfill(6)
    owner = df[owner_col].astype(str).str.strip().str.upper()
    return set(f"{c}|{o}" for c, o in zip(ccn, owner))
def _compact_diff_ownership(prior_path: Path, current_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["ccn_coverage"] = _compact_diff_ccn_coverage(prior_path, current_path)
    try:
        prior_keys = _stable_ownership_keys(prior_path)
        curr_keys = _stable_ownership_keys(current_path)
        new_keys = sorted(curr_keys - prior_keys)
        removed_keys = sorted(prior_keys - curr_keys)
        out["facility_events"] = {
            "new_owner_links_count": len(new_keys),
            "removed_owner_links_count": len(removed_keys),
            "new_owner_links_sample": new_keys[:20],
            "removed_owner_links_sample": removed_keys[:20],
        }
    except Exception as exc:
        out["facility_events_error"] = str(exc)
    return out
def _prior_extract_path(
    prior_key: ReleaseKey,
    basename: str,
    logical_dest: str,
    root: Path | None,
) -> Optional[Path]:
    mon = prior_key.month_abbr
    year = prior_key.year
    # Replace month/year suffix in basename
    prior_basename = re.sub(
        r"[A-Za-z]{3}\d{4}\.csv$",
        f"{mon}{year}.csv",
        basename,
    )
    return _extracted_path_for_basename(prior_basename, logical_dest, root)
def build_release_diff(
    key: ReleaseKey,
    current_manifest: dict[str, Any],
    prior_manifest: Optional[dict[str, Any]],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    prior_key = prior_release_key(key)
    prior_label = prior_key.label if prior_key else None
    current_by = _member_index(current_manifest)
    prior_by = _member_index(prior_manifest)
    new_members = sorted(set(current_by) - set(prior_by))
    removed_members = sorted(set(prior_by) - set(current_by))
    changed_schema: list[dict[str, Any]] = []
    changed_coverage: list[dict[str, Any]] = []
    changed_contents: list[dict[str, Any]] = []
    facility_events: list[dict[str, Any]] = []
    dataset_diffs: dict[str, Any] = {}
    logical_by_basename = expected_active_csv_names(key)
    for basename, cur in current_by.items():
        prior = prior_by.get(basename)
        if not prior:
            continue
        if cur.get("schema_fingerprint") != prior.get("schema_fingerprint"):
            changed_schema.append(
                {
                    "basename": basename,
                    "prior_fingerprint": prior.get("schema_fingerprint"),
                    "current_fingerprint": cur.get("schema_fingerprint"),
                    "ingestion_status": cur.get("ingestion_status"),
                }
            )
        prior_cov = prior.get("coverage") or {}
        cur_cov = cur.get("coverage") or {}
        if prior_cov.get("row_count") != cur_cov.get("row_count"):
            changed_coverage.append(
                {
                    "basename": basename,
                    "prior_row_count": prior_cov.get("row_count"),
                    "current_row_count": cur_cov.get("row_count"),
                }
            )
        if cur.get("source_sha256") != prior.get("source_sha256"):
            changed_contents.append(
                {
                    "basename": basename,
                    "prior_sha256": prior.get("source_sha256"),
                    "current_sha256": cur.get("source_sha256"),
                    "ingestion_status": cur.get("ingestion_status"),
                }
            )
    if prior_key and prior_manifest:
        for basename, logical in logical_by_basename.items():
            cur_path = _extracted_path_for_basename(basename, logical, root)
            prior_path = _prior_extract_path(prior_key, basename, logical, root)
            if not cur_path or not prior_path or not cur_path.is_file() or not prior_path.is_file():
                continue
            if logical == "provider_info" and basename.startswith("NH_DataCollectionIntervals_"):
                dataset_diffs["intervals"] = _compact_diff_intervals(prior_path, cur_path)
            elif logical == "citations" and basename.startswith("NH_HealthCitations_"):
                cit_diff = _compact_diff_citations(prior_path, cur_path)
                dataset_diffs["citations"] = cit_diff
                fe = cit_diff.get("stable_key_diff")
                if fe and fe.get("new_events_count", 0) > 0:
                    facility_events.append(
                        {
                            "source": "citations",
                            "basename": basename,
                            "new_events_count": fe["new_events_count"],
                            "sample": fe.get("new_events_sample", []),
                        }
                    )
            elif logical == "ownership":
                own_diff = _compact_diff_ownership(prior_path, cur_path)
                dataset_diffs["ownership"] = own_diff
                fe = own_diff.get("facility_events")
                if fe and fe.get("new_owner_links_count", 0) > 0:
                    facility_events.append(
                        {
                            "source": "ownership",
                            "basename": basename,
                            "new_owner_links_count": fe["new_owner_links_count"],
                            "sample": fe.get("new_owner_links_sample", []),
                        }
                    )
            elif logical == "provider_info" and basename.startswith("NH_ProviderInfo_"):
                dataset_diffs["provider_info_ccn"] = _compact_diff_ccn_coverage(prior_path, cur_path)
    return {
        "release_key": key.label,
        "prior_release_key": prior_label,
        "diff_at": datetime.now(timezone.utc).isoformat(),
        "new_members": new_members,
        "removed_members": removed_members,
        "changed_schema": changed_schema,
        "changed_coverage": changed_coverage,
        "changed_contents": changed_contents,
        "new_facility_events": facility_events,
        "dataset_diffs": dataset_diffs,
    }
def compute_promotion_blocked(
    manifest: dict[str, Any],
    diff: dict[str, Any],
    *,
    prior_manifest: Optional[dict[str, Any]] = None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    prior_by = _member_index(prior_manifest)
    new_members = set(diff.get("new_members") or [])
    removed_members = set(diff.get("removed_members") or [])
    for member in manifest.get("source_members", []):
        basename = member.get("basename", "")
        status = member.get("ingestion_status", "")
        if status == "unmapped_new_source" and basename in new_members:
            reasons.append(f"unmapped new source: {basename}")
        if status == "rejected":
            reasons.append(f"rejected member: {basename}")
    for row in diff.get("changed_schema") or []:
        if row.get("ingestion_status") == "ingested":
            reasons.append(f"unexpected schema change on ingested source: {row.get('basename')}")
    for basename in removed_members:
        prior = prior_by.get(basename) or {}
        if prior.get("ingestion_status") == "ingested":
            reasons.append(f"prior ingested member missing from archive: {basename}")
    return bool(reasons), sorted(set(reasons))
def compare_provider_month_ccn_coverage(
    prior_path: Path,
    current_path: Path,
) -> dict[str, Any]:
    ccn_col = "CMS Certification Number (CCN)"
    prior = pd.read_csv(prior_path, usecols=[ccn_col], dtype=str, low_memory=False)
    curr = pd.read_csv(current_path, usecols=[ccn_col], dtype=str, low_memory=False)
    prior_ccn = set(prior[ccn_col].astype(str).str.zfill(6))
    curr_ccn = set(curr[ccn_col].astype(str).str.zfill(6))
    dropped = sorted(prior_ccn - curr_ccn)
    added = sorted(curr_ccn - prior_ccn)
    return {
        "prior_rows": len(prior),
        "current_rows": len(curr),
        "prior_unique_ccn": len(prior_ccn),
        "current_unique_ccn": len(curr_ccn),
        "dropped_ccns": dropped,
        "added_ccns": added,
    }
