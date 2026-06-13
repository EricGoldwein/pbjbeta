#!/usr/bin/env python3
"""
Read-only disk dedupe / compression audit for PBJapp vs pbj-system.

Writes under outputs/disk_audit/ (or --out-dir):
  disk_dedupe_report.json
  disk_dedupe_report.md
  duplicate_files.csv
  compression_candidates.csv

No files are moved, deleted, renamed, or compressed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PBJAPP = _ROOT
DEFAULT_PBJ_SYSTEM = _ROOT.parent / "pbj-system"
DEFAULT_OUT = _ROOT / "outputs" / "disk_audit"

HASH_MIN_BYTES = 50 * 1024 * 1024  # 50 MiB
CY_RE = re.compile(r"CY(\d{4})Q([1-4])", re.I)
PROVIDER_NORM_RE = re.compile(r"ProviderInfoNorm_(\d{4})_(\d{2})\.csv", re.I)

SKIP_DIR_NAMES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".ein_quarter_cache",
}

# Logical PBJapp roots scanned for manifest (relative to repo root).
PBJAPP_DATA_ROOTS = [
    "PBJcsv",
    "standardized_PBJ",
    "NonNursecsv",
    "standardized_NonNurse",
    "EIN",
    "provider_info",
    "provider_info_normalized",
    "provider_info_extracted",
    "indexed",
    "metrics_backups",
    "cms-cost-report",
    "Citations",
    "ownership",
    "backups",
    "outputs",
    "Archive",
    "deployments",
    "facility_red_flag_report",
]

# pbj-system layout aliases for cross-project quarter matching.
PBJ_SYSTEM_ALIASES: list[tuple[str, str, str]] = [
    ("standardized_PBJ", "PBJ/PBJ_Nurse/standardized_PBJNurse", "PBJ_dailynursestaffing_*.csv"),
    (
        "standardized_NonNurse",
        "PBJ/PBJ_NonNurse/standardized_PBJNonNurse",
        "PBJ_dailynonnursestaffing_*.csv",
    ),
    ("provider_info_normalized", "ProviderInfo/provider_info_normalized", "ProviderInfoNorm_*.csv"),
    ("provider_info_extracted", "ProviderInfo/provider_info_extracted", "NH_ProviderInfo_*.csv"),
]

REGEN_SCRIPTS = {
    "standardized_PBJ": ["standardize_pbj_files.py", "run_pipeline_update.py (--only pbj)"],
    "standardized_NonNurse": [
        "standardize_nonnursepbj_files.py",
        "run_pipeline_update.py (--only nonnurse)",
        "scripts/ingest_cms_nonnurse_quarter.py",
    ],
    "provider_info_normalized": ["normalize_provider_info.py", "run_pipeline_update.py (--only providerinfo)"],
}


@dataclass
class FileRecord:
    path: str
    rel: str
    project: str
    size: int
    ext: str
    quarter: Optional[str] = None
    sha256: Optional[str] = None


@dataclass
class FolderSummary:
    rel: str
    files: int
    bytes: int
    quarters: list[str] = field(default_factory=list)
    classification: str = "unknown"
    rationale: str = ""
    compression_phase: str = "D"


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit, scale in (("KiB", 1024), ("MiB", 1024**2), ("GiB", 1024**3), ("TiB", 1024**4)):
        if n < scale * 1024:
            return f"{n / scale:.2f} {unit}"
    return f"{n / 1024**4:.2f} TiB"


def _quarter_from_name(name: str) -> Optional[str]:
    m = CY_RE.search(name)
    return f"CY{m.group(1)}Q{m.group(2)}" if m else None


def _sha256_file(path: Path, limit_mb: Optional[int] = None) -> str:
    h = hashlib.sha256()
    read = 0
    cap = (limit_mb or 0) * 1024 * 1024
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
            read += len(chunk)
            if cap and read >= cap:
                break
    return h.hexdigest()


def _iter_files(root: Path, project: str, *, rel_prefix: str = "") -> Iterable[FileRecord]:
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                st = p.stat()
            except OSError:
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            if rel_prefix:
                rel = f"{rel_prefix}/{rel}" if rel_prefix else rel
            yield FileRecord(
                path=str(p.resolve()),
                rel=rel,
                project=project,
                size=int(st.st_size),
                ext=p.suffix.lower(),
                quarter=_quarter_from_name(fn),
            )


def _manifest_root(root: Path, project: str, sub: str) -> list[FileRecord]:
    base = root / sub
    return list(_iter_files(base, project, rel_prefix=sub))


def _folder_summary(rel: str, records: list[FileRecord]) -> FolderSummary:
    quarters = sorted({r.quarter for r in records if r.quarter})
    total = sum(r.size for r in records)
    cls, why, phase = _classify_folder(rel)
    return FolderSummary(
        rel=rel,
        files=len(records),
        bytes=total,
        quarters=quarters,
        classification=cls,
        rationale=why,
        compression_phase=phase,
    )


def _classify_folder(rel: str) -> tuple[str, str, str]:
    """Return (classification, rationale, compression_phase)."""
    rules: dict[str, tuple[str, str, str]] = {
        "PBJcsv": (
            "must_stay_live",
            "Raw CMS nurse daily CSV source; canonical input for standardization",
            "D",
        ),
        "NonNursecsv": (
            "must_stay_live",
            "Raw CMS non-nurse daily CSV source; canonical input",
            "D",
        ),
        "EIN": (
            "must_stay_live",
            "National Employee Detail zips/PUF; facility_ein_lib has separate env path logic — treat conservatively",
            "D",
        ),
        "provider_info": (
            "must_stay_live",
            "Raw CMS provider-info yearly zips and NH_*.csv extracts",
            "D",
        ),
        "standardized_PBJ": (
            "safe_compress_or_regenerate",
            "Generated from PBJcsv via standardize_pbj_files.py; DuckDB/metrics read path",
            "C",
        ),
        "standardized_NonNurse": (
            "safe_compress_or_regenerate",
            "Generated from NonNursecsv; nonnurse_staffing_lib slice builder input",
            "C",
        ),
        "provider_info_normalized": (
            "safe_compress_or_regenerate",
            "Generated from provider_info via normalize_provider_info.py",
            "C",
        ),
        "provider_info_extracted": (
            "safe_delete_after_verify",
            "Intermediate extract; redundant when normalized + raw zips exist (see pbj-system STORAGE_NOTES)",
            "A",
        ),
        "indexed": (
            "safe_cold_archive",
            "Local rebuildable index/cache (nonnurse_index_lib referenced but optional)",
            "C",
        ),
        "metrics_backups": (
            "safe_delete_after_verify",
            "No code references; backup copies of generated metrics",
            "A",
        ),
        "cms-cost-report": (
            "safe_cold_archive",
            "Generated cost-report analysis outputs under cms-cost-report/output/",
            "C",
        ),
        "Citations": (
            "must_stay_live",
            "National NH health citations CSV source for facility slices",
            "D",
        ),
        "ownership": (
            "safe_compress",
            "CMS ownership/chain performance extracts; not in 11-root hub but used by donor/ownership tooling",
            "C",
        ),
        "backups": (
            "safe_delete_after_verify",
            "Session/code backup trees superseded by git",
            "A",
        ),
        "outputs": (
            "safe_delete_after_verify",
            "Local QA / extract logs and audit outputs (regenerable)",
            "A",
        ),
        "Archive": (
            "safe_cold_archive",
            "Explicit archive folder; verify contents before delete",
            "C",
        ),
        "deployments": (
            "do_not_touch",
            "Active Vercel bundles + per-CCN slices; runtime/deployment dependency",
            "D",
        ),
        "facility_red_flag_report": (
            "safe_cold_archive",
            "Generated HTML/report artifacts",
            "C",
        ),
    }
    top = rel.split("/")[0]
    return rules.get(top, ("unknown", "Not classified — manual review", "D"))


def _root_file_candidates(pbjapp: Path) -> list[dict[str, Any]]:
    patterns = [
        ("facility_quarterly_metrics.pre_unify_*.csv", "backup/superseded", "A"),
        ("provider_info_combined*.csv", "generated rollup", "C"),
        ("facility_quarterly_metrics.csv", "generated metrics rollup", "C"),
        ("non_nurse_facility_metrics.csv", "generated metrics rollup", "C"),
        ("facility_lite_metrics.csv", "generated lite rollup (also under pbj_lite/)", "C"),
        ("nursing_homes_including_rehab_services_*.zip", "raw CMS provider zip (may duplicate pbj-system)", "B"),
        ("*.mp4", "demo media", "A"),
        ("facility_quarterly_metrics.parquet", "generated parquet rollup", "C"),
    ]
    out: list[dict[str, Any]] = []
    for pat, kind, phase in patterns:
        for p in sorted(pbjapp.glob(pat)):
            if not p.is_file():
                continue
            out.append(
                {
                    "path": str(p),
                    "kind": kind,
                    "size": p.stat().st_size,
                    "phase": phase,
                }
            )
    return out


def _compare_by_name_size(
    app_records: dict[str, FileRecord],
    sys_records: dict[str, FileRecord],
    *,
    hash_large: bool,
) -> dict[str, Any]:
    common_names = sorted(set(app_records) & set(sys_records))
    exact_dup: list[dict[str, Any]] = []
    same_name_diff_size: list[dict[str, Any]] = []
    for name in common_names:
        a, b = app_records[name], sys_records[name]
        row = {
            "filename": name,
            "app_path": a.path,
            "sys_path": b.path,
            "app_size": a.size,
            "sys_size": b.size,
            "quarter": a.quarter or b.quarter,
        }
        if a.size == b.size:
            row["size_match"] = True
            if hash_large and a.size >= HASH_MIN_BYTES:
                row["app_sha256"] = _sha256_file(Path(a.path))
                row["sys_sha256"] = _sha256_file(Path(b.path))
                row["hash_match"] = row["app_sha256"] == row["sys_sha256"]
            exact_dup.append(row)
        else:
            row["size_match"] = False
            same_name_diff_size.append(row)
    return {
        "common_by_filename": len(common_names),
        "exact_size_matches": exact_dup,
        "same_name_diff_size": same_name_diff_size,
        "only_app": sorted(set(app_records) - set(sys_records)),
        "only_system": sorted(set(sys_records) - set(app_records)),
    }


def _build_cross_project(pbjapp: Path, pbjsys: Path, *, hash_large: bool) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for app_sub, sys_sub, _pat in PBJ_SYSTEM_ALIASES:
        app_dir = pbjapp / app_sub
        sys_dir = pbjsys / sys_sub
        app_recs = {Path(r.path).name: r for r in _manifest_root(pbjapp, "pbjapp", app_sub)}
        sys_recs = {Path(r.path).name: r for r in _iter_files(sys_dir, "pbj-system", rel_prefix=sys_sub)}
        cmp = _compare_by_name_size(app_recs, sys_recs, hash_large=hash_large)
        cmp["app_folder"] = app_sub
        cmp["system_folder"] = sys_sub
        results.append(cmp)
    return results


def _compression_rows(
    folder_summaries: list[FolderSummary],
    root_candidates: list[dict[str, Any]],
    cross: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fs in folder_summaries:
        rows.append(
            {
                "path": fs.rel,
                "type": "folder",
                "size_bytes": fs.bytes,
                "size_human": _human_bytes(fs.bytes),
                "files": fs.files,
                "classification": fs.classification,
                "phase": fs.compression_phase,
                "rationale": fs.rationale,
                "regenerable_from": ", ".join(REGEN_SCRIPTS.get(fs.rel.split("/")[0], [])),
            }
        )
    for rc in root_candidates:
        rows.append(
            {
                "path": rc["path"],
                "type": "file",
                "size_bytes": rc["size"],
                "size_human": _human_bytes(int(rc["size"])),
                "files": 1,
                "classification": rc["kind"],
                "phase": rc["phase"],
                "rationale": "Root-level generated or media artifact",
                "regenerable_from": "",
            }
        )
    for block in cross:
        for row in block.get("exact_size_matches", []):
            if row.get("hash_match"):
                rows.append(
                    {
                        "path": row["filename"],
                        "type": "cross_project_duplicate",
                        "size_bytes": row["app_size"],
                        "size_human": _human_bytes(int(row["app_size"])),
                        "files": 2,
                        "classification": "safe_delete_after_verify",
                        "phase": "B",
                        "rationale": f"Exact duplicate in pbj-system ({block['system_folder']})",
                        "regenerable_from": "pbj-system copy; PBJapp is superset",
                    }
                )
    return rows


def _duplicate_csv_rows(cross: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in cross:
        for row in block.get("exact_size_matches", []):
            rows.append(
                {
                    "filename": row["filename"],
                    "quarter": row.get("quarter", ""),
                    "app_folder": block["app_folder"],
                    "system_folder": block["system_folder"],
                    "app_path": row["app_path"],
                    "system_path": row["sys_path"],
                    "size_bytes": row["app_size"],
                    "size_match": row.get("size_match", True),
                    "hash_match": row.get("hash_match", ""),
                    "app_sha256": row.get("app_sha256", ""),
                    "sys_sha256": row.get("sys_sha256", ""),
                }
            )
    return rows


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# PBJapp disk dedupe / compression audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Summary",
        "",
        f"- **PBJapp data scanned:** {_human_bytes(report['pbjapp_data_bytes'])} across {report['pbjapp_data_files']} files",
        f"- **pbj-system total:** {_human_bytes(report['pbj_system_bytes'])} across {report['pbj_system_files']} files",
        f"- **Exact cross-project duplicates (size+hash, ≥50MB):** {report['hash_verified_duplicates']}",
        "",
        "## Quarter coverage (PBJapp)",
        "",
    ]
    for fs in report["folder_summaries"]:
        if fs["quarters"]:
            q = fs["quarters"]
            lines.append(
                f"- `{fs['rel']}`: {len(q)} quarters ({q[0]} … {q[-1]}) — **{fs['classification']}** (phase {fs['compression_phase']})"
            )
    lines.extend(["", "## Regeneration map", ""])
    for folder, scripts in REGEN_SCRIPTS.items():
        lines.append(f"- `{folder}` ← {', '.join(scripts)}")
    lines.extend(["", "## Compression phases", ""])
    for phase, title in [
        ("A", "Safest compression/delete candidates"),
        ("B", "Duplicates requiring checksum verification"),
        ("C", "Cold archive candidates"),
        ("D", "Do not touch (live dependencies)"),
    ]:
        lines.append(f"### Phase {phase}: {title}")
        for row in report["compression_candidates"]:
            if row["phase"] == phase:
                lines.append(f"- `{row['path']}` ({row['size_human']}) — {row['rationale']}")
        lines.append("")
    lines.extend(["## Near-duplicate folders (pbj-system)", ""])
    for block in report["cross_project"]:
        lines.append(
            f"- `{block['app_folder']}` vs `{block['system_folder']}`: "
            f"{block['common_by_filename']} common names, "
            f"{len(block['only_app'])} only PBJapp, {len(block['only_system'])} only pbj-system"
        )
        if block["only_app"]:
            tail = block["only_app"][-5:]
            lines.append(f"  - PBJapp-only (tail): {', '.join(tail)}")
    lines.extend(["", "## PowerShell commands (DO NOT RUN until verified)", ""])
    lines.extend(report.get("powershell_commands", []))
    return "\n".join(lines) + "\n"


def _powershell_commands() -> list[str]:
    return [
        "```powershell",
        "# Phase A — compress stale backups/outputs (example; review paths first)",
        "Compress-Archive -Path 'C:\\Users\\egold\\PycharmProjects\\PBJapp\\metrics_backups' `",
        "  -DestinationPath 'D:\\PBJ-cold\\metrics_backups.zip' -CompressionLevel Optimal",
        "",
        "# Phase A — remove provider_info_extracted AFTER verifying normalized + raw zips",
        "# Compare counts first:",
        "python scripts\\audit_disk_dedupe.py --hash-large",
        "",
        "# Phase B — verify then remove pbj-system nurse subset (subset of PBJapp; hash-verified in report)",
        "# ONLY after confirming PBJapp standardized_PBJ has all quarters:",
        "Remove-Item -Recurse -Force 'C:\\Users\\egold\\PycharmProjects\\pbj-system\\PBJ\\PBJ_Nurse\\standardized_PBJNurse'",
        "",
        "# Phase C — cold-move indexed (rebuildable) — copy then verify pipeline",
        "robocopy 'C:\\Users\\egold\\PycharmProjects\\PBJapp\\indexed' 'D:\\PBJ-cold\\indexed' /E /COPY:DAT /MOV",
        "",
        "# Phase D — NEVER compress/delete without junctions:",
        "# PBJcsv, NonNursecsv, EIN, deployments\\pbj320-315461 (active)",
        "```",
    ]


def run_audit(
    pbjapp: Path,
    pbjsys: Path,
    out_dir: Path,
    *,
    hash_large: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    app_by_folder: dict[str, list[FileRecord]] = {}
    for sub in PBJAPP_DATA_ROOTS:
        app_by_folder[sub] = _manifest_root(pbjapp, "pbjapp", sub)

    sys_records = list(_iter_files(pbjsys, "pbj-system"))
    folder_summaries = [_folder_summary(rel, recs) for rel, recs in app_by_folder.items()]
    root_candidates = _root_file_candidates(pbjapp)
    cross = _build_cross_project(pbjapp, pbjsys, hash_large=hash_large)

    hash_verified = 0
    for block in cross:
        for row in block.get("exact_size_matches", []):
            if row.get("hash_match"):
                hash_verified += 1
            elif hash_large and row.get("app_size", 0) >= HASH_MIN_BYTES and row.get("size_match"):
                row["app_sha256"] = _sha256_file(Path(row["app_path"]))
                row["sys_sha256"] = _sha256_file(Path(row["sys_path"]))
                row["hash_match"] = row["app_sha256"] == row["sys_sha256"]
                if row["hash_match"]:
                    hash_verified += 1

    compression_rows = _compression_rows(folder_summaries, root_candidates, cross)
    dup_rows = _duplicate_csv_rows(cross)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pbjapp": str(pbjapp.resolve()),
        "pbj_system": str(pbjsys.resolve()),
        "pbjapp_data_files": sum(len(v) for v in app_by_folder.values()),
        "pbjapp_data_bytes": sum(r.size for recs in app_by_folder.values() for r in recs),
        "pbj_system_files": len(sys_records),
        "pbj_system_bytes": sum(r.size for r in sys_records),
        "hash_min_bytes": HASH_MIN_BYTES,
        "hash_verified_duplicates": hash_verified,
        "folder_summaries": [asdict(fs) for fs in folder_summaries],
        "root_file_candidates": root_candidates,
        "cross_project": cross,
        "compression_candidates": compression_rows,
        "powershell_commands": _powershell_commands(),
    }

    json_path = out_dir / "disk_dedupe_report.json"
    md_path = out_dir / "disk_dedupe_report.md"
    dup_csv = out_dir / "duplicate_files.csv"
    comp_csv = out_dir / "compression_candidates.csv"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")

    if dup_rows:
        with dup_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dup_rows[0].keys()))
            w.writeheader()
            w.writerows(dup_rows)
    else:
        dup_csv.write_text("", encoding="utf-8")

    if compression_rows:
        with comp_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(compression_rows[0].keys()))
            w.writeheader()
            w.writerows(compression_rows)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only PBJapp vs pbj-system disk dedupe audit")
    parser.add_argument("--pbjapp", type=Path, default=DEFAULT_PBJAPP)
    parser.add_argument("--pbj-system", type=Path, default=DEFAULT_PBJ_SYSTEM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--hash-large",
        action="store_true",
        help=f"SHA-256 files >= {_human_bytes(HASH_MIN_BYTES)} in cross-project comparison",
    )
    args = parser.parse_args()

    report = run_audit(args.pbjapp, args.pbj_system, args.out_dir, hash_large=args.hash_large)
    print(f"Wrote {args.out_dir / 'disk_dedupe_report.json'}")
    print(f"PBJapp data: {_human_bytes(report['pbjapp_data_bytes'])} / {report['pbjapp_data_files']} files")
    print(f"pbj-system:  {_human_bytes(report['pbj_system_bytes'])} / {report['pbj_system_files']} files")
    print(f"Hash-verified duplicates: {report['hash_verified_duplicates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
