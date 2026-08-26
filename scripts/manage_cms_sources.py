#!/usr/bin/env python3
"""
Single entry point for CMS national source layout (EIN + non-nurse).

Policy
------
EIN:
  - ``EIN/monolithic/`` holds the CMS multi-quarter PUF (bulk history).
  - ``EIN/quarters/`` holds only the *tail* — quarters newer than that PUF.
  - When you replace the monolithic with a newer CMS drop, run ``ein consolidate``
    to archive loose quarter zips that are now redundant.

Non-nurse:
  - ``NonNursecsv/`` + ``standardized_NonNurse/`` — one CSV per quarter (no monolithic).

Facility slices (superdynamic v2) live in ``deployments/pbj320-<CCN>/``.
``create_vercel_deployment.py`` refreshes them when national moves ahead.

Examples
--------
  python scripts/manage_cms_sources.py status
  python scripts/manage_cms_sources.py status --ccn 315461
  python scripts/manage_cms_sources.py ein ingest ~/Downloads/PBJ_Employee_Detail_....zip
  python scripts/manage_cms_sources.py ein set-monolithic ~/Downloads/Payroll_Based_Journal_....zip
  python scripts/manage_cms_sources.py ein consolidate
  python scripts/manage_cms_sources.py provider ingest-release --year 2026 --month 6
  python scripts/manage_cms_sources.py nonnurse ingest ~/Downloads/PBJ_Daily_Non_Nurse_....zip --standardize
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cms_data_paths import (  # noqa: E402
    citations_dir,
    ein_monolithic_dir,
    ein_quarters_dir,
    facility_deploy_dir,
    provider_info_dir,
    provider_info_normalized_dir,
    repo_root,
    standardized_nonnurse_dir,
)
import io  # noqa: E402
import re  # noqa: E402
import zipfile  # noqa: E402
PUF_BASENAME = "Payroll Based Journal Employee Detail Nursing Home Staffing.zip"

from facility_ein_lib import (  # noqa: E402
    _ein_all_row_level_members_monolithic,
    _iter_ein_supplemental_zip_paths,
    discover_ein_quarter_range_on_disk,
    ein_cy_quarter_from_ein_source_name,
    ein_quarter_sort_key_to_label,
    parse_ein_quarter_bound,
    resolve_ein_primary_zip,
)
import zipfile  # noqa: E402


def _py() -> str:
    return sys.executable


def _run_script(rel: str, *args: str) -> int:
    candidates = [_ROOT / rel, _ROOT / "scripts" / rel]
    script = next((p for p in candidates if p.is_file()), None)
    if script is None:
        print(f"ERROR: missing script {rel}", file=sys.stderr)
        return 1
    return subprocess.call([_py(), str(script), *args], cwd=str(_ROOT))


def _monolithic_puf_path() -> str | None:
    p = resolve_ein_primary_zip(str(_ROOT))
    return p if p and os.path.isfile(p) else None


def _quarters_in_monolithic_only(puf: str) -> set[str]:
    out: set[str] = set()
    try:
        with zipfile.ZipFile(puf, "r") as zf:
            for member in _ein_all_row_level_members_monolithic(zf):
                cy = ein_cy_quarter_from_ein_source_name(member) or ein_cy_quarter_from_ein_source_name(
                    os.path.basename(puf)
                )
                if cy:
                    out.add(cy)
    except zipfile.BadZipFile:
        pass
    return out


def _loose_quarter_zips() -> dict[str, str]:
    """CYyyyyQn -> path for zips directly under EIN/quarters/ (not _archive)."""
    qdir = ein_quarters_dir()
    out: dict[str, str] = {}
    if not qdir.is_dir():
        return out
    for zp in sorted(qdir.glob("*.zip")):
        cy = ein_cy_quarter_from_ein_source_name(zp.name)
        if cy:
            out[cy] = str(zp.resolve())
    return out


def _dir_size_mb(path: Path) -> float:
    total = 0
    if path.is_file():
        return path.stat().st_size / 1e6
    if not path.is_dir():
        return 0.0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / 1e6


def cmd_status(args: argparse.Namespace) -> int:
    root = str(_ROOT)
    puf = _monolithic_puf_path()
    lo, hi = discover_ein_quarter_range_on_disk(root, puf or "")
    loose = _loose_quarter_zips()
    in_puf = _quarters_in_monolithic_only(puf) if puf else set()

    print("=== CMS national sources ===\n")
    print("EIN policy: monolithic = bulk history; quarters/ = tail after PUF max\n")

    if puf:
        puf_lo = min(parse_ein_quarter_bound(q) or 0 for q in in_puf) if in_puf else None
        puf_hi = max(parse_ein_quarter_bound(q) or 0 for q in in_puf) if in_puf else None
        puf_span = (
            f"{ein_quarter_sort_key_to_label(puf_lo)} … {ein_quarter_sort_key_to_label(puf_hi)}"
            if puf_lo is not None and puf_hi is not None
            else "?"
        )
        print(f"Monolithic PUF:  {os.path.basename(puf)}")
        print(f"  span:          {puf_span} ({len(in_puf)} quarters inside)")
        print(f"  size:          {_dir_size_mb(Path(puf)):.0f} MB")
    else:
        print("Monolithic PUF:  (missing — place under EIN/monolithic/)")

    print(f"\nQuarters tail:   {len(loose)} zip(s) in EIN/quarters/")
    redundant: list[str] = []
    for cy in sorted(loose):
        zp = loose[cy]
        mb = os.path.getsize(zp) / 1e6
        tag = ""
        if cy in in_puf:
            tag = "  [REDUNDANT — also in monolithic; run: ein consolidate]"
            redundant.append(cy)
        print(f"  {cy}  {os.path.basename(zp)}  ({mb:.0f} MB){tag}")

    if lo is not None and hi is not None:
        print(f"\nCombined EIN span on disk: {ein_quarter_sort_key_to_label(lo)} … {ein_quarter_sort_key_to_label(hi)}")

    nn = sorted(standardized_nonnurse_dir().glob("PBJ_dailynonnurse*.csv"))
    print(f"\nNon-nurse:       {len(nn)} standardized CSV(s)", end="")
    if nn:
        print(f"  ({ein_cy_quarter_from_ein_source_name(nn[0].name) or '?'} … "
              f"{ein_cy_quarter_from_ein_source_name(nn[-1].name) or '?'})")
    else:
        print()

    if redundant:
        print(f"\nRecommend: python scripts/manage_cms_sources.py ein consolidate")
        print(f"  (archive {len(redundant)} redundant tail zip(s); saves disk)")

    _print_provider_status()

    cit_dir = citations_dir()
    cit_files = sorted(cit_dir.glob("NH_HealthCitations_*.csv")) if cit_dir.is_dir() else []
    print(f"\nHealth deficiencies (Citations/):  {len(cit_files)} national file(s)")
    if cit_files:
        print(f"  latest: {cit_files[-1].name}")
    print("  (separate CMS dataset — not inside provider-info monthly zips)")

    if args.ccn:
        ccn = str(args.ccn).strip().zfill(6)
        print(f"\n--- Facility {ccn} ---")
        return _run_script("check_facility_cms_data_ready.py", ccn)

    print("\nNext EIN quarter:     python scripts/manage_cms_sources.py ein ingest <zip>")
    print("Next provider month:  python scripts/manage_cms_sources.py provider ingest-release --year YYYY --month M")
    print("Package v2:           python create_vercel_deployment.py <CCN>")
    return 0


def _parse_provider_month_from_name(name: str) -> tuple[int, int] | None:
    m = re.search(r"([A-Za-z]{3})(\d{4})", os.path.basename(name))
    if not m:
        return None
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    mo = months.get(m.group(1).lower()[:3])
    if not mo:
        return None
    return int(m.group(2)), mo


def _print_provider_status() -> None:
    raw_dir = provider_info_dir()
    norm_dir = provider_info_normalized_dir()
    raw_pi = sorted(raw_dir.glob("NH_ProviderInfo_*.csv")) if raw_dir.is_dir() else []
    raw_int = sorted(raw_dir.glob("NH_DataCollectionIntervals_*.csv")) if raw_dir.is_dir() else []
    outer_zips = sorted(raw_dir.glob("nursing_homes_including_rehab_services_*.zip")) if raw_dir.is_dir() else []
    norm_files = sorted(norm_dir.glob("ProviderInfoNorm_*.csv")) if norm_dir.is_dir() else []

    pending_norm = 0
    for p in raw_pi:
        parsed = _parse_provider_month_from_name(p.name)
        if not parsed:
            continue
        y, m = parsed
        if not (norm_dir / f"ProviderInfoNorm_{y}_{m:02d}.csv").is_file():
            pending_norm += 1

    print(f"\nProvider info (monthly CMS archives):")
    print(f"  Yearly outer zips:     {len(outer_zips)} in provider_info/")
    print(f"  Extracted ProviderInfo CSVs: {len(raw_pi)}", end="")
    if raw_pi:
        print(f"  ({raw_pi[0].name} … {raw_pi[-1].name})")
    else:
        print()
    print(f"  Extracted interval CSVs:     {len(raw_int)}")
    print(f"  Normalized snapshots:        {len(norm_files)} in provider_info_normalized/")
    if pending_norm:
        print(f"  Pending normalize:           {pending_norm} month(s) — run: provider normalize")


def _extract_provider_csvs_from_inner_bytes(inner_bytes: bytes, *, dry_run: bool) -> list[str]:
    dest = provider_info_dir()
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as z_inner:
        for name in z_inner.namelist():
            bn = os.path.basename(name)
            if not bn.lower().endswith(".csv"):
                continue
            if not (
                bn.startswith("NH_ProviderInfo_")
                or bn.startswith("NH_DataCollectionIntervals_")
            ):
                continue
            out = dest / bn
            if out.is_file() and out.stat().st_size > 5000:
                print(f"  [skip] {bn} (already on disk)")
                continue
            if dry_run:
                print(f"  [dry-run] would extract {bn}")
            else:
                out.write_bytes(z_inner.read(name))
                print(f"  [OK] extracted {bn}")
            written.append(bn)
    return written


def cmd_provider_extract_month(args: argparse.Namespace) -> int:
    year = int(args.year)
    month = int(args.month)
    if month < 1 or month > 12:
        print("ERROR: --month must be 1-12", file=sys.stderr)
        return 1
    raw_dir = provider_info_dir()
    outer = raw_dir / f"nursing_homes_including_rehab_services_{year}.zip"
    if not outer.is_file() and year <= 2019:
        outer = raw_dir / f"nh_archive_{year}.zip"
    if not outer.is_file():
        print(f"ERROR: no yearly archive for {year}: {outer}", file=sys.stderr)
        return 1
    if year <= 2019:
        inner_name = f"nh_archive_{month:02d}_{year}.zip"
    else:
        inner_name = f"nursing_homes_including_rehab_services_{month:02d}_{year}.zip"
    print(f"Extracting {inner_name} from {outer.name} ...")
    try:
        with zipfile.ZipFile(outer, "r") as z_outer:
            inner_bytes = z_outer.read(inner_name)
    except KeyError:
        print(f"ERROR: {inner_name} not found inside {outer.name}", file=sys.stderr)
        return 1
    except zipfile.BadZipFile as exc:
        print(f"ERROR: bad zip {outer}: {exc}", file=sys.stderr)
        return 1
    files = _extract_provider_csvs_from_inner_bytes(inner_bytes, dry_run=bool(args.dry_run))
    if not files:
        print("No new provider-info CSVs extracted.")
        return 0
    if not args.dry_run and not args.no_normalize:
        return cmd_provider_normalize(argparse.Namespace(force=False))
    return 0


def cmd_provider_ingest_zip(args: argparse.Namespace) -> int:
    zp = os.path.abspath(args.zip_path)
    if not os.path.isfile(zp):
        print(f"ERROR: not found: {zp}", file=sys.stderr)
        return 1
    written: list[str] = []
    with zipfile.ZipFile(zp, "r") as zf:
        inner_zips = [n for n in zf.namelist() if n.lower().endswith(".zip")]
        csv_direct = [
            n
            for n in zf.namelist()
            if n.lower().endswith(".csv")
            and (
                "NH_ProviderInfo_" in n
                or "NH_DataCollectionIntervals_" in n
            )
        ]
        if csv_direct:
            dest = provider_info_dir()
            dest.mkdir(parents=True, exist_ok=True)
            for name in csv_direct:
                bn = os.path.basename(name)
                out = dest / bn
                if out.is_file() and out.stat().st_size > 5000:
                    print(f"  [skip] {bn}")
                    continue
                if args.dry_run:
                    print(f"  [dry-run] would extract {bn}")
                else:
                    out.write_bytes(zf.read(name))
                    print(f"  [OK] extracted {bn}")
                written.append(bn)
        elif len(inner_zips) == 1:
            written = _extract_provider_csvs_from_inner_bytes(
                zf.read(inner_zips[0]), dry_run=bool(args.dry_run)
            )
        else:
            print(
                "ERROR: expected direct NH_ProviderInfo_/NH_DataCollectionIntervals_ CSVs "
                "or a single inner monthly zip.",
                file=sys.stderr,
            )
            return 1
    if not written:
        print("No new files extracted.")
        return 0
    if not args.dry_run and not args.no_normalize:
        return cmd_provider_normalize(argparse.Namespace(force=False))
    return 0


def cmd_provider_normalize(args: argparse.Namespace) -> int:
    extra = ["--force"] if getattr(args, "force", False) else []
    return _run_script("normalize_provider_info.py", *extra)


def cmd_provider_ingest_release(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(_ROOT / "scripts"))
    import cms_provider_release_lib as cpr  # noqa: E402

    key = cpr.release_key(int(args.year), int(args.month))
    print(f"CMS provider release ingest: {key.label}")
    try:
        manifest = cpr.extract_active_csvs(key, root=_ROOT, dry_run=bool(args.dry_run))
    except (FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    written = [e for e in manifest.get("extracted_active_files", []) if e.get("action") == "written"]
    skipped = manifest.get("skipped_identical", [])
    print(f"  active files written: {len(written)}  skipped (identical): {len(skipped)}")
    mpath = cpr.manifest_dir(key, _ROOT) / "release_manifest.json"
    dpath = cpr.manifest_dir(key, _ROOT) / "release_diff.json"
    print(f"  manifest: {mpath}")
    print(f"  release_diff: {dpath}")
    promo = manifest.get("promotion") or {}
    if promo:
        print(f"  promotion: {promo.get('status', 'unknown')}")
        for reason in promo.get("reasons") or []:
            print(f"    block: {reason}")
    elif manifest.get("promotion_blocked"):
        print("  PROMOTION BLOCKED:")
        for reason in manifest.get("promotion_blocked_reasons") or []:
            print(f"    - {reason}")
    else:
        print("  promotion: OK (not blocked)")
    diff = manifest.get("release_diff") or {}
    if diff.get("new_members"):
        print(f"  new members: {', '.join(diff['new_members'])}")
    unmapped = [
        m.get("basename")
        for m in manifest.get("source_members", [])
        if m.get("ingestion_status") == "unmapped_new_source"
    ]
    if unmapped:
        print(f"  unmapped_new_source: {', '.join(unmapped)}")
    if args.dry_run:
        return 0
    if not args.no_normalize:
        rc = cmd_provider_normalize(argparse.Namespace(force=False))
        if rc != 0:
            return rc
        cpr.update_manifest_normalized_outputs(key, _ROOT)
        handoff_path = cpr.write_pbj_root_handoff(key, _ROOT)
        print(f"  pbj_root_handoff: {handoff_path}")
    # CCN coverage vs prior month when both NH snapshots exist
    mon = key.month_abbr
    curr = provider_info_dir() / f"NH_ProviderInfo_{mon}{key.year}.csv"
    prior_key = cpr.release_key(key.year, key.month - 1) if key.month > 1 else None
    if prior_key and curr.is_file():
        prior_name = f"NH_ProviderInfo_{prior_key.month_abbr}{prior_key.year}.csv"
        prior = provider_info_dir() / prior_name
        if prior.is_file():
            cov = cpr.compare_provider_month_ccn_coverage(prior, curr)
            print(f"  CCN coverage: {cov['prior_unique_ccn']} -> {cov['current_unique_ccn']}")
            if cov["dropped_ccns"]:
                print(f"  dropped CCNs ({len(cov['dropped_ccns'])}): {', '.join(cov['dropped_ccns'])}")
    if manifest.get("promotion_blocked"):
        return 2
    return 0


def cmd_ein_ingest(args: argparse.Namespace) -> int:
    path = os.path.abspath(args.zip_path)
    extra = ["--dry-run"] if args.dry_run else []
    return _run_script("install_ein_supplemental_zip.py", path, *extra)


def cmd_ein_organize(args: argparse.Namespace) -> int:
    extra = ["--dry-run"] if args.dry_run else []
    return _run_script("organize_ein_folder.py", *extra)


def cmd_ein_set_monolithic(args: argparse.Namespace) -> int:
    src = os.path.abspath(args.zip_path)
    if not os.path.isfile(src):
        print(f"ERROR: not found: {src}", file=sys.stderr)
        return 1
    mono_dir = ein_monolithic_dir()
    dest = mono_dir / PUF_BASENAME
    if args.dry_run:
        print(f"Would install monolithic PUF:\n  {src}\n  -> {dest}")
        print("Then run: python scripts/manage_cms_sources.py ein consolidate")
        return 0
    mono_dir.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = mono_dir / "_archive" / f"{dest.stem}_replaced_{stamp}.zip"
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dest), str(archived))
        print(f"[OK] Archived prior PUF -> {archived}")
    shutil.copy2(src, dest)
    print(f"[OK] Installed monolithic PUF -> {dest}")
    if not args.no_consolidate:
        print("\nRunning consolidate...")
        ns = argparse.Namespace(dry_run=False)
        return cmd_ein_consolidate(ns)
    return 0


def cmd_ein_consolidate(args: argparse.Namespace) -> int:
    """Archive loose quarter zips that duplicate quarters already in the monolithic PUF."""
    puf = _monolithic_puf_path()
    if not puf:
        print("No monolithic PUF — nothing to consolidate against.", file=sys.stderr)
        return 1
    in_puf = _quarters_in_monolithic_only(puf)
    loose = _loose_quarter_zips()
    to_move = [(cy, path) for cy, path in loose.items() if cy in in_puf]
    if not to_move:
        print("[OK] No redundant quarter zips — tail is clean.")
        return 0

    archive_dir = ein_quarters_dir() / "_archive"
    print(f"Redundant tail zips ({len(to_move)}) — also inside monolithic PUF:")
    for cy, path in sorted(to_move):
        mb = os.path.getsize(path) / 1e6
        print(f"  {cy}  {os.path.basename(path)}  ({mb:.0f} MB)")
        if not args.dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = archive_dir / f"{Path(path).stem}_consolidated_{stamp}.zip"
            shutil.move(path, dest)
            print(f"    -> {dest}")

    if args.dry_run:
        print("\n(dry-run — no files moved)")
    else:
        _run_script("validate_ein_supplemental_zips.py", "--quick", "--write-manifest")
        _write_sources_state()
        print("\n[OK] Consolidated. quarters/ now holds only the post-PUF tail.")
    return 0


def cmd_nonnurse_ingest(args: argparse.Namespace) -> int:
    path = os.path.abspath(args.zip_path)
    cmd = [path]
    if args.standardize:
        cmd.append("--standardize")
    if args.dry_run:
        cmd.append("--dry-run")
    return _run_script("ingest_cms_nonnurse_quarter.py", *cmd)


def _write_sources_state() -> None:
    """Machine-readable snapshot for tooling (optional)."""
    puf = _monolithic_puf_path()
    loose = _loose_quarter_zips()
    in_puf = _quarters_in_monolithic_only(puf) if puf else set()
    lo, hi = discover_ein_quarter_range_on_disk(str(_ROOT), puf or "")
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "ein": {
            "monolithic_path": puf,
            "monolithic_quarters": sorted(in_puf),
            "tail_quarters": sorted(loose.keys()),
            "combined_span": {
                "min": ein_quarter_sort_key_to_label(lo) if lo is not None else None,
                "max": ein_quarter_sort_key_to_label(hi) if hi is not None else None,
            },
        },
        "paths": {
            "monolithic_dir": str(ein_monolithic_dir()),
            "quarters_dir": str(ein_quarters_dir()),
            "standardized_nonnurse": str(standardized_nonnurse_dir()),
            "facility_deploy_template": "deployments/pbj320-<CCN>/",
        },
    }
    out = repo_root() / "EIN" / "sources_state.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage CMS national PBJ sources (EIN monolithic+tail, non-nurse)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show national layout + optional facility readiness")
    p_status.add_argument("--ccn", help="Also check deploy slice for this CCN")

    p_ein = sub.add_parser("ein", help="EIN monolithic + quarters tail")
    ein_sub = p_ein.add_subparsers(dest="ein_cmd", required=True)

    p_ing = ein_sub.add_parser("ingest", help="Validate and install a new quarter zip into quarters/")
    p_ing.add_argument("zip_path")
    p_ing.add_argument("--dry-run", action="store_true")

    p_org = ein_sub.add_parser("organize", help="File loose EIN downloads from staging/supplemental into quarters/")
    p_org.add_argument("--dry-run", action="store_true")

    p_set = ein_sub.add_parser(
        "set-monolithic",
        help="Replace monolithic PUF and optionally consolidate redundant tail zips",
    )
    p_set.add_argument("zip_path", help="New CMS multi-quarter Employee Detail zip")
    p_set.add_argument("--dry-run", action="store_true")
    p_set.add_argument("--no-consolidate", action="store_true")

    p_con = ein_sub.add_parser(
        "consolidate",
        help="Archive quarter zips that duplicate quarters already in monolithic PUF",
    )
    p_con.add_argument("--dry-run", action="store_true")

    p_nn = sub.add_parser("nonnurse", help="Non-nurse daily staffing")
    nn_sub = p_nn.add_subparsers(dest="nn_cmd", required=True)
    p_nn_ing = nn_sub.add_parser("ingest", help="Install CMS non-nurse quarter download")
    p_nn_ing.add_argument("zip_path")
    p_nn_ing.add_argument("--standardize", action="store_true")
    p_nn_ing.add_argument("--dry-run", action="store_true")

    p_prov = sub.add_parser(
        "provider",
        help="CMS provider info monthly archives (NH_ProviderInfo + DataCollectionIntervals)",
    )
    prov_sub = p_prov.add_subparsers(dest="prov_cmd", required=True)

    p_pext = prov_sub.add_parser(
        "extract",
        help="Extract one processing month from yearly nursing_homes_including_rehab_services_YYYY.zip",
    )
    p_pext.add_argument("--year", type=int, required=True)
    p_pext.add_argument("--month", type=int, required=True, help="1-12")
    p_pext.add_argument("--dry-run", action="store_true")
    p_pext.add_argument("--no-normalize", action="store_true")

    p_pzip = prov_sub.add_parser("ingest", help="Extract from a CMS monthly provider-info zip download")
    p_pzip.add_argument("zip_path")
    p_pzip.add_argument("--dry-run", action="store_true")
    p_pzip.add_argument("--no-normalize", action="store_true")

    p_pnorm = prov_sub.add_parser(
        "normalize",
        help="Normalize new NH_ProviderInfo_*.csv only (skips existing ProviderInfoNorm_*)",
    )
    p_pnorm.add_argument("--force", action="store_true")

    p_prelease = prov_sub.add_parser(
        "ingest-release",
        help="Extract active pipeline files from nested monthly archive inside yearly CMS zip",
    )
    p_prelease.add_argument("--year", type=int, required=True)
    p_prelease.add_argument("--month", type=int, required=True, help="1-12")
    p_prelease.add_argument("--dry-run", action="store_true")
    p_prelease.add_argument("--no-normalize", action="store_true")

    p_pacq = prov_sub.add_parser(
        "acquire",
        help=(
            "Resolve CMS Provider Information (4pq5-n9py), download if newer than local, "
            "validate, normalize via normalize_provider_info.py, write handoff (no pbj-root write)"
        ),
    )
    p_pacq.add_argument("--dry-run", action="store_true")
    p_pacq.add_argument("--force", action="store_true")
    p_pacq.add_argument("--skip-normalize", action="store_true")
    p_pacq.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command == "status":
        rc = cmd_status(args)
        _write_sources_state()
        return rc
    if args.command == "ein":
        if args.ein_cmd == "ingest":
            rc = cmd_ein_ingest(args)
            if rc == 0 and not args.dry_run:
                _write_sources_state()
            return rc
        if args.ein_cmd == "organize":
            return cmd_ein_organize(args)
        if args.ein_cmd == "set-monolithic":
            rc = cmd_ein_set_monolithic(args)
            if rc == 0 and not args.dry_run:
                _write_sources_state()
            return rc
        if args.ein_cmd == "consolidate":
            rc = cmd_ein_consolidate(args)
            if rc == 0 and not args.dry_run:
                _write_sources_state()
            return rc
    if args.command == "nonnurse" and args.nn_cmd == "ingest":
        return cmd_nonnurse_ingest(args)
    if args.command == "provider":
        if args.prov_cmd == "extract":
            return cmd_provider_extract_month(args)
        if args.prov_cmd == "ingest":
            return cmd_provider_ingest_zip(args)
        if args.prov_cmd == "ingest-release":
            return cmd_provider_ingest_release(args)
        if args.prov_cmd == "normalize":
            return cmd_provider_normalize(args)
        if args.prov_cmd == "acquire":
            return cmd_provider_acquire(args)

    return 1


def cmd_provider_acquire(args: argparse.Namespace) -> int:
    """CMS metastore → raw NH_ProviderInfo → existing normalize → handoff."""
    sys.path.insert(0, str(_ROOT / "scripts"))
    import cms_provider_info_acquire as acq  # noqa: E402

    try:
        report = acq.acquire_and_process(
            root=_ROOT,
            dry_run=bool(args.dry_run),
            force=bool(args.force),
            skip_normalize=bool(args.skip_normalize),
        )
    except acq.AcquireError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"provider acquire status: {report.get('status')}")
        cms = report.get("cms") or {}
        print(
            f"  CMS vintage: {cms.get('data_vintage_label')} "
            f"file={cms.get('distribution_filename')} id={cms.get('dataset_id')}"
        )
        if report.get("handoff_path"):
            print(f"  handoff: {report['handoff_path']}")
        if report.get("normalized"):
            print(f"  normalized: {report['normalized']}")
        print(f"  ready_for_public_handoff: {report.get('ready_for_public_handoff')}")
        print(f"  cross_repo_write: {report.get('cross_repo_write', False)}")
    ok = report.get("status") in {
        "CURRENT",
        "READY_FOR_PUBLIC_HANDOFF",
        "WOULD_ACQUIRE",
        "ACQUIRED_RAW_ONLY",
    }
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
