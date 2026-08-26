#!/usr/bin/env python3
"""Copy validated PBJapp artifacts to sibling pbj-root in expected paths and formats."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import cms_data_paths
from cms_provider_release_lib import load_manifest, release_key

PBJ_ROOT_ENV = "PBJ_ROOT"
PBJ_METRICS_FILES = (
    "facility_quarterly_metrics.csv",
    "state_quarterly_metrics.csv",
    "national_quarterly_metrics.csv",
)
CHOW_ZIP_BASENAME = "Skilled Nursing Facility Change of Ownership.zip"
CHOW_CANONICAL_REL = Path("ownership/_sources/cms_chow/2026-Q1") / CHOW_ZIP_BASENAME
CHAIN_CSV_BASENAME = "Chain_Performance_20260610.csv"
CHAIN_CANONICAL_REL = Path("ownership/_sources/cms_chain_performance/2026-05") / CHAIN_CSV_BASENAME

# pbj-root scripts run after file copies (see pbj-root QUARTER_RELEASE_PLAYBOOK.md)
PROVIDER_NORM_GATES = (
    "python scripts/backfill_provider_norm_urban.py",
    "python scripts/validate_provider_norm_snapshot.py",
    "python scripts/simulate_render_deploy_gates.py",
)
PROVIDER_NORM_DERIVED = (
    "python scripts/build_state_page_aggregates.py",  # case-mix medians, high-risk, rural shares
)
PBJ_METRICS_DERIVED = (
    "python scripts/patch_state_quarterly_lpn.py",
    "python scripts/patch_state_quarterly_medians.py",
)
CHOW_DERIVED = ("python scripts/build_chow_index.py",)
OWNERSHIP_SNF_FULL_REBUILD = (
    "python scripts/build_snf_owners_index.py",
    "python scripts/build_snf_owners_ccn_index.py",
    "python scripts/validate_ownership_linkage.py",
)
OWNERSHIP_INDEX_ONLY_REBUILD = ("python scripts/build_snf_owners_index.py --index-only",)

_MONTH_LOOKUP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
# Derived gzip/SQLite artifacts for /owners/* state lists (NY, CT, FL, NJ, …)
OWNERSHIP_DERIVED_ARTIFACTS = (
    "ownership/state_owner_index.json.gz",
    "ownership/snf_owners_org_index.json.gz",
    "ownership/snf_owners_ccn_index.json.gz",
    "ownership/snf_owners_lookup.sqlite",
)


@dataclass
class CopyResult:
    source: Path
    destination: Path
    action: str
    sha256: str = ""


def resolve_pbj_root(explicit: str | None = None) -> Path:
    """Resolve pbj-root directory (never modifies it without an explicit sync command)."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
    else:
        env = str(os.environ.get(PBJ_ROOT_ENV) or "").strip()
        p = Path(env).expanduser().resolve() if env else (_ROOT.parent / "pbj-root").resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"pbj-root not found: {p} (set {PBJ_ROOT_ENV} or pass --pbj-root)")
    return p


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_file(
    src: Path,
    dst: Path,
    *,
    dry_run: bool,
    force: bool,
) -> CopyResult:
    if not src.is_file():
        raise FileNotFoundError(f"missing source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_sha = _sha256_file(src)
    if dst.is_file():
        dst_sha = _sha256_file(dst)
        if dst_sha == src_sha:
            return CopyResult(src, dst, "skip_identical", src_sha)
        if not force:
            raise ValueError(
                f"destination differs (use --force): {dst}\n  source sha256={src_sha}\n  dest sha256={dst_sha}"
            )
    if dry_run:
        return CopyResult(src, dst, "dry_run", src_sha)
    shutil.copy2(src, dst)
    return CopyResult(src, dst, "copied", src_sha)


def _copy_nh_snapshot_for_backfill(
    key: Any,
    pbj: Path,
    *,
    dry_run: bool,
    force: bool,
) -> CopyResult | None:
    """Copy paired NH_ProviderInfo to pbj-root for local backfill (gitignored, not deployed)."""
    nh_name = f"NH_ProviderInfo_{key.month_abbr}{key.year}.csv"
    src = cms_data_paths.provider_info_dir() / nh_name
    if not src.is_file():
        print(f"WARNING: paired NH snapshot missing in PBJapp: {src}", file=sys.stderr)
        return None
    dst = pbj / "provider_info" / nh_name
    return _copy_file(src, dst, dry_run=dry_run, force=force)


def _print_results(results: list[CopyResult]) -> None:
    for r in results:
        print(f"[{r.action}] {r.source} -> {r.destination}")


def _parse_snf_owners_release_date(path: Path) -> date | None:
    """Parse CMS SNF_All_Owners filename date (ISO or Month_YYYY)."""
    stem = path.stem.lower()
    m_iso = re.search(r"(\d{4})[._-](\d{1,2})(?:[._-](\d{1,2}))?", stem)
    if m_iso:
        y, mo = int(m_iso.group(1)), int(m_iso.group(2))
        day = int(m_iso.group(3)) if m_iso.group(3) else 1
        if 1 <= mo <= 12 and 1 <= day <= 31:
            return date(y, mo, day)
    m_word = re.search(r"owners[_-]?([a-z]+)[_-]?(\d{4})", stem)
    if m_word:
        mo = _MONTH_LOOKUP.get(m_word.group(1))
        if mo:
            return date(int(m_word.group(2)), mo, 1)
    return None


def _policy_active_snf_all_owners_csv(root: Path) -> Path | None:
    """Policy-selected national SNF_All_Owners CSV after active-release handoff staging."""
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        from ownership.ownership_active_release_handoff import (
            ensure_active_ownership_source_staged,
        )
        from ownership.ownership_release_policy import resolve_ownership_source_path

        ensure_active_ownership_source_staged(root)
        return resolve_ownership_source_path(root)
    except Exception:
        return None


def _newest_snf_all_owners_csv(root: Path) -> Path | None:
    """Deprecated alias — use policy-selected active release path."""
    return _policy_active_snf_all_owners_csv(root)


def _artifact_mtime(pbj: Path, rel: str) -> float | None:
    path = pbj / rel.replace("/", os.sep)
    return path.stat().st_mtime if path.is_file() else None


def cmd_status(args: argparse.Namespace) -> int:
    pbj = resolve_pbj_root(args.pbj_root)
    print(f"pbj-root: {pbj}")
    rows: list[tuple[str, Path, Path]] = [
        ("provider-norm (latest)", _ROOT / "provider_info_normalized", pbj / "provider_info"),
        ("pbj-metrics", _ROOT, pbj),
        ("chow-zip", _ROOT / CHOW_CANONICAL_REL, pbj / "ownership" / CHOW_ZIP_BASENAME),
        ("chain-performance", _ROOT / CHAIN_CANONICAL_REL, pbj / "ownership" / CHAIN_CSV_BASENAME),
    ]
    snf_src = _newest_snf_all_owners_csv(_ROOT)
    snf_dst = _newest_snf_all_owners_csv(pbj)
    if snf_src and snf_dst:
        match = _sha256_file(snf_src) == _sha256_file(snf_dst)
        print(f"\nsnf-all-owners: src={snf_src.name} dst={snf_dst.name} sha_match={match}")
    else:
        print(
            f"\nsnf-all-owners: src={'yes' if snf_src else 'no'} "
            f"dst={'yes' if snf_dst else 'no'}"
        )
    for label, src_hint, dst in rows:
        if label == "provider-norm (latest)":
            norms = sorted((_ROOT / "provider_info_normalized").glob("ProviderInfoNorm_*.csv"))
            src = norms[-1] if norms else None
            dst = pbj / "provider_info" / src.name if src else dst
        elif label == "pbj-metrics":
            print(f"\n{label}:")
            for name in PBJ_METRICS_FILES:
                s = _ROOT / name
                d = pbj / name
                print(f"  {name}: src={'yes' if s.is_file() else 'no'} dst={'yes' if d.is_file() else 'no'}")
            continue
        else:
            src = src_hint
        if src and src.is_file() and dst.is_file():
            match = _sha256_file(src) == _sha256_file(dst)
            print(f"\n{label}: src={src.name} dst={dst.name} sha_match={match}")
        else:
            print(f"\n{label}: src={'yes' if src and src.is_file() else 'no'} dst={'yes' if dst.is_file() else 'no'}")
    return 0


def cmd_checklist(args: argparse.Namespace) -> int:
    """Systematic handoff checklist: copy status, derived artifacts, suggested commands."""
    pbj = resolve_pbj_root(args.pbj_root)
    print(f"pbj-root: {pbj}\n")
    print("== Copy targets ==")
    cmd_status(args)

    print("\n== State owner lists (/owners/ny, /owners/ct, ...) ==")
    snf_src = _newest_snf_all_owners_csv(_ROOT)
    snf_dst = _newest_snf_all_owners_csv(pbj)
    if not snf_src:
        print("  SNF source: MISSING in PBJapp ownership/ — ingest CMS SNF All Owners drop first")
    else:
        src_d = _parse_snf_owners_release_date(snf_src)
        print(f"  PBJapp newest: {snf_src.name}" + (f" ({src_d.isoformat()})" if src_d else ""))
    if not snf_dst:
        print("  pbj-root newest: MISSING")
    else:
        dst_d = _parse_snf_owners_release_date(snf_dst)
        print(f"  pbj-root newest: {snf_dst.name}" + (f" ({dst_d.isoformat()})" if dst_d else ""))
    if snf_src and snf_dst:
        src_d = _parse_snf_owners_release_date(snf_src) or date.min
        dst_d = _parse_snf_owners_release_date(snf_dst) or date.min
        if src_d > dst_d:
            print("  -> PBJapp SNF is NEWER - run: snf-all-owners --rebuild")
        elif src_d < dst_d:
            print("  -> pbj-root SNF is NEWER - ingest newer CMS drop into PBJapp before sync")
        elif _sha256_file(snf_src) != _sha256_file(snf_dst):
            print("  -> Same release date, different bytes - run: snf-all-owners --force --rebuild")
        else:
            print("  -> SNF CSV in sync")

    snf_mtime = _artifact_mtime(pbj, snf_dst.name if snf_dst else "") if snf_dst else None
    stale_derived: list[str] = []
    for rel in OWNERSHIP_DERIVED_ARTIFACTS:
        mt = _artifact_mtime(pbj, rel)
        label = "ok" if mt else "MISSING"
        if snf_mtime and mt and mt < snf_mtime:
            label = "STALE (older than SNF CSV)"
            stale_derived.append(rel)
        print(f"  {rel}: {label}")
    if stale_derived:
        print("  -> Run: ownership-rebuild")

    print("\n== Not synced by this script (manual / separate CMS cadence) ==")
    print("  provider_info_combined_latest.csv - rebuild in pbj-root when legal-name crosswalk needed")
    print("  NH_Ownership_*.csv - monthly facility contacts (provider zip); not used for /owners/* lists")
    print("  facility_quarterly_metrics.csv.gz - run build_facility_quarterly_deploy_snapshot.py in pbj-root")
    print("  SFF / staffing compliance - see pbj-root QUARTER_RELEASE_PLAYBOOK.md")

    print("\n== One-command orchestration ==")
    print("  Monthly CMS provider:  provider-release --release-key YYYY-MM --force")
    print("  PBJ quarterlies:       pbj-metrics --rebuild")
    print("  SNF owners + lists:    snf-all-owners --rebuild   (when PBJapp has newest SNF CSV)")
    print("  All wired steps:       full-handoff --release-key YYYY-MM [--with-pbj-metrics] [--with-snf-owners]")
    return 0


def cmd_snf_all_owners(args: argparse.Namespace) -> int:
    """Copy policy-selected SNF_All_Owners CSV to pbj-root (unless --force blocks downgrade)."""
    pbj = resolve_pbj_root(args.pbj_root)
    src = _newest_snf_all_owners_csv(_ROOT)
    if not src:
        raise FileNotFoundError(
            "Policy-selected SNF_All_Owners source missing — check ownership_release_policy.json"
        )
    dst = pbj / "ownership" / src.name
    dst_existing = _newest_snf_all_owners_csv(pbj)
    if dst_existing and not args.force:
        src_d = _parse_snf_owners_release_date(src) or date.min
        dst_d = _parse_snf_owners_release_date(dst_existing) or date.min
        if src_d < dst_d:
            print(
                f"SKIP: PBJapp {src.name} ({src_d}) is older than pbj-root {dst_existing.name} ({dst_d}). "
                "Ingest newer CMS drop into PBJapp or pass --force.",
                file=sys.stderr,
            )
            return 1
    results = [_copy_file(src, dst, dry_run=bool(args.dry_run), force=bool(args.force))]
    policy_src = _ROOT / "ownership" / "ownership_release_policy.json"
    policy_dst = pbj / "ownership" / "ownership_release_policy.json"
    if policy_src.is_file():
        results.append(
            _copy_file(policy_src, policy_dst, dry_run=bool(args.dry_run), force=bool(args.force))
        )
    policy_py = _ROOT / "ownership" / "ownership_release_policy.py"
    if policy_py.is_file():
        results.append(
            _copy_file(
                policy_py,
                pbj / "ownership" / "ownership_release_policy.py",
                dry_run=bool(args.dry_run),
                force=bool(args.force),
            )
        )
    _print_results(results)
    if args.rebuild and not args.dry_run:
        return _run_pbj_root_commands(pbj, OWNERSHIP_SNF_FULL_REBUILD, dry_run=False)
    if args.rebuild and args.dry_run:
        print("\n[dry-run] would run build_snf_owners_index + build_snf_owners_ccn_index + validate_ownership_linkage")
    elif not args.dry_run:
        print("\nTip: pass --rebuild to refresh state_owner_index.json.gz and owner search indexes")
    return 0


def cmd_ownership_rebuild(args: argparse.Namespace) -> int:
    """Rebuild /owners/* state lists and owner indexes in pbj-root."""
    pbj = resolve_pbj_root(args.pbj_root)
    if args.index_only:
        cmds = OWNERSHIP_INDEX_ONLY_REBUILD
    else:
        cmds = OWNERSHIP_SNF_FULL_REBUILD
    return _run_pbj_root_commands(pbj, cmds, dry_run=bool(args.dry_run))


def cmd_full_handoff(args: argparse.Namespace) -> int:
    """Run provider-release + optional PBJ metrics + optional SNF owners in one pass."""
    args.run_gates = not bool(getattr(args, "no_run_gates", False))
    args.rebuild = True
    rc = cmd_provider_release(args)
    if rc != 0:
        return rc
    if getattr(args, "with_pbj_metrics", False):
        metrics_args = argparse.Namespace(**{**vars(args), "rebuild": True})
        rc = cmd_pbj_metrics(metrics_args)
        if rc != 0:
            return rc
    if getattr(args, "with_snf_owners", False):
        snf_args = argparse.Namespace(**{**vars(args), "rebuild": True})
        src = _newest_snf_all_owners_csv(_ROOT)
        dst = _newest_snf_all_owners_csv(resolve_pbj_root(args.pbj_root))
        if src and dst:
            src_d = _parse_snf_owners_release_date(src) or date.min
            dst_d = _parse_snf_owners_release_date(dst) or date.min
            if src_d < dst_d and not args.force:
                print(
                    f"\nSNF skip: pbj-root {dst.name} is newer than PBJapp {src.name} "
                    "(use --force to overwrite)",
                    file=sys.stderr,
                )
            else:
                rc = cmd_snf_all_owners(snf_args)
                if rc != 0 and not args.force:
                    return rc
        elif src:
            rc = cmd_snf_all_owners(snf_args)
            if rc != 0:
                return rc
    print("\nRun `sync_to_pbj_root.py checklist` for remaining manual items.")
    return 0


def cmd_provider_norm(args: argparse.Namespace) -> int:
    pbj = resolve_pbj_root(args.pbj_root)
    key = release_key(*map(int, args.release_key.split("-")))
    handoff_path = cms_data_paths.provider_info_dir() / "_manifests" / key.label / "pbj_root_handoff.json"
    manifest = load_manifest(key, _ROOT)
    if handoff_path.is_file():
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        dest_rel = handoff.get("pbj_root_sync", {}).get("destination_file") or (
            f"provider_info/ProviderInfoNorm_{key.year}_{key.month:02d}.csv"
        )
        expected_sha = str(handoff.get("pbj_root_sync", {}).get("sha256") or "")
    else:
        dest_rel = f"provider_info/ProviderInfoNorm_{key.year}_{key.month:02d}.csv"
        expected_sha = ""
        handoff = {}
    src = cms_data_paths.provider_info_normalized_dir() / f"ProviderInfoNorm_{key.year}_{key.month:02d}.csv"
    dst = pbj / dest_rel.replace("/", os.sep)
    result = _copy_file(src, dst, dry_run=bool(args.dry_run), force=bool(args.force))
    if expected_sha and result.sha256 and result.sha256 != expected_sha:
        raise ValueError(
            f"Norm sha256 mismatch vs pbj_root_handoff.json: file={result.sha256} expected={expected_sha}"
        )
    nh_result = _copy_nh_snapshot_for_backfill(
        key, pbj, dry_run=bool(args.dry_run), force=bool(args.force)
    )
    results = [result]
    if nh_result is not None:
        results.append(nh_result)
    _print_results(results)
    if manifest and manifest.get("promotion_blocked") and not args.force:
        print("WARNING: release_manifest promotion_blocked=true — review before pbj-root commit", file=sys.stderr)
    if args.run_gates and not args.dry_run:
        rc = _run_pbj_root_gates(pbj, dry_run=False)
        if rc != 0:
            return rc
    if args.rebuild and not args.dry_run:
        return _rebuild_provider_derivatives(pbj, dry_run=False)
    if args.rebuild and args.dry_run:
        print("\n[dry-run] would run build_state_page_aggregates.py")
    return 0


def cmd_pbj_metrics(args: argparse.Namespace) -> int:
    pbj = resolve_pbj_root(args.pbj_root)
    force = bool(args.force) or bool(args.rebuild)
    results: list[CopyResult] = []
    for name in PBJ_METRICS_FILES:
        results.append(
            _copy_file(_ROOT / name, pbj / name, dry_run=bool(args.dry_run), force=force)
        )
    _print_results(results)
    if args.rebuild and not args.dry_run:
        verify_cmd = f'python scripts/verify_pbjapp_sync.py --source "{_ROOT}"'
        rc = _run_pbj_root_commands(pbj, (verify_cmd,), dry_run=False)
        if rc != 0:
            return rc
        rc = _rebuild_pbj_metrics_derivatives(pbj, dry_run=False)
        if rc != 0:
            return rc
        print(
            "\nPost-rebuild: state/national/region CSVs include LPN and *_Median columns "
            "patched in pbj-root (byte-identical to PBJapp is not expected)."
        )
        return 0
    if args.rebuild and args.dry_run:
        print(
            "\n[dry-run] would verify copied CSVs, then run "
            "patch_state_quarterly_lpn.py + patch_state_quarterly_medians.py"
        )
    elif not args.dry_run:
        verify = pbj / "scripts" / "verify_pbjapp_sync.py"
        if verify.is_file():
            print(f'\nTip: pass --rebuild or run: python scripts/verify_pbjapp_sync.py --source "{_ROOT}"')
    return 0


def cmd_chow_zip(args: argparse.Namespace) -> int:
    pbj = resolve_pbj_root(args.pbj_root)
    src = _ROOT / CHOW_CANONICAL_REL
    dst = pbj / "ownership" / CHOW_ZIP_BASENAME
    result = _copy_file(src, dst, dry_run=bool(args.dry_run), force=bool(args.force))
    _print_results([result])
    if args.rebuild_index and not args.dry_run:
        return _run_pbj_root_commands(pbj, CHOW_DERIVED, dry_run=False)
    if args.rebuild_index and args.dry_run:
        print("\n[dry-run] would run build_chow_index.py")
    elif not args.dry_run:
        print("\nTip: pass --rebuild-index to run build_chow_index.py in pbj-root")
    return 0


def cmd_chain_performance(args: argparse.Namespace) -> int:
    pbj = resolve_pbj_root(args.pbj_root)
    src = _ROOT / CHAIN_CANONICAL_REL
    dst = pbj / "ownership" / CHAIN_CSV_BASENAME
    result = _copy_file(src, dst, dry_run=bool(args.dry_run), force=bool(args.force))
    _print_results([result])
    print("\nNote: pbj-root load_chain_performance() accepts Chain_Performance_*.csv under ownership/.")
    return 0


def _run_pbj_root_commands(
    pbj: Path,
    commands: tuple[str, ...],
    *,
    dry_run: bool,
    pbjapp_source: Path | None = None,
) -> int:
    for cmd in commands:
        if pbjapp_source is not None:
            cmd = cmd.replace("{pbjapp}", str(pbjapp_source))
        print(f"\n>> {cmd}")
        if dry_run:
            continue
        rc = subprocess.call(cmd, shell=True, cwd=str(pbj))
        if rc != 0:
            print(f"ERROR: command failed (exit {rc}): {cmd}", file=sys.stderr)
            return rc
    return 0


def _run_pbj_root_gates(pbj: Path, *, dry_run: bool = False) -> int:
    return _run_pbj_root_commands(pbj, PROVIDER_NORM_GATES, dry_run=dry_run)


def _rebuild_pbj_metrics_derivatives(pbj: Path, *, dry_run: bool) -> int:
    """Recompute LPN columns and facility-level medians on state/region/national metrics."""
    return _run_pbj_root_commands(pbj, PBJ_METRICS_DERIVED, dry_run=dry_run)


def _rebuild_provider_derivatives(pbj: Path, *, dry_run: bool) -> int:
    """Rebuild state-page aggregates (case-mix medians, etc.) after provider Norm sync."""
    return _run_pbj_root_commands(pbj, PROVIDER_NORM_DERIVED, dry_run=dry_run)


def cmd_provider_release(args: argparse.Namespace) -> int:
    """Copy provider Norm + ownership sources and run pbj-root gates/derived rebuilds."""
    args.run_gates = not bool(getattr(args, "no_run_gates", False))
    args.rebuild = True
    rc = cmd_provider_norm(args)
    if rc != 0:
        return rc
    args.rebuild_index = not bool(getattr(args, "skip_chow_index", False))
    rc = cmd_chow_zip(args)
    if rc != 0:
        return rc
    return cmd_chain_performance(args)


def cmd_run_gates(args: argparse.Namespace) -> int:
    return _run_pbj_root_gates(resolve_pbj_root(args.pbj_root), dry_run=bool(args.dry_run))


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pbj-root", default="", help=f"pbj-root path (default: sibling or {PBJ_ROOT_ENV})")
    common.add_argument("--dry-run", action="store_true")
    common.add_argument("--force", action="store_true", help="Overwrite differing destination files")

    parser = argparse.ArgumentParser(description="Sync validated PBJapp artifacts to pbj-root")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", parents=[common], help="Show source vs pbj-root presence")
    p_status.set_defaults(func=cmd_status)

    p_norm = sub.add_parser("provider-norm", parents=[common], help="Copy ProviderInfoNorm for a release key")
    p_norm.add_argument("--release-key", required=True, help="YYYY-MM")
    p_norm.add_argument("--run-gates", action="store_true", help="Run pbj-root validate gates after copy")
    p_norm.add_argument(
        "--rebuild",
        action="store_true",
        help="Run build_state_page_aggregates.py (case-mix medians, state page bundle)",
    )
    p_norm.set_defaults(func=cmd_provider_norm)

    p_metrics = sub.add_parser("pbj-metrics", parents=[common], help="Copy quarterly metrics CSVs")
    p_metrics.add_argument(
        "--rebuild",
        action="store_true",
        help="Run patch_state_quarterly_lpn.py + patch_state_quarterly_medians.py in pbj-root",
    )
    p_metrics.set_defaults(func=cmd_pbj_metrics)

    p_chow = sub.add_parser("chow-zip", parents=[common], help="Copy canonical CHOW ZIP to pbj-root/ownership/")
    p_chow.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Run build_chow_index.py in pbj-root after copy",
    )
    p_chow.set_defaults(func=cmd_chow_zip)

    p_chain = sub.add_parser("chain-performance", parents=[common], help="Copy Chain_Performance CSV to pbj-root/ownership/")
    p_chain.set_defaults(func=cmd_chain_performance)

    p_release = sub.add_parser(
        "provider-release",
        parents=[common],
        help="Provider Norm + CHOW zip + chain CSV + provider gates/aggregates (monthly CMS release)",
    )
    p_release.add_argument("--release-key", required=True, help="YYYY-MM")
    p_release.add_argument("--no-run-gates", action="store_true", help="Skip provider Norm validate gates")
    p_release.add_argument(
        "--skip-chow-index",
        action="store_true",
        help="Skip build_chow_index.py (default: rebuild index after CHOW zip copy)",
    )
    p_release.set_defaults(func=cmd_provider_release, run_gates=True)

    p_checklist = sub.add_parser("checklist", parents=[common], help="Full handoff checklist with suggested commands")
    p_checklist.set_defaults(func=cmd_checklist)

    p_snf = sub.add_parser(
        "snf-all-owners",
        parents=[common],
        help="Copy newest SNF_All_Owners CSV and rebuild /owners/* state lists",
    )
    p_snf.add_argument(
        "--rebuild",
        action="store_true",
        help="Run build_snf_owners_index + ccn index + validate_ownership_linkage",
    )
    p_snf.set_defaults(func=cmd_snf_all_owners)

    p_own = sub.add_parser("ownership-rebuild", parents=[common], help="Rebuild owner indexes in pbj-root (no copy)")
    p_own.add_argument(
        "--index-only",
        action="store_true",
        help="Refresh derived gzip indexes only (--index-only; no SQLite rebuild)",
    )
    p_own.set_defaults(func=cmd_ownership_rebuild)

    p_full = sub.add_parser(
        "full-handoff",
        parents=[common],
        help="provider-release + optional pbj-metrics and SNF owners",
    )
    p_full.add_argument("--release-key", required=True, help="YYYY-MM")
    p_full.add_argument("--no-run-gates", action="store_true", help="Skip provider Norm validate gates")
    p_full.add_argument("--skip-chow-index", action="store_true", help="Skip build_chow_index.py")
    p_full.add_argument("--with-pbj-metrics", action="store_true", help="Also run pbj-metrics --rebuild")
    p_full.add_argument(
        "--with-snf-owners",
        action="store_true",
        help="Copy SNF_All_Owners when PBJapp has newest drop + rebuild state owner lists",
    )
    p_full.set_defaults(func=cmd_full_handoff, run_gates=True)

    p_gates = sub.add_parser("run-gates", parents=[common], help="Run pbj-root provider deploy gates only")
    p_gates.set_defaults(func=cmd_run_gates)

    args = parser.parse_args()
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
