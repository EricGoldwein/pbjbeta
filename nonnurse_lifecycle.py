"""Non-nurse normalization and complete-series validation for candidate promotion."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

from release_control_plane import ReleaseState, record_candidate


def _pbj_root() -> Path:
    return Path(os.environ.get("PBJ_REPO_ROOT") or Path(__file__).resolve().parent).resolve()


def normalize_validate_candidate(
    raw_path: Path,
    release_id: str,
    *,
    control_root: Path,
    pbj_root: Path | None = None,
    transform: Callable[[Path, Path], None] | None = None,
    validate_pair: Callable[[Path, Path], Any] | None = None,
) -> dict[str, Any]:
    pbj_root = (pbj_root or _pbj_root()).resolve()
    if not re.fullmatch(r"CY\d{4}Q[1-4]", release_id, re.I):
        raise RuntimeError(f"ambiguous non-nurse release identity: {release_id}")
    expected_name = f"PBJ_dailynonnursestaffing_{release_id.upper()}.csv"
    if raw_path.name.lower() != expected_name.lower():
        raise RuntimeError(f"raw filename does not match release {release_id}: {raw_path.name}")
    std_dir = pbj_root / "standardized_NonNurse"
    std_dir.mkdir(parents=True, exist_ok=True)
    std_path = std_dir / expected_name

    if transform is None or validate_pair is None:
        for path in (pbj_root, pbj_root / "scripts"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from standardize_nonnursepbj_files import standardize_column_names
        from pbj_standardize_validate import atomic_write_csv, read_pbj_csv, validate_dataframe_vs_raw_counts, validate_standardized_vs_raw

        def real_transform(raw: Path, output: Path) -> None:
            frame = read_pbj_csv(raw)
            standardized, _ = standardize_column_names(frame, str(raw))
            pre = validate_dataframe_vs_raw_counts(standardized, raw)
            if not pre.ok:
                raise RuntimeError(f"pre-write completeness failed: {pre.reason}")
            atomic_write_csv(standardized, output, encoding="utf-8")

        transform = real_transform
        validate_pair = validate_standardized_vs_raw

    transform(raw_path, std_path)
    result = validate_pair(raw_path, std_path)
    if not getattr(result, "ok", False):
        raise RuntimeError(f"standardized completeness failed: {getattr(result, 'reason', 'unknown')}")

    raw_dir = raw_path.parent
    raw_series = sorted(raw_dir.glob("PBJ_dailynonnursestaffing_CY*.csv"))
    if not raw_series:
        raise RuntimeError("non-nurse raw series is empty")
    source_set = []
    for raw_member in raw_series:
        match = re.search(r"CY\d{4}Q[1-4]", raw_member.name, re.I)
        if not match:
            continue
        standardized = std_dir / raw_member.name
        if not standardized.is_file():
            raise RuntimeError(f"incomplete standardized series: missing {standardized.name}")
        pair = validate_pair(raw_member, standardized)
        if not getattr(pair, "ok", False):
            raise RuntimeError(f"incomplete standardized series {standardized.name}: {getattr(pair, 'reason', 'unknown')}")
        source_set.append({"role": match.group(0).upper(), "source_path": str(standardized)})
    if release_id.upper() not in {item["role"] for item in source_set}:
        raise RuntimeError(f"validated series does not include {release_id}")
    record = record_candidate(
        "cms.pbj_non_nurse_staffing",
        release_id.upper(),
        ReleaseState.VALIDATED,
        source_path=std_path,
        validation={"status": "PASS", "series_members": len(source_set)},
        metadata={"source_set": source_set, "normalization": "PBJapp established non-nurse standardizer"},
        root=control_root,
    )
    return {"candidate": record, "standardized_path": str(std_path), "series_members": len(source_set)}

