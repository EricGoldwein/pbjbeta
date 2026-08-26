"""Tests for build_provider_info_combined validation gates."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_provider_info_combined import build_provider_info_combined, validate_norm_inputs


def _write_norm(dir_path: Path, name: str, rows: list[dict]) -> Path:
    p = dir_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_validate_rejects_within_file_duplicate_keys(tmp_path: Path) -> None:
    p = _write_norm(
        tmp_path,
        "ProviderInfoNorm_2026_06.csv",
        [
            {"ccn": "015009", "processing_date": "2026-06-01", "provider_name": "A"},
            {"ccn": "015009", "processing_date": "2026-06-01", "provider_name": "B"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_norm_inputs([p])


def test_validate_rejects_column_mismatch(tmp_path: Path) -> None:
    a = _write_norm(tmp_path, "ProviderInfoNorm_2026_05.csv", [{"ccn": "1", "processing_date": "2026-05-01"}])
    b = _write_norm(
        tmp_path,
        "ProviderInfoNorm_2026_06.csv",
        [{"ccn": "1", "processing_date": "2026-06-01", "extra_col": "x"}],
    )
    with pytest.raises(ValueError, match="column mismatch"):
        validate_norm_inputs([a, b])


def test_build_requires_month_when_asked(tmp_path: Path) -> None:
    _write_norm(tmp_path, "ProviderInfoNorm_2026_05.csv", [{"ccn": "015009", "processing_date": "2026-05-01"}])
    with pytest.raises(FileNotFoundError, match="required normalized month missing"):
        build_provider_info_combined(norm_paths=sorted(tmp_path.glob("*.csv")), require_month="2026-06")


def test_build_dedupes_across_months(tmp_path: Path) -> None:
    _write_norm(tmp_path, "ProviderInfoNorm_2026_05.csv", [{"ccn": "015009", "processing_date": "2026-05-01"}])
    _write_norm(tmp_path, "ProviderInfoNorm_2026_06.csv", [{"ccn": "015009", "processing_date": "2026-06-01"}])
    out = build_provider_info_combined(norm_paths=sorted(tmp_path.glob("*.csv")))
    assert len(out) == 2
