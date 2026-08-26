"""Tests for pbj-root sync helper."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cms_provider_release_lib import release_key, write_pbj_root_handoff
from sync_to_pbj_root import (
    PROVIDER_NORM_DERIVED,
    _copy_file,
    _newest_snf_all_owners_csv,
    _parse_snf_owners_release_date,
    _provider_promotion_status,
    resolve_pbj_root,
)


def test_write_pbj_root_handoff_writes_expected_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("cms_provider_release_lib.cms_data_paths.repo_root", lambda root=None: tmp_path)
    monkeypatch.setattr(
        "cms_provider_release_lib.cms_data_paths.provider_info_normalized_dir",
        lambda root=None: tmp_path / "provider_info_normalized",
    )
    monkeypatch.setattr(
        "cms_provider_release_lib.cms_data_paths.provider_info_dir",
        lambda root=None: tmp_path / "provider_info",
    )
    monkeypatch.setattr(
        "cms_provider_release_lib.cms_data_paths.ownership_dir",
        lambda root=None: tmp_path / "ownership",
    )
    norm_dir = tmp_path / "provider_info_normalized"
    norm_dir.mkdir(parents=True)
    norm = norm_dir / "ProviderInfoNorm_2026_06.csv"
    norm.write_text("ccn,processing_date\n015009,2026-06-01\n", encoding="utf-8")
    key = release_key(2026, 6)
    out = write_pbj_root_handoff(key, tmp_path)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["release_key"] == "2026-06"
    assert data["pbj_root_sync"]["destination_file"] == "provider_info/ProviderInfoNorm_2026_06.csv"
    assert "verify_provider_release_handoff" in " ".join(data["gates_to_run_in_pbj_root"])
    assert "generate_search_index.py" in " ".join(data["derived_rebuilds_in_pbj_root"])
    assert data["provider_promotion"]["ready_for_pbj_commit"] is False
    assert data["sync_command"].startswith("python scripts/sync_to_pbj_root.py")


def test_copy_file_skip_identical(tmp_path: Path) -> None:
    src = tmp_path / "a.csv"
    dst = tmp_path / "b.csv"
    src.write_text("x\n", encoding="utf-8")
    dst.write_text("x\n", encoding="utf-8")
    result = _copy_file(src, dst, dry_run=False, force=False)
    assert result.action == "skip_identical"


def test_resolve_pbj_root_explicit(tmp_path: Path) -> None:
    p = resolve_pbj_root(str(tmp_path))
    assert p == tmp_path.resolve()


def test_parse_snf_owners_release_date_iso_and_month() -> None:
    assert _parse_snf_owners_release_date(Path("SNF_All_Owners_2026.04.01.csv")) == date(2026, 4, 1)
    assert _parse_snf_owners_release_date(Path("SNF_All_Owners_May_2026.csv")) == date(2026, 5, 1)


def test_policy_active_snf_all_owners_skips_facility_slices(tmp_path: Path) -> None:
    own = tmp_path / "ownership"
    own.mkdir()
    (own / "SNF_All_Owners_facility_335513.csv").write_text("x\n", encoding="utf-8")
    may = own / "SNF_All_Owners_May_2026.csv"
    april = own / "SNF_All_Owners_2026.04.01.csv"
    may.write_text("b\n", encoding="utf-8")
    april.write_text("a\n", encoding="utf-8")
    may_sha = hashlib.sha256(b"b\n").hexdigest()
    policy = {
        "active_release_date": "2026-05-01",
        "active_release_handoff": {
            "inbound_sources": [],
        },
        "releases": {
            "2026-05-01": {
                "ownership_source_filename": "SNF_All_Owners_May_2026.csv",
                "ownership_source_sha256": may_sha,
                "bridge_lookup_filename": "release_2026-05-01_lookup.json",
                "bridge_pairing_status": "exact_release_date_match",
                "status": "active",
            }
        },
    }
    (own / "ownership_release_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    assert _newest_snf_all_owners_csv(tmp_path) == may


def test_provider_promotion_not_ready_without_nh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sync_to_pbj_root as stpr

    monkeypatch.setattr(stpr.cms_data_paths, "repo_root", lambda root=None: tmp_path)
    monkeypatch.setattr(
        stpr.cms_data_paths,
        "provider_info_dir",
        lambda root=None: tmp_path / "provider_info",
    )
    monkeypatch.setattr(
        stpr.cms_data_paths,
        "provider_info_normalized_dir",
        lambda root=None: tmp_path / "provider_info_normalized",
    )
    norm_dir = tmp_path / "provider_info_normalized"
    norm_dir.mkdir(parents=True)
    (norm_dir / "ProviderInfoNorm_2026_06.csv").write_text("ccn\n1\n", encoding="utf-8")
    key = release_key(2026, 6)
    status = _provider_promotion_status(key)
    assert status["norm_present"] is True
    assert status["nh_present"] is False
    assert status["ready_for_pbj_commit"] is False
    assert "self-check" in status["validate_mode"]


def test_provider_norm_derived_includes_search_index() -> None:
    assert any("generate_search_index" in cmd for cmd in PROVIDER_NORM_DERIVED)
