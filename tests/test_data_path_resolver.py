"""Tests for data_path_resolver and cms_data_paths delegation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cms_data_paths
import data_path_resolver
from data_path_resolver import resolve_data_path


REPO = Path(__file__).resolve().parents[1]


def _clear_path_env(monkeypatch) -> None:
    for name in list(data_path_resolver.ENV_KEYS.values()) + [
        data_path_resolver.GLOBAL_DATA_ROOT_ENV,
        data_path_resolver.GLOBAL_REPO_ROOT_ENV,
    ]:
        monkeypatch.delenv(name, raising=False)


def test_resolve_repo_default_without_overrides(monkeypatch, tmp_path: Path) -> None:
    _clear_path_env(monkeypatch)
    resolved = resolve_data_path("pbjcsv", root=tmp_path)
    assert resolved.source == "repo_default"
    assert resolved.path == (tmp_path / "PBJcsv").resolve()


def test_resolve_env_override(monkeypatch, tmp_path: Path) -> None:
    _clear_path_env(monkeypatch)
    override = tmp_path / "external" / "PBJcsv"
    monkeypatch.setenv("PBJ_PBJCSV", str(override))
    resolved = resolve_data_path("pbjcsv", root=tmp_path)
    assert resolved.source == "env"
    assert resolved.path == override.resolve()


def test_resolve_local_json_override(monkeypatch, tmp_path: Path) -> None:
    _clear_path_env(monkeypatch)
    custom = tmp_path / "alt" / "EIN"
    custom.mkdir(parents=True)
    cfg = tmp_path / "data_paths.local.json"
    cfg.write_text(json.dumps({"ein": str(custom)}), encoding="utf-8")
    resolved = resolve_data_path("ein", root=tmp_path)
    assert resolved.source == "local_json"
    assert resolved.path == custom.resolve()


def test_resolve_data_root_fallback(monkeypatch, tmp_path: Path) -> None:
    _clear_path_env(monkeypatch)
    data_root = tmp_path / "D_PBJdata"
    data_root.mkdir()
    monkeypatch.setenv("PBJ_DATA_ROOT", str(data_root))
    resolved = resolve_data_path("standardized_nonnurse", root=tmp_path)
    assert resolved.source == "data_root"
    assert resolved.path == (data_root / "standardized_NonNurse").resolve()


def test_cms_data_paths_default_matches_repo_relative(monkeypatch) -> None:
    _clear_path_env(monkeypatch)
    root = cms_data_paths.repo_root()
    assert cms_data_paths.nurse_raw_dir() == (root / "PBJcsv").resolve()
    assert cms_data_paths.standardized_nurse_dir() == (root / "standardized_PBJ").resolve()
    assert cms_data_paths.nonnurse_raw_dir() == (root / "NonNursecsv").resolve()
    assert cms_data_paths.standardized_nonnurse_dir() == (root / "standardized_NonNurse").resolve()
    assert cms_data_paths.ein_root() == (root / "EIN").resolve()
    assert cms_data_paths.provider_info_dir() == (root / "provider_info").resolve()
    assert cms_data_paths.provider_info_normalized_dir() == (root / "provider_info_normalized").resolve()
    assert cms_data_paths.provider_info_extracted_dir() == (root / "provider_info_extracted").resolve()
    assert cms_data_paths.indexed_dir() == (root / "indexed").resolve()
    assert cms_data_paths.metrics_backups_dir() == (root / "metrics_backups").resolve()
    assert cms_data_paths.deployments_dir() == (root / "deployments").resolve()
    assert cms_data_paths.facility_deploy_dir("315461") == (root / "deployments" / "pbj320-315461").resolve()


def test_cms_data_paths_explicit_root_ignores_env(monkeypatch, tmp_path: Path) -> None:
    _clear_path_env(monkeypatch)
    monkeypatch.setenv("PBJ_PBJCSV", str(tmp_path / "env_should_not_win"))
    assert cms_data_paths.nurse_raw_dir(root=tmp_path) == (tmp_path / "PBJcsv").resolve()


def test_ein_subdirs_use_resolved_ein_root(monkeypatch, tmp_path: Path) -> None:
    _clear_path_env(monkeypatch)
    ein = tmp_path / "EIN"
    assert cms_data_paths.ein_monolithic_dir(root=tmp_path) == (ein / "monolithic").resolve()
    assert cms_data_paths.ein_quarters_dir(root=tmp_path) == (ein / "quarters").resolve()
    assert cms_data_paths.ein_staging_dir(root=tmp_path) == (ein / "staging").resolve()
    assert cms_data_paths.ein_extracted_dir(root=tmp_path) == (ein / "extracted").resolve()
    assert cms_data_paths.ein_quarters_manifest(root=tmp_path) == (ein / "quarters" / "manifest.json").resolve()


def test_citations_dir_not_delegated_to_resolver(monkeypatch, tmp_path: Path) -> None:
    _clear_path_env(monkeypatch)
    assert cms_data_paths.citations_dir(root=tmp_path) == (tmp_path / "Citations").resolve()


def test_optional_repo_root_canonical_returns_none(monkeypatch) -> None:
    _clear_path_env(monkeypatch)
    root = cms_data_paths.repo_root()
    assert cms_data_paths.optional_repo_root(root) is None
    assert cms_data_paths.optional_repo_root(str(root)) is None


def test_optional_repo_root_non_canonical_returns_path(tmp_path: Path) -> None:
    assert cms_data_paths.optional_repo_root(tmp_path) == tmp_path.resolve()
