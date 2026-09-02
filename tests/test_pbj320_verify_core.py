"""Tests for generic verification core."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pbj320_verify_core import (
    load_baseline_norm_rows,
    prepare_norm_release_diff_candidates,
    prior_release_id,
    resolve_stage_artifact_cache_dir,
)

DATA_OPS_ROOT = ROOT
PBJ_ROOT = DATA_OPS_ROOT.parent / "pbj-root"
PI_2026_08_CACHE = DATA_OPS_ROOT / "state/pbj320_stage_artifacts/cms.provider_info/2026-08"
PI_2026_08_PUBLISH_BASE_SHA = "a2eec47b040986698e7e0f5b4749c53c19f7e22b"


def test_prior_release_id_december_rollover() -> None:
    assert prior_release_id("2026-01") == "2025-12"
    assert prior_release_id("2026-08") == "2026-07"


def test_load_baseline_norm_falls_back_to_prior_month(tmp_path: Path, monkeypatch) -> None:
    import pbj320_verify_core as core

    dev = tmp_path / "pbj-root"
    dev.mkdir()
    (dev / ".git").mkdir()

    july_csv = (
        "ccn,provider_name,overall_rating,health_inspection_rating,sff_status,processing_date\n"
        "015112,OLD,3,3,,2026-07-01\n"
    ).encode("utf-8")

    def _fake_git_show(_dev, sha, rel_path):
        if rel_path.endswith("ProviderInfoNorm_2026_07.csv"):
            return july_csv
        return None

    monkeypatch.setattr(core, "git_show_bytes", _fake_git_show)
    rows = load_baseline_norm_rows(
        dev_pbj_root=dev,
        baseline_sha="abc123",
        norm_rel="provider_info/ProviderInfoNorm_2026_08.csv",
        release_id="2026-08",
    )
    assert rows["015112"]["overall_rating"] == "3"


def test_prepare_norm_diff_prior_month_baseline_and_stage_cache(tmp_path: Path, monkeypatch) -> None:
    """Regression: candidates > 0 when Aug Norm absent at publish base, July baseline + stage cache expected."""
    import pbj320_verify_core as core

    dev = tmp_path / "pbj-root"
    dev.mkdir()
    (dev / ".git").mkdir()

    july_csv = (
        "ccn,provider_name,overall_rating,health_inspection_rating,sff_status,processing_date\n"
        "015112,OLD NAME,3,3,,2026-07-01\n"
        "015113,STABLE,4,4,,2026-07-01\n"
    ).encode("utf-8")

    cache = tmp_path / "stage_cache"
    norm_dir = cache / "provider_info"
    norm_dir.mkdir(parents=True)
    aug_path = norm_dir / "ProviderInfoNorm_2026_08.csv"
    aug_path.write_text(
        "ccn,provider_name,overall_rating,health_inspection_rating,sff_status,processing_date\n"
        "015112,NEW NAME,5,4,,2026-08-01\n"
        "015113,STABLE,4,4,,2026-08-01\n",
        encoding="utf-8",
    )

    def _fake_git_show(_dev, sha, rel_path):
        if sha != "base0001":
            return None
        if rel_path.endswith("ProviderInfoNorm_2026_08.csv"):
            return None
        if rel_path.endswith("ProviderInfoNorm_2026_07.csv"):
            return july_csv
        return None

    monkeypatch.setattr(core, "git_show_bytes", _fake_git_show)

    manifest = {
        "publication_base_sha": "base0001",
        "stage_artifact_cache": str(cache),
        "artifacts": [{"destination_id": "provider_norm", "path": "provider_info/ProviderInfoNorm_2026_08.csv"}],
    }

    candidates, provenance = prepare_norm_release_diff_candidates(
        manifest=manifest,
        release_id="2026-08",
        dev_pbj_root=dev,
    )

    assert provenance["baseline"]["source_mode"] == "git_show_prior_month_at_publication_base"
    assert provenance["baseline"]["source_rel"].endswith("ProviderInfoNorm_2026_07.csv")
    assert provenance["expected"]["source_mode"] == "stage_artifact_cache"
    assert provenance["expected"]["row_count"] > 0
    assert len(candidates) > 0
    assert any(c["ccn"] == "015112" and "provider_name" in c["fields"] for c in candidates)


def test_stage_cache_from_manifest_ignores_control_plane_env(tmp_path: Path, monkeypatch) -> None:
    """Manifest absolute stage_artifact_cache must resolve even when registry env points elsewhere."""
    cache = tmp_path / "real_cache"
    cache.mkdir()
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(tmp_path / "other_registry"))

    manifest = {"stage_artifact_cache": str(cache)}
    assert resolve_stage_artifact_cache_dir(manifest) == cache


@pytest.mark.integration
def test_prepare_norm_diff_2026_08_git_baseline_and_local_stage_cache() -> None:
    """Live regression against operator paths: a2eec47 July baseline vs 2026-08 stage artifact cache."""
    import pbj320_verify_core as core

    aug_norm = PI_2026_08_CACHE / "provider_info/ProviderInfoNorm_2026_08.csv"
    if not (PBJ_ROOT / ".git").is_dir() or not aug_norm.is_file():
        pytest.skip("pbj-root git or 2026-08 stage cache not available on this machine")

    aug_at_base = core.git_show_bytes(
        PBJ_ROOT,
        PI_2026_08_PUBLISH_BASE_SHA,
        "provider_info/ProviderInfoNorm_2026_08.csv",
    )
    july_at_base = core.git_show_bytes(
        PBJ_ROOT,
        PI_2026_08_PUBLISH_BASE_SHA,
        "provider_info/ProviderInfoNorm_2026_07.csv",
    )
    if aug_at_base is not None or july_at_base is None:
        pytest.skip("publication base SHA no longer matches expected July-only baseline scenario")

    manifest = {
        "publication_base_sha": PI_2026_08_PUBLISH_BASE_SHA,
        "stage_artifact_cache": str(PI_2026_08_CACHE),
        "artifacts": [{"destination_id": "provider_norm", "path": "provider_info/ProviderInfoNorm_2026_08.csv"}],
    }
    pub = {"dev_pbj_root": str(PBJ_ROOT), "publish_base_sha": PI_2026_08_PUBLISH_BASE_SHA}

    candidates, provenance = prepare_norm_release_diff_candidates(
        manifest=manifest,
        release_id="2026-08",
        dev_pbj_root=core.resolve_dev_pbj_root_for_verify(publication_record=pub),
        publication_record=pub,
    )

    assert provenance["baseline"]["source_mode"] == "git_show_prior_month_at_publication_base"
    assert provenance["expected"]["source_mode"] == "stage_artifact_cache"
    assert provenance["expected"]["row_count"] > 0
    assert len(candidates) > 0
