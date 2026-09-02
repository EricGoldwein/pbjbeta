"""Tests for production verification and publication contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _full_pi_manifest() -> dict:
    return {
        "schema_version": 2,
        "publication_contract_version": 2,
        "status": "STAGED",
        "active_release_id": "2026-08",
        "publication_base_sha": "a" * 40,
        "validation_gates": [{"command": "gate", "passed": True, "summary": "ok"}],
        "artifacts": [
            {
                "destination_id": "provider_norm",
                "publication_class": "commit_destination",
                "path": "provider_info/ProviderInfoNorm_2026_08.csv",
                "proposed_sha256": "1" * 64,
            },
            {
                "destination_id": "provider_combined_latest",
                "publication_class": "commit_destination",
                "path": "provider_info_combined_latest.csv",
                "proposed_sha256": "2" * 64,
            },
            {
                "destination_id": "state_page_aggregates",
                "publication_class": "shared_derived",
                "path": "data/state_page_aggregates.json.gz",
                "proposed_sha256": "3" * 64,
                "inputs": [{"mode": "GOVERNED_OVERLAY", "sha256": "x", "source_id": "cms.provider_info"}],
            },
            {
                "destination_id": "search_index",
                "publication_class": "shared_derived",
                "path": "search_index.json",
                "proposed_sha256": "5919a13dc8c69c807f91984a5cd6444eef20400728f6d57810636317a300ae68",
                "inputs": [
                    {"mode": "GOVERNED_OVERLAY", "sha256": "x", "source_id": "cms.provider_info"},
                    {"mode": "PUBLISHED_BASELINE", "sha256": "y", "source_id": "cms.sff_pdf_list"},
                ],
            },
        ],
    }


def test_contract_marks_schema_v1_stale() -> None:
    from pbj320_publication_contract import evaluate_stage_publish_eligibility
    from pbj320_source_adapters import PI_STAGE_PUBLISH_SPEC

    result = evaluate_stage_publish_eligibility(
        {"schema_version": 1, "status": "STAGED"},
        PI_STAGE_PUBLISH_SPEC,
    )
    assert result["stale"] is True
    assert result["publishable"] is False


def test_merge_destination_layers_publication_supersedes_stage() -> None:
    from pbj320_publication_contract import merge_destination_layers_for_display

    stage = {"destination_layers": {"committed": "NO", "pushed": "NO", "pbj320_destination": "STAGED"}}
    pub = {
        "push_succeeded": True,
        "destination_layers": {
            "committed": "YES",
            "pushed": "YES",
            "deployed": "YES",
            "production_verified": "YES",
        },
    }
    merged = merge_destination_layers_for_display(stage, pub)
    assert merged["committed"] == "YES"
    assert merged["pushed"] == "YES"
    assert merged["production_verified"] == "YES"
    assert merged["pbj320_destination"] == "STAGED"


def test_publication_state_summary_pushed_not_staged_only() -> None:
    from pbj320_publication_contract import merge_destination_layers_for_display, publication_state_summary

    stage = {"destination_layers": {"pbj320_destination": "STAGED", "committed": "NO", "pushed": "NO"}}
    pub = {"push_succeeded": True, "destination_layers": {"committed": "YES", "pushed": "YES"}}
    merged = merge_destination_layers_for_display(stage, pub)
    summary = publication_state_summary(merged, pub)
    assert summary["headline"] == "Published to PBJ320."
    assert "not committed" not in summary["detail"].lower()
    assert "committed · pushed" in summary["trail"]
    assert summary["pbj320_destination"] == "STAGED"


def test_derive_norm_release_diff_candidates_from_csv() -> None:
    from pbj320_verify_core import derive_norm_release_diff_candidates

    baseline = {
        "015112": {
            "provider_name": "Old Name",
            "overall_rating": "3",
            "health_inspection_rating": "3",
            "sff_status": "",
            "processing_date": "2026-07-01",
        }
    }
    expected = {
        "015112": {
            "provider_name": "Old Name",
            "overall_rating": "1",
            "health_inspection_rating": "1",
            "sff_status": "",
            "processing_date": "2026-08-01",
        },
        "015134": {
            "provider_name": "Highlands Rehabilitation and Wellness Center",
            "overall_rating": "2",
            "health_inspection_rating": "2",
            "sff_status": "",
            "processing_date": "2026-08-01",
        },
    }
    candidates = derive_norm_release_diff_candidates(
        baseline_rows=baseline,
        expected_rows=expected,
        processing_prefix="2026-08",
    )
    by_ccn = {c["ccn"]: c for c in candidates}
    assert "015112" in by_ccn
    assert "overall_rating" in by_ccn["015112"]["fields"]
    assert "015134" in by_ccn
    assert by_ccn["015134"]["expected"]["provider_name"].startswith("Highlands")


def test_verify_data_level_uses_api_for_ratings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pbj320_verify_core import verify_data_level_release_diff

    live_index = {"f": [{"c": "015112", "n": "Test Facility", "h": ""}]}
    candidates = [
        {
            "ccn": "015112",
            "fields": ["overall_rating", "provider_name"],
            "expected": {"overall_rating": "1", "provider_name": "Test Facility"},
        }
    ]

    def _fake_http(url: str, timeout: float = 45.0) -> tuple[int, bytes]:
        if "/api/public/provider/" in url:
            payload = json.dumps({"facility": {"name": "Test Facility"}, "cms_ratings": {"overall": "1"}})
            return 200, payload.encode("utf-8")
        return 404, b""

    monkeypatch.setattr("pbj320_verify_core.http_get_bytes", _fake_http)
    ok, ccn, field, surfaces = verify_data_level_release_diff(
        origin="https://www.pbj320.com",
        candidates=candidates,
        live_index=live_index,
    )
    assert ok is True
    assert ccn == "015112"
    assert field == "overall_rating"
    assert surfaces[0].startswith("api_provider")


def test_verify_requires_pushed_publication(tmp_path: Path) -> None:
    from pbj320_verify_production import ProviderInfoVerifyError, verify_provider_info_production

    with pytest.raises(ProviderInfoVerifyError, match="not pushed"):
        verify_provider_info_production("2026-08", root=tmp_path)


def test_verify_records_verified_only_when_checks_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import release_control_plane as rcp
    from pbj320_publication import stage_artifact_cache_path
    from pbj320_publish_provider_info import publication_record_path
    from pbj320_stage_provider_info import stage_manifest_path
    from pbj320_verify_production import verify_provider_info_production

    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)
    pub_path = publication_record_path("2026-08", root=tmp_path)
    pub_path.parent.mkdir(parents=True, exist_ok=True)
    pub_path.write_text(
        json.dumps(
            {
                "push_succeeded": True,
                "commit_sha": "18f7680e34593acf837dd9558cd0c9c2553261d3",
                "push_remote": "origin",
                "push_branch": "master",
                "publish_base_sha": "a" * 40,
                "destination_layers": {"committed": "YES", "pushed": "YES"},
            }
        ),
        encoding="utf-8",
    )
    manifest_path = stage_manifest_path("2026-08", root=tmp_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    cache = stage_artifact_cache_path("cms.provider_info", "2026-08", root=tmp_path)
    cache.mkdir(parents=True, exist_ok=True)
    search_body = json.dumps(
        {
            "f": [
                {"c": "015112", "n": "TEST", "h": ""},
            ]
        }
    )
    import hashlib

    live_bytes = search_body.encode("utf-8")
    expected_sha = hashlib.sha256(live_bytes).hexdigest()
    manifest = _full_pi_manifest()
    manifest["artifacts"][-1]["proposed_sha256"] = expected_sha
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    search_path = cache / "search_index.json"
    search_path.write_text(search_body, encoding="utf-8")
    norm_dir = cache / "provider_info"
    norm_dir.mkdir(exist_ok=True)
    norm_dir.joinpath("ProviderInfoNorm_2026_08.csv").write_text(
        "ccn,provider_name,overall_rating,health_inspection_rating,sff_status,processing_date\n"
        "015112,TEST,1,1,,2026-08-01\n",
        encoding="utf-8",
    )
    (cache / "data").mkdir(exist_ok=True)
    (cache / "data" / "state_page_aggregates.json.gz").write_bytes(b"agg")

    def _fake_http(url: str, timeout: float = 45.0) -> tuple[int, bytes]:
        if url.endswith("/search_index.json"):
            return 200, live_bytes
        if "/api/public/provider/" in url:
            payload = json.dumps({"facility": {"name": "TEST"}, "cms_ratings": {"overall": "1", "health_inspection": "1"}})
            return 200, payload.encode("utf-8")
        return 404, b""

    monkeypatch.setattr("pbj320_verify_core.http_get_bytes", _fake_http)
    monkeypatch.setattr("pbj320_verify_core.git_commit_on_branch", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "pbj320_verify_core.load_baseline_norm_rows",
        lambda **_k: {
            "015112": {
                "provider_name": "TEST",
                "overall_rating": "3",
                "health_inspection_rating": "3",
                "sff_status": "",
                "processing_date": "2026-07-01",
            }
        },
    )

    (tmp_path / "pbj-root").mkdir()
    result = verify_provider_info_production("2026-08", root=tmp_path, pbj_root=tmp_path / "pbj-root")
    assert result["all_pass"] is True
    saved = json.loads(pub_path.read_text(encoding="utf-8"))
    assert saved["production_verified"] is True
    assert saved["destination_layers"]["production_verified"] == "YES"


def test_premium_only_never_exposes_publish_spec() -> None:
    from pbj320_source_adapters import is_premium_only_source, stage_publish_spec

    assert is_premium_only_source("cms.health_citations") is True
    assert stage_publish_spec("cms.health_citations") is None


def test_stage_adapters_not_ready_without_artifacts_on_failure(tmp_path: Path) -> None:
    from pbj320_stage_adapters import NOT_READY_STATUS, stage_sff_for_pbj320

    result = stage_sff_for_pbj320(release_id="2099-01", root=tmp_path / "missing")
    assert result["status"] == NOT_READY_STATUS
    assert result["artifacts"] == []
    assert result["publishable"] is False
