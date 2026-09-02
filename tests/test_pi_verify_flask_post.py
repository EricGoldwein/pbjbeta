"""Flask integration: POST /actions/provider-info/verify-production uses shared verifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_ops_app import create_app


def _write_pi_verify_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import release_control_plane as rcp
    from pbj320_publication import stage_artifact_cache_path
    from pbj320_publish_provider_info import publication_record_path
    from pbj320_stage_provider_info import stage_manifest_path

    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)

    manifest = {
        "schema_version": 2,
        "publication_contract_version": 2,
        "status": "STAGED",
        "active_release_id": "2026-08",
        "publication_base_sha": "a" * 40,
        "validation_gates": [{"command": "gate", "passed": True, "summary": "ok"}],
        "artifacts": [
            {
                "destination_id": "provider_norm",
                "path": "provider_info/ProviderInfoNorm_2026_08.csv",
                "proposed_sha256": "1" * 64,
            },
            {
                "destination_id": "search_index",
                "path": "search_index.json",
                "proposed_sha256": "2" * 64,
            },
        ],
    }
    manifest_path = stage_manifest_path("2026-08", root=tmp_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    cache = stage_artifact_cache_path("cms.provider_info", "2026-08", root=tmp_path)
    cache.mkdir(parents=True, exist_ok=True)
    search_body = json.dumps({"f": [{"c": "015112", "n": "TEST", "h": ""}]})
    import hashlib

    live_bytes = search_body.encode("utf-8")
    expected_sha = hashlib.sha256(live_bytes).hexdigest()
    manifest["artifacts"][1]["proposed_sha256"] = expected_sha
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    pub_path = publication_record_path("2026-08", root=tmp_path)
    pub_path.parent.mkdir(parents=True, exist_ok=True)
    pub_path.write_text(
        json.dumps(
            {
                "push_succeeded": True,
                "commit_sha": "c" * 40,
                "push_remote": "origin",
                "push_branch": "master",
                "publish_base_sha": "a" * 40,
                "destination_layers": {"committed": "YES", "pushed": "YES"},
            }
        ),
        encoding="utf-8",
    )

    (cache / "search_index.json").write_text(search_body, encoding="utf-8")
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
            payload = json.dumps({"facility": {"name": "TEST"}, "cms_ratings": {"overall": "1"}})
            return 200, payload.encode("utf-8")
        return 404, b""

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
    monkeypatch.setattr("pbj320_verify_core.http_get_bytes", _fake_http)


def test_verify_production_post_persists_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cms_data_paths
    import release_control_plane as rcp
    from pbj320_publish_provider_info import publication_record_path
    from pbj320_verify_production import verify_provider_info_production as _verify_pi

    (tmp_path / "pbj-root").mkdir()
    monkeypatch.setattr(
        "pbj320_stage_provider_info._resolve_pbj_root",
        lambda explicit=None: tmp_path / "pbj-root",
    )
    monkeypatch.setattr(rcp, "control_plane_root", lambda data_root=None: tmp_path)
    monkeypatch.setattr(cms_data_paths, "repo_root", lambda: tmp_path)

    _write_pi_verify_fixtures(tmp_path, monkeypatch)

    def _verify_with_fixture_root(release_id: str, *, dry_run: bool = False, **kwargs: object):
        return _verify_pi(
            release_id,
            root=tmp_path,
            pbj_root=tmp_path / "pbj-root",
            dry_run=dry_run,
        )

    monkeypatch.setattr(
        "pbj320_verify_production.verify_provider_info_production",
        _verify_with_fixture_root,
    )
    monkeypatch.delenv("PBJ_ACTIVE_RELEASE_REGISTRY", raising=False)
    monkeypatch.setenv("PBJ_DATA_OPS_PASSWORD", "test-password")

    app = create_app()
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["data_ops_authenticated"] = True

    resp = client.post(
        "/actions/provider-info/verify-production",
        data={"release_id": "2026-08"},
        follow_redirects=False,
    )
    assert resp.status_code in {302, 303}

    pub = json.loads(publication_record_path("2026-08", root=tmp_path).read_text(encoding="utf-8"))
    checks = {c["check_id"]: c["result"] for c in pub.get("production_verification_checks") or []}
    assert checks, f"no checks persisted: {pub.keys()}"
    assert checks.get("data_level_release_diff_visible") == "PASS"
    assert pub.get("production_verified") is True
    assert pub["destination_layers"]["deployed"] == "YES"
    assert pub["destination_layers"]["production_verified"] == "YES"

    follow = client.get("/provider-info/stage-manifest/2026-08")
    html = follow.get_data(as_text=True)
    assert "Production verified" in html or "Published to PBJ320" in html
    assert "Staged only" not in html
