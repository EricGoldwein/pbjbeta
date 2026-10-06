"""Regression: legacy PI Stage manifest must not expose Publish."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_ops_app import create_app


def _write_legacy_pi_stage_manifest(tmp_path: Path, release_id: str = "2026-08") -> Path:
    import release_control_plane as rcp

    manifest_dir = rcp._state_dir(tmp_path) / "pbj320_stages" / "cms.provider_info"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{release_id}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_id": "cms.provider_info",
                "active_release_id": release_id,
                "status": "STAGED",
                "stage_timestamp": "2026-08-28T12:00:00+00:00",
                "pbj_root": str(tmp_path / "pbj-root"),
                "canonical": {"sha256": "abc123", "state": "CURRENT"},
                "destination_layers": {"pbj320_destination": "STAGED"},
                "files_added": ["provider_info/ProviderInfoNorm_2026_08.csv"],
                "files_modified": [],
                "artifacts": [
                    {
                        "destination_id": "provider_norm",
                        "publication_class": "commit_destination",
                        "path": "provider_info/ProviderInfoNorm_2026_08.csv",
                        "proposed_sha256": "deadbeef",
                        "material_change": True,
                    }
                ],
                "validation_gates": [{"command": "fake-gate", "passed": True, "summary": "ok"}],
                "next_human_step": "Review stage manifest, then commit.",
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


@pytest.mark.parametrize('byte_verified', [False, True])
def test_schema_v1_manifest_next_action_is_restage_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, byte_verified: bool
) -> None:
    import cms_data_ops as ops

    _write_legacy_pi_stage_manifest(tmp_path)
    monkeypatch.setattr(
        "pbj320_stage_provider_info.audit_provider_info_pbj320_destination",
        lambda **_: {
            "source_id": "cms.provider_info",
            "active_release_id": "2026-08",
            "canonical_current": True,
            "destination_staged": True,
        },
    )

    action = ops._next_operator_action(
        "cms.provider_info",
        record={"actions_enabled": ["check_cms"]},
        snapshot={"validation_status": "PASS"},
        control_row={
            "active": {"active_release_id": "2026-08", "status": "ACTIVE"},
            "pending": None,
        },
        release_availability={"new_release_available": False, "publisher_latest_label": "Aug 2026", "cms_byte_verified_current": byte_verified},
        root=tmp_path,
    )

    assert action["label"] == ("Stage again against current production" if byte_verified else "Check CMS")
    assert action["endpoint"] == ("action_pi_stage_pbj320" if byte_verified else "action_pi_check")
    assert "Publish to PBJ320" not in action["label"]


def test_publication_state_summary_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import release_control_plane as rcp
    from pbj320_publication_contract import merge_destination_layers_for_display, publication_state_summary

    _write_legacy_pi_stage_manifest(tmp_path)
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)

    pub_dir = rcp._state_dir(tmp_path) / "pbj320_publications" / "cms.provider_info"
    pub_dir.mkdir(parents=True, exist_ok=True)
    pub_path = pub_dir / "2026-08.json"
    pub_path.write_text(
        json.dumps(
            {
                "push_succeeded": True,
                "production_verified": True,
                "destination_layers": {
                    "committed": "YES",
                    "pushed": "YES",
                    "deployed": "YES",
                    "production_verified": "YES",
                },
            }
        ),
        encoding="utf-8",
    )

    manifest_path = rcp._state_dir(tmp_path) / "pbj320_stages" / "cms.provider_info" / "2026-08.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    publication = json.loads(pub_path.read_text(encoding="utf-8"))
    merged = merge_destination_layers_for_display(manifest, publication)
    summary = publication_state_summary(merged, publication)
    assert summary["headline"] == "Production verified."
    assert "Staged only" not in summary["headline"]
    assert merged["pbj320_destination"] == "STAGED"


def test_legacy_pi_stage_manifest_renders_200_with_restage_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import release_control_plane as rcp

    _write_legacy_pi_stage_manifest(tmp_path)
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)
    monkeypatch.setenv("PBJ_DATA_OPS_PASSWORD", "test-password")

    build_called = {"count": 0}

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        build_called["count"] += 1
        raise AssertionError("build_publish_review_context must not run for stale manifest")

    monkeypatch.setattr(
        "pbj320_publish_provider_info.build_publish_review_context",
        _fail_if_called,
    )

    app = create_app()
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["data_ops_authenticated"] = True

    resp = client.get("/provider-info/stage-manifest/2026-08")
    html = resp.get_data(as_text=True)

    assert resp.status_code == 200
    assert build_called["count"] == 0
    assert "Stage candidate out of date" in html
    assert "Stage again against current production" in html
    assert "publication_base_sha" in html
    assert "I confirm Publish to PBJ320" not in html
    assert "Publish unavailable" in html
    assert "Staged only" not in html
