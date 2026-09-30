from __future__ import annotations

from pathlib import Path

import pytest

import data_ops_app as app_module
from data_ops_app import create_app


def _client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PBJ_DATA_OPS_PASSWORD", "test-ops-pw")
    monkeypatch.setenv("PBJ_DATA_OPS_SECRET", "test-secret")
    app = create_app()
    client = app.test_client()
    assert client.post("/login", data={"password": "test-ops-pw"}).status_code == 302
    return client


def test_check_cms_releases_uses_detect_only_authoritative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    handlers = {"cms.fixture": lambda _acquire: {"status": "CURRENT"}}

    monkeypatch.setattr("release_check.production_handlers", lambda: handlers)

    def fake_check_releases(*, acquire: bool, external_handlers):
        calls.append({"acquire": acquire, "external_handlers": external_handlers})
        return {
            "checked_at": "2026-09-30T12:00:00+00:00",
            "datasets": [
                {
                    "mechanism": "external recurring release",
                    "status": "CURRENT",
                    "new_release_available": False,
                },
                {
                    "mechanism": "external recurring release",
                    "status": "DETECTED",
                    "new_release_available": True,
                },
                {
                    "mechanism": "external recurring release",
                    "status": "FAILED",
                    "new_release_available": None,
                },
                {
                    "mechanism": "derived from ACTIVE release",
                    "status": "MISSING",
                    "new_release_available": True,
                },
            ],
        }

    monkeypatch.setattr("release_check.check_releases", fake_check_releases)
    client = _client(monkeypatch)
    response = client.post("/actions/control-panel/check-releases", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/sources?check_cms=0")
    assert calls == [{"acquire": False, "external_handlers": handlers}]
    with client.session_transaction() as session:
        message = session["_flashes"][-1][1]
    assert "1 current · 1 newer · 1 errors" in message
    assert "1 non-CMS attention" in message


def test_refresh_active_file_health_does_not_check_cms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setenv("PBJ_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(
        app_module,
        "refresh_source_health",
        lambda: calls.append("source_health") or {},
    )
    monkeypatch.setattr(
        app_module,
        "refresh_facility_index",
        lambda *_args, **_kwargs: calls.append("facility_index") or {},
    )
    monkeypatch.setattr(
        "release_check.check_releases",
        lambda **_kwargs: pytest.fail("Refresh active-file health must not check CMS releases"),
    )

    client = _client(monkeypatch)
    response = client.post("/actions/control-panel/refresh", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/sources?check_cms=0")
    assert calls == ["source_health", "facility_index"]


def test_sources_labels_release_and_health_actions_distinctly() -> None:
    template = (
        Path(__file__).resolve().parents[1] / "templates" / "data_ops" / "sources.html"
    ).read_text(encoding="utf-8")
    assert "Check CMS releases" in template
    assert "Refresh active-file health" in template
    assert "Detect only" in template
    assert "Local files only" in template
    assert "Active-file health" in template
    assert "<th>Checked</th>" not in template.split("Source diagnostics", 1)[0]


def test_individual_acquire_route_calls_only_requested_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_acquire(source_id: str) -> dict[str, str]:
        calls.append(source_id)
        return {"release_id": "CY2026Q2", "status": "VALIDATED"}

    monkeypatch.setattr("release_check.acquire_detected_source", fake_acquire)
    client = _client(monkeypatch)
    response = client.post(
        "/actions/sources/cms.pbj_non_nurse_staffing/acquire",
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert calls == ["cms.pbj_non_nurse_staffing"]


def test_acquire_detected_source_uses_only_selected_production_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import release_check

    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    registry = state / "active_releases.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "cms.pbj_non_nurse_staffing": {
                        "active_release_id": "CY2026Q1",
                        "hash": "old-hash",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (state / "release_candidates.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "cms.pbj_non_nurse_staffing": {
                        "release_id": "CY2026Q2",
                        "state": "DETECTED",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []
    monkeypatch.setattr(release_check, "registry_path", lambda _root=None: registry)
    monkeypatch.setattr(
        release_check,
        "production_handlers",
        lambda: {
            "cms.pbj_non_nurse_staffing": lambda acquire: calls.append(
                f"nonnurse:{acquire}"
            )
            or {"status": "VALIDATED"},
            "cms.snf_all_owners": lambda acquire: calls.append(f"owners:{acquire}")
            or {"status": "ACQUIRED"},
        },
    )

    result = release_check.acquire_detected_source(
        "cms.pbj_non_nurse_staffing", root=tmp_path
    )

    assert result["status"] == "VALIDATED"
    assert calls == ["nonnurse:True"]


def test_individual_acquire_rejects_source_without_independent_handler(
    tmp_path: Path,
) -> None:
    import release_check

    with pytest.raises(ValueError, match="no independent production acquisition handler"):
        release_check.acquire_detected_source("cms.health_citations", root=tmp_path)


def test_detected_source_next_action_is_individual_acquire() -> None:
    import cms_data_ops as ops

    action = ops._next_operator_action(
        "cms.snf_all_owners",
        record={"actions_enabled": []},
        snapshot=None,
        control_row={
            "active": {"active_release_id": "2026-07-31"},
            "pending": {"release_id": "2026-08-31", "state": "DETECTED"},
        },
    )

    assert action["label"] == "Acquire 2026-08-31"
    assert action["endpoint"] == "action_source_acquire"
    assert action["endpoint_args"] == {"source_id": "cms.snf_all_owners"}
    assert action["method"] == "post"


def test_authoritative_release_check_persists_checked_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import release_check
    from release_source_catalog import ReleaseSource, UpdateMechanism

    source = ReleaseSource(
        "cms.fixture",
        "Fixture",
        UpdateMechanism.EXTERNAL_RECURRING,
    )
    monkeypatch.setattr(release_check, "SOURCES", (source,))
    result = release_check.check_releases(
        acquire=False,
        root=tmp_path,
        external_handlers={
            "cms.fixture": lambda acquire: {
                "status": "CURRENT",
                "new_release_available": False,
                "acquire_requested": acquire,
            }
        },
    )

    persisted = json.loads(
        (tmp_path / "state" / "release_checks.json").read_text(encoding="utf-8")
    )
    assert persisted["checked_at"] == result["checked_at"]
    assert persisted["datasets"][0]["status"] == "CURRENT"
    assert persisted["datasets"][0]["acquire_requested"] is False
