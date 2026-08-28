"""Focused Release Review verification (no activation)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_ops_app import create_app
from release_control_plane import control_panel_payload
import cms_data_ops as ops


def test_live_focused_health_citations_review_html(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(ROOT / "state" / "active_releases.json"))
    monkeypatch.setenv("PBJ_REPO_ROOT", str(ROOT.parent / "PBJapp"))
    monkeypatch.setenv("PBJ_DATA_OPS_PASSWORD", "test-password")
    control = control_panel_payload(ROOT)
    items = ops.release_review_items(
        check_cms=False,
        root=ROOT,
        control=control,
        focus_source_id="cms.health_citations",
        focus_release_id="2026-08",
    )
    if not items:
        pytest.skip("live state has no focused Health Citations VALIDATED review item")
    assert items[0]["approvable"] is True
    assert items[0]["primary_action_label"] == "Activate Aug 2026"

    app = create_app()
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["data_ops_authenticated"] = True
    resp = client.get(
        "/release-review",
        query_string={"source_id": "cms.health_citations", "release_id": "2026-08"},
    )
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "Ready to activate" in html
    assert "Activate Aug 2026" in html
    assert "NH_HealthCitations_Aug2026.csv" in html
    assert "54d6e3e7" in html or "Candidate SHA-256" in html
    assert "NH_HealthCitations_Jul2026.csv" not in html.split("Candidate artifact")[0] if "Candidate artifact" in html else True
    assert "governed_pending_validated" not in html
    assert "Zweli gates for Provider Info" not in html


def test_live_health_citations_source_detail_operator_screen(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(ROOT / "state" / "active_releases.json"))
    monkeypatch.setenv("PBJ_REPO_ROOT", str(ROOT.parent / "PBJapp"))
    monkeypatch.setenv("PBJ_DATA_OPS_PASSWORD", "test-password")

    app = create_app()
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["data_ops_authenticated"] = True
    resp = client.get("/sources/cms.health_citations")
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "probe failed" not in html.lower()
    assert "Up to date" in html or "NEEDS ATTENTION" in html
    assert "Aug 2026" in html
    assert "At a glance" in html
    assert "Canonical data" in html
    assert "NH_HealthCitations_Aug2026.csv" in html or "54d6e3e7" in html


def test_live_post_activation_ownership_downstream_and_sources_cleanup(monkeypatch: pytest.MonkeyPatch):
    """After human pair activation: SNF members quiet when current; downstream row only when stale."""
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(ROOT / "state" / "active_releases.json"))
    monkeypatch.setenv("PBJ_REPO_ROOT", str(ROOT.parent / "PBJapp"))
    monkeypatch.setenv("PBJ_DATA_OPS_PASSWORD", "test-password")

    from ownership_downstream_rebuild import audit_ownership_downstream_stale
    from ownership_pairing import pairing_status

    pair = pairing_status(ROOT)
    if pair.get("pending", {}).get("owners_release"):
        pytest.skip("live state still has pending ownership pair")
    active = (pair.get("active") or {}).get("owners_release")
    if active != "2026-07-31":
        pytest.skip(f"expected ACTIVE 2026-07-31 pair, got {active!r}")

    audit = audit_ownership_downstream_stale(root=ROOT, pbj_root=ROOT.parent / "PBJapp")

    app = create_app()
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["data_ops_authenticated"] = True

    sources = client.get("/sources")
    sources_html = sources.get_data(as_text=True)
    needs_block = sources_html.split("Active releases")[0]
    assert sources.status_code == 200
    assert "Activate pair" not in sources_html
    assert "SNF Owners / Enrollments" not in needs_block
    assert "SNF All Owners" not in needs_block or audit.get("is_stale")
    assert "SNF Enrollments" not in needs_block or audit.get("is_stale")
    if audit.get("is_stale"):
        assert "Ownership data" in needs_block
        assert "Rebuild downstream" in needs_block
        assert "/actions/ownership-downstream/rebuild" in needs_block
    else:
        assert "Ownership data" not in needs_block

    panel = client.get("/sources/cms.snf_all_owners/panel")
    html = panel.get_data(as_text=True)
    assert panel.status_code == 200
    if audit.get("is_stale"):
        assert "Rebuild downstream" in html
    else:
        assert "Up to date" in html or "Check CMS" in html
    assert "Release Review" not in html or audit.get("is_stale")


def test_live_ownership_pair_modal_and_sources_cleanup(monkeypatch: pytest.MonkeyPatch):
    """Pre-activation pair modal (skipped when pair already ACTIVE)."""
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(ROOT / "state" / "active_releases.json"))
    monkeypatch.setenv("PBJ_REPO_ROOT", str(ROOT.parent / "PBJapp"))
    monkeypatch.setenv("PBJ_DATA_OPS_PASSWORD", "test-password")

    from ownership_pairing import pairing_status

    pair = pairing_status(ROOT)
    if not pair.get("pending", {}).get("owners_release"):
        pytest.skip("live state has no pending ownership pair for activation modal")

    app = create_app()
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["data_ops_authenticated"] = True

    sources = client.get("/sources")
    sources_html = sources.get_data(as_text=True)
    assert sources.status_code == 200
    assert "Needs attention" in sources_html
    assert "Provider Information actions" not in sources_html
    assert "PBJ nurse staffing actions" not in sources_html
    assert "last_pi_action" not in sources_html

    panel = client.get("/sources/cms.snf_ownership_pair/panel")
    html = panel.get_data(as_text=True)
    assert panel.status_code == 200
    assert "Pair status" in html
    assert "Jul 31" in html
    assert "Activate pair" in html
    assert "Ready to activate" in html
    assert "SNF All Owners" in html
    assert "SNF Enrollments" in html
    assert "Zweli" not in html
    assert "Verified" in html or "2026-07-31" in html
    assert "Review pair" not in html
    assert "Release Review" not in html
    assert "/actions/ownership-pair/activate" in html
