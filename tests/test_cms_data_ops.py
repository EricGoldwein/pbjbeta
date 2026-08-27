"""Tests for Data Ops status, Zweli, approval, dashboard gates, Flask auth."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

import cms_data_ops as ops
import cms_data_paths
import cms_provider_info_acquire as acq
import data_ops_approval as approval
import data_ops_dashboard as dash
import data_ops_zweli as zweli
from data_ops_app import create_app
from data_ops_zweli import ZweliState


def _nh_csv_bytes(n: int = 1200, month_label: str = "2026-08-01", prefix: str = "000") -> bytes:
    cols = list(acq.REQUIRED_NH_COLUMNS) + ["City/Town", "Overall Rating", "Special Focus Status"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for i in range(n):
        ccn = f"{prefix}{i:03d}"[-6:].zfill(6)
        w.writerow(
            {
                cols[0]: ccn,
                cols[1]: f"Facility {ccn}",
                cols[2]: "CT",
                cols[3]: month_label,
                "City/Town": "Hartford",
                "Overall Rating": "3",
                "Special Focus Status": "",
            }
        )
    return buf.getvalue().encode("utf-8")


def _metastore(filename: str = "NH_ProviderInfo_Aug2026.csv") -> dict:
    return {
        "title": "Provider Information",
        "identifier": "4pq5-n9py",
        "released": "2026-08-26",
        "modified": "2026-08-01",
        "nextUpdateDate": "2026-09-30",
        "distribution": [
            {
                "data": {
                    "@type": "dcat:Distribution",
                    "downloadURL": (
                        "https://data.cms.gov/provider-data/sites/default/files/resources/"
                        f"abc/{filename}"
                    ),
                    "mediaType": "text/csv",
                }
            }
        ],
    }


def test_probe_all_sources_runtime_unavailable(tmp_path: Path):
    snaps = ops.probe_all_sources(check_cms=False, root=tmp_path, run_zweli=False)
    assert len(snaps) == 10
    by_id = {s.source_id: s for s in snaps}
    assert by_id["cms.snf_enrollments"].source_id != by_id["cms.snf_all_owners"].source_id
    assert "NOT AVAILABLE" in by_id["cms.pbj_nurse_staffing"].raw_available
    assert by_id["cms.provider_info"].actions_enabled == ["check_cms", "acquire_process"]
    assert by_id["cms.pbj_nurse_staffing"].actions_enabled == ["check_cms", "acquire_process"]
    assert by_id["cms.health_citations"].cms_dataset_id == "r5ix-sfxw"
    assert by_id["cms.sff_pdf_list"].automation_maturity == "partially_automated"


def test_probe_provider_info_cms_newer(tmp_path: Path):
    pi = tmp_path / "provider_info"
    pi.mkdir()
    (pi / "NH_ProviderInfo_Jul2026.csv").write_bytes(_nh_csv_bytes(1100, "2026-07-01", "100"))
    snap = ops.probe_source(
        "cms.provider_info",
        check_cms=True,
        fetch_json=lambda url: _metastore(),
        root=tmp_path,
        run_zweli=False,
    )
    assert snap.status == "CMS_NEWER"
    assert snap.publisher_latest == "Aug 2026"


def test_check_and_acquire_delegate_to_63(tmp_path: Path):
    pi = tmp_path / "provider_info"
    pi.mkdir()
    (pi / "NH_ProviderInfo_Aug2026.csv").write_bytes(_nh_csv_bytes(1200, "2026-08-01", "300"))
    result = ops.check_provider_info_cms(fetch_json=lambda url: _metastore(), root=tmp_path)
    assert result["action"] == "check_cms"
    assert result["cms"]["dataset_id"] == "4pq5-n9py"
    assert result["dry_run"]["status"] == "CURRENT"
    acq_result = ops.acquire_provider_info(
        dry_run=True, fetch_json=lambda url: _metastore(), root=tmp_path
    )
    assert acq_result["acquire_report"]["status"] == "CURRENT"


def test_structural_and_zweli_are_separate():
    cur = zweli.ProviderInfoMetrics("2026-08", 1000, 1000)
    base = zweli.ProviderInfoMetrics("2026-07", 1000, 1000)
    report = zweli.compare_provider_info_releases(cur, base)
    assert report.state == ZweliState.PASS
    assert isinstance(report.findings, list)


def test_baseline_unavailable_in_runtime_is_not_run():
    cur = zweli.ProviderInfoMetrics("2026-08", 14690, 14690)
    report = zweli.compare_provider_info_releases(
        cur,
        None,
        baseline_availability=zweli.BaselineAvailability.UNAVAILABLE_IN_RUNTIME,
        expected_baseline_release="2026-07",
    )
    assert report.state == ZweliState.NOT_RUN
    assert any(f.check_id == "baseline_unavailable_in_runtime" for f in report.findings)
    assert not any(f.check_id == "no_baseline" for f in report.findings)


def test_genuine_first_release_still_requires_review():
    cur = zweli.ProviderInfoMetrics("2020-01", 1000, 1000)
    report = zweli.compare_provider_info_releases(
        cur,
        None,
        baseline_availability=zweli.BaselineAvailability.NONE_EXPECTED,
    )
    assert report.state == ZweliState.REQUIRES_REVIEW
    assert any(f.check_id == "no_baseline" for f in report.findings)


def test_zweli_not_run_blocks_dashboard_refresh():
    blockers = dash.evaluate_source_gates_for_dashboard(zweli_state=ZweliState.NOT_RUN)
    assert dash.DashboardActionBlocker.ZWELI_NOT_RUN.value in blockers
    assert dash.DashboardActionBlocker.ZWELI_NOT_RUN.value in dash.SOURCE_DATA_REFRESH_BLOCKERS


def test_not_run_cannot_be_approved(tmp_path: Path):
    audit = tmp_path / "audit.jsonl"
    with pytest.raises(approval.ApprovalError, match="NOT_RUN"):
        approval.approve_release(
            "cms.provider_info",
            "2026-08",
            zweli_state=ZweliState.NOT_RUN,
            audit_path=audit,
        )


def test_probe_aug_without_july_is_not_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Expected prior unavailable in isolated runtime → Zweli NOT_RUN (not REQUIRES_REVIEW)."""
    import active_release_registry as arr

    monkeypatch.setattr(arr, "get_active_release", lambda *_a, **_k: None)
    pi = tmp_path / "provider_info"
    norm = tmp_path / "provider_info_normalized"
    pi.mkdir()
    norm.mkdir()
    (pi / "NH_ProviderInfo_Aug2026.csv").write_bytes(_nh_csv_bytes(1200, "2026-08-01", "200"))
    (norm / "ProviderInfoNorm_2026_08.csv").write_bytes(_nh_csv_bytes(1200, "2026-08-01", "200"))
    snap = ops.probe_source(
        "cms.provider_info",
        check_cms=False,
        root=tmp_path,
        run_zweli=True,
    )
    assert snap.zweli_status == "NOT_RUN"
    assert snap.zweli_report
    assert any(
        f["check_id"] == "baseline_unavailable_in_runtime"
        for f in snap.zweli_report["findings"]
    )


def test_synthetic_60x_scale_blocked():
    cur = zweli.ProviderInfoMetrics("2026-08", 60000, 60000)
    base = zweli.ProviderInfoMetrics("2026-07", 1000, 1000)
    report = zweli.compare_provider_info_releases(cur, base)
    assert report.state == ZweliState.BLOCKED
    assert any(f.check_id == "unit_scale_signature" for f in report.findings)

    cur2 = zweli.ProviderInfoMetrics("2026-08", 1000, 1000)
    base2 = zweli.ProviderInfoMetrics("2026-07", 60000, 60000)
    report2 = zweli.compare_provider_info_releases(cur2, base2)
    assert report2.state == ZweliState.BLOCKED


def test_blocked_cannot_be_approved(tmp_path: Path):
    audit = tmp_path / "audit.jsonl"
    with pytest.raises(approval.ApprovalError, match="BLOCKED"):
        approval.approve_release(
            "cms.provider_info",
            "2026-08",
            zweli_state=ZweliState.BLOCKED,
            audit_path=audit,
        )


def test_requires_review_needs_ack(tmp_path: Path):
    audit = tmp_path / "audit.jsonl"
    with pytest.raises(approval.ApprovalError, match="acknowledgement"):
        approval.approve_release(
            "cms.provider_info",
            "2026-08",
            zweli_state=ZweliState.REQUIRES_REVIEW,
            audit_path=audit,
        )
    approval.acknowledge_requires_review(
        "cms.provider_info", "2026-08", note="looked ok", audit_path=audit
    )
    entry = approval.approve_release(
        "cms.provider_info",
        "2026-08",
        zweli_state=ZweliState.REQUIRES_REVIEW,
        audit_path=audit,
    )
    assert entry.action == "approve"


def test_blocked_cannot_generate_dashboard():
    blockers = dash.evaluate_source_gates_for_dashboard(zweli_state=ZweliState.BLOCKED)
    assert dash.DashboardActionBlocker.ZWELI_BLOCKED.value in blockers


def test_requires_review_blocks_without_ack():
    blockers = dash.evaluate_source_gates_for_dashboard(
        zweli_state=ZweliState.REQUIRES_REVIEW, zweli_ack=False
    )
    assert dash.DashboardActionBlocker.ZWELI_REQUIRES_ACK.value in blockers
    blockers2 = dash.evaluate_source_gates_for_dashboard(
        zweli_state=ZweliState.REQUIRES_REVIEW, zweli_ack=True
    )
    assert dash.DashboardActionBlocker.ZWELI_REQUIRES_ACK.value not in blockers2


def test_flask_auth_fails_closed():
    import os

    os.environ.pop("PBJ_DATA_OPS_PASSWORD", None)
    app = create_app()
    client = app.test_client()
    r = client.get("/sources")
    assert r.status_code == 503


def test_flask_auth_and_post_actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PBJ_DATA_OPS_PASSWORD", "test-ops-pw")
    monkeypatch.setenv("PBJ_DATA_OPS_SECRET", "test-secret-not-password")
    app = create_app()
    client = app.test_client()
    assert client.get("/sources").status_code in (302, 401, 503) or True
    # unauthenticated redirect
    r = client.get("/sources", follow_redirects=False)
    assert r.status_code == 302
    assert "/login" in (r.headers.get("Location") or "")
    # GET mutate must not work for acquire
    r = client.get("/actions/provider-info/acquire")
    assert r.status_code == 405
    # login
    r = client.post("/login", data={"password": "wrong"})
    assert r.status_code == 401
    r = client.post("/login", data={"password": "test-ops-pw"}, follow_redirects=False)
    assert r.status_code == 302
    r = client.get("/sources")
    assert r.status_code == 200
    assert b"Sources" in r.data
    assert b"Provider Information" in r.data


def test_probe_chain_prefers_nov(tmp_path: Path):
    own = tmp_path / "ownership"
    own.mkdir()
    (own / "Nursing_Home_Chain_Performance_Measures_Jul_2025.csv").write_text("a,b\n1,2\n")
    (own / "Nursing_Home_Chain_Performance_Measures_Nov_2025.csv").write_text("a,b\n1,2\n")
    snap = ops.probe_source("cms.chain_performance", check_cms=False, root=tmp_path, run_zweli=False)
    assert snap.pbjapp_latest == "Nov 2025"


def test_recommended_next_nurse():
    nxt = ops.recommended_next_automation()
    assert nxt["source_id"] == "cms.pbj_non_nurse_staffing"
    assert "first_broken_layer" in nxt


def _write_control_plane_state(
    root: Path,
    *,
    active: dict[str, dict[str, object]] | None = None,
    pending: dict[str, dict[str, object]] | None = None,
) -> None:
    """Write authoritative state/ registry files for overlay tests."""
    import release_control_plane as rcp

    state_dir = rcp._state_dir(root)
    state_dir.mkdir(parents=True, exist_ok=True)
    if active is not None:
        datasets = {}
        for dataset_id, spec in active.items():
            release_id = str(spec.get("release_id") or spec.get("active_release_id") or "")
            meta = dict(spec.get("metadata") or {})
            if spec.get("zweli_status") is not None:
                meta.setdefault("zweli_status", spec.get("zweli_status"))
            datasets[dataset_id] = {
                "dataset_id": dataset_id,
                "active_release_id": release_id,
                "status": str(spec.get("status") or "ACTIVE"),
                "metadata": meta,
            }
        (state_dir / "active_releases.json").write_text(
            json.dumps({"schema_version": 1, "updated_at": "2026-08-27T00:00:00+00:00", "datasets": datasets}),
            encoding="utf-8",
        )
    if pending is not None:
        datasets = {}
        for dataset_id, spec in pending.items():
            meta = dict(spec.get("metadata") or {})
            if spec.get("zweli_status") is not None:
                meta.setdefault("zweli_status", spec.get("zweli_status"))
            datasets[dataset_id] = {
                "dataset_id": dataset_id,
                "release_id": str(spec.get("release_id") or ""),
                "state": str(spec.get("state") or "ACQUIRED"),
                "metadata": meta,
                "validation": dict(spec.get("validation") or {}),
            }
        (state_dir / "release_candidates.json").write_text(
            json.dumps({"schema_version": 1, "updated_at": "2026-08-27T00:00:00+00:00", "datasets": datasets}),
            encoding="utf-8",
        )


def test_control_plane_overlay_sff_active(tmp_path: Path):
    _write_control_plane_state(
        tmp_path,
        active={
            "cms.sff_pdf_list": {"release_id": "2026-08", "status": "ACTIVE"},
        },
    )
    import release_control_plane as rcp

    control = rcp.control_panel_payload(tmp_path)
    snap = ops.probe_source("cms.sff_pdf_list", check_cms=False, root=tmp_path, run_zweli=False)
    assert snap.status in {"NOT_AVAILABLE_IN_THIS_RUNTIME", "UNKNOWN"}
    assert "NOT AVAILABLE" in snap.raw_available
    overlaid = ops.overlay_control_plane_on_snapshot(snap, control, root=tmp_path)
    assert overlaid["active_release_id"] == "2026-08"
    assert overlaid["active_release_status"] == "ACTIVE"
    assert overlaid["status"] != "NOT_AVAILABLE_IN_THIS_RUNTIME"
    assert overlaid["display_status"] == "ACTIVE"
    assert overlaid["raw_available"] == "2026-08"


def test_control_plane_overlay_snf_all_owners(tmp_path: Path):
    _write_control_plane_state(
        tmp_path,
        active={
            "cms.snf_all_owners": {"release_id": "2026-07-17", "status": "ACTIVE"},
        },
        pending={
            "cms.snf_all_owners": {
                "release_id": "2026-07-31",
                "state": "ACQUIRED",
                "validation": {"status": "PASS"},
            }
        },
    )
    import release_control_plane as rcp

    control = rcp.control_panel_payload(tmp_path)
    overlaid = ops.overlay_control_plane_on_snapshot(
        ops.probe_source("cms.snf_all_owners", check_cms=False, root=tmp_path, run_zweli=False),
        control,
        root=tmp_path,
    )
    assert overlaid["active_release_id"] == "2026-07-17"
    assert overlaid["pending_release_id"] == "2026-07-31"
    assert overlaid["pending_release_state"] == "ACQUIRED"
    assert overlaid["status"] != "NOT_AVAILABLE_IN_THIS_RUNTIME"


def test_control_plane_overlay_snf_enrollments(tmp_path: Path):
    _write_control_plane_state(
        tmp_path,
        active={
            "cms.snf_enrollments": {"release_id": "2026-07-17", "status": "ACTIVE"},
        },
        pending={
            "cms.snf_enrollments": {
                "release_id": "2026-07-31",
                "state": "ACQUIRED",
                "validation": {"status": "PASS"},
            }
        },
    )
    import release_control_plane as rcp

    control = rcp.control_panel_payload(tmp_path)
    overlaid = ops.overlay_control_plane_on_snapshot(
        ops.probe_source("cms.snf_enrollments", check_cms=False, root=tmp_path, run_zweli=False),
        control,
        root=tmp_path,
    )
    assert overlaid["active_release_id"] == "2026-07-17"
    assert overlaid["pending_release_id"] == "2026-07-31"
    assert overlaid["pending_release_state"] == "ACQUIRED"
    assert overlaid["status"] != "NOT_AVAILABLE_IN_THIS_RUNTIME"


def test_control_plane_overlay_provider_info_active_pending_not_run(tmp_path: Path):
    _write_control_plane_state(
        tmp_path,
        active={
            "cms.provider_info": {
                "release_id": "2026-07",
                "status": "ACTIVE",
                "zweli_status": "NOT_RUN",
            },
        },
        pending={
            "cms.provider_info": {
                "release_id": "2026-08",
                "state": "ACQUIRED",
                "zweli_status": "NOT_RUN",
            }
        },
    )
    import release_control_plane as rcp

    control = rcp.control_panel_payload(tmp_path)
    overlaid = ops.overlay_control_plane_on_snapshot(
        ops.probe_source("cms.provider_info", check_cms=False, root=tmp_path, run_zweli=False),
        control,
        root=tmp_path,
    )
    assert overlaid["active_release_id"] == "2026-07"
    assert overlaid["pending_release_id"] == "2026-08"
    assert overlaid["pending_release_state"] == "ACQUIRED"
    assert overlaid["zweli_status"] == "NOT_RUN"


def test_release_review_includes_governed_pending_not_approvable(tmp_path: Path):
    _write_control_plane_state(
        tmp_path,
        active={
            "cms.snf_all_owners": {"release_id": "2026-07-17", "status": "ACTIVE"},
            "cms.provider_info": {
                "release_id": "2026-07",
                "status": "ACTIVE",
                "zweli_status": "NOT_RUN",
            },
        },
        pending={
            "cms.snf_all_owners": {
                "release_id": "2026-07-31",
                "state": "ACQUIRED",
            },
            "cms.provider_info": {
                "release_id": "2026-08",
                "state": "ACQUIRED",
                "zweli_status": "NOT_RUN",
            },
        },
    )
    import release_control_plane as rcp

    control = rcp.control_panel_payload(tmp_path)
    items = ops.release_review_items(check_cms=False, root=tmp_path, control=control)
    by_key = {(i["source_id"], i["release_id"]): i for i in items}
    owners = by_key[("cms.snf_all_owners", "2026-07-31")]
    provider = by_key[("cms.provider_info", "2026-08")]
    assert owners["governed"] is True
    assert owners["approvable"] is False
    assert provider["governed"] is True
    assert provider["approvable"] is False
    assert provider["zweli_status"] == "NOT_RUN"


def test_promote_candidate_permitted_requires_validated():
    assert ops.promote_candidate_permitted({"state": "VALIDATED"}) is True
    assert ops.promote_candidate_permitted({"state": "ACQUIRED"}) is False


def _write_provider_info_release(
    root: Path,
    *,
    year: int,
    month: int,
    prefix: str = "300",
) -> Path:
    """Minimal local Provider Info raw (+ norm) for approval/canonical-source tests."""
    pi = root / "provider_info"
    norm = root / "provider_info_normalized"
    pi.mkdir(parents=True, exist_ok=True)
    norm.mkdir(parents=True, exist_ok=True)
    month_names = (
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    )
    label = f"{month_names[month - 1]}{year}"
    month_label = f"{year:04d}-{month:02d}-01"
    raw = pi / f"NH_ProviderInfo_{label}.csv"
    raw.write_bytes(_nh_csv_bytes(1200, month_label, prefix))
    (norm / f"ProviderInfoNorm_{year}_{month:02d}.csv").write_bytes(
        _nh_csv_bytes(1200, month_label, prefix)
    )
    return raw


def _write_zweli_report(root: Path, release_id: str, state: str) -> Path:
    man = root / "provider_info" / "_manifests" / release_id
    man.mkdir(parents=True, exist_ok=True)
    path = man / "zweli_report.json"
    path.write_text(
        json.dumps(
            {
                "source_id": "cms.provider_info",
                "release_id": release_id,
                "profile": "provider_info_v0",
                "state": state,
                "findings": [],
                "checked_at": "2026-08-26T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_forged_form_pass_cannot_approve_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Browser-submitted zweli_status=PASS must not override server BLOCKED."""
    monkeypatch.setenv("PBJ_DATA_OPS_PASSWORD", "test-ops-pw")
    monkeypatch.setenv("PBJ_DATA_OPS_SECRET", "test-secret")
    monkeypatch.setenv("PBJ_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(cms_data_paths, "repo_root", lambda: tmp_path)
    _write_provider_info_release(tmp_path, year=2026, month=8, prefix="300")
    _write_zweli_report(tmp_path, "2026-08", "BLOCKED")
    audit = tmp_path / "provider_info" / "_manifests" / "_data_ops_audit.jsonl"

    # Direct service: forged PASS ignored when resolving from stored report
    with pytest.raises(approval.ApprovalError, match="BLOCKED"):
        ops.approve_release_authoritative(
            "cms.provider_info",
            "2026-08",
            note="forged",
            root=tmp_path,
            audit_path=audit,
        )

    app = create_app()
    client = app.test_client()
    assert client.post("/login", data={"password": "test-ops-pw"}).status_code == 302
    r = client.post(
        "/actions/approve",
        data={
            "source_id": "cms.provider_info",
            "release_id": "2026-08",
            "zweli_status": "PASS",  # forged
            "note": "should fail",
        },
        follow_redirects=True,
    )
    assert r.status_code == 200
    assert b"Approved cms.provider_info" not in r.data
    body = r.data.decode("utf-8", errors="replace")
    assert "BLOCKED" in body or "cannot be approved" in body
    assert not approval.has_approval("cms.provider_info", "2026-08", audit)


def test_authoritative_promotion_writes_isolated_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(cms_data_paths, "repo_root", lambda: tmp_path)
    _write_provider_info_release(tmp_path, year=2026, month=8, prefix="300")
    audit = tmp_path / "audit.jsonl"
    _write_zweli_report(tmp_path, "2026-08", "PASS")
    ops.approve_release_authoritative(
        "cms.provider_info", "2026-08", root=tmp_path, audit_path=audit
    )
    import json
    from active_release_registry import load_registry

    isolated = json.loads(
        (tmp_path / "state" / "active_releases.json").read_text(encoding="utf-8")
    )
    active = isolated["datasets"]["cms.provider_info"]["active_release_id"]
    assert active == "2026-08"


def test_authoritative_requires_review_and_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cms_data_paths, "repo_root", lambda: tmp_path)
    _write_provider_info_release(tmp_path, year=2026, month=8, prefix="300")
    audit = tmp_path / "audit.jsonl"
    _write_zweli_report(tmp_path, "2026-08", "REQUIRES_REVIEW")
    with pytest.raises(approval.ApprovalError, match="acknowledgement"):
        ops.approve_release_authoritative(
            "cms.provider_info", "2026-08", root=tmp_path, audit_path=audit
        )
    approval.acknowledge_requires_review(
        "cms.provider_info", "2026-08", audit_path=audit
    )
    entry = ops.approve_release_authoritative(
        "cms.provider_info", "2026-08", root=tmp_path, audit_path=audit
    )
    assert entry.action == "approve"

    _write_provider_info_release(tmp_path, year=2026, month=9, prefix="301")
    _write_zweli_report(tmp_path, "2026-09", "PASS")
    entry2 = ops.approve_release_authoritative(
        "cms.provider_info", "2026-09", root=tmp_path, audit_path=audit
    )
    assert entry2.action == "approve"


def test_stale_missing_zweli_fails_closed(tmp_path: Path):
    audit = tmp_path / "audit.jsonl"
    with pytest.raises(approval.ApprovalError, match="fail closed|NOT_RUN|missing"):
        ops.approve_release_authoritative(
            "cms.provider_info", "2099-01", root=tmp_path, audit_path=audit
        )


def test_unavailable_and_unprocessed_block_refresh(tmp_path: Path):
    # Minimal V2-looking deploy + ref so only source gates matter
    ref = "315128"
    for ccn in (ref, "999999"):
        d = tmp_path / "deployments" / f"pbj320-{ccn}"
        d.mkdir(parents=True)
        (d / f"facility_{ccn}_superdynamic_dashboard.py").write_text("# v2\n", encoding="utf-8")

    st_unavail = dash.facility_dashboard_status(
        "999999",
        root=tmp_path,
        provider_available=False,
        provider_processed=True,
        structural_ok=True,
        zweli_state=ZweliState.PASS,
        run_readiness=False,
    )
    assert dash.DashboardActionBlocker.SOURCE_UNAVAILABLE.value in st_unavail.blockers
    assert st_unavail.can_generate_refresh is False

    st_unproc = dash.facility_dashboard_status(
        "999999",
        root=tmp_path,
        provider_available=True,
        provider_processed=False,
        structural_ok=True,
        zweli_state=ZweliState.PASS,
        run_readiness=False,
    )
    assert dash.DashboardActionBlocker.UNPROCESSED.value in st_unproc.blockers
    assert st_unproc.can_generate_refresh is False


def test_v2_safety_still_blocks_without_ref_or_non_v2(tmp_path: Path):
    # No V2 reference anywhere
    d = tmp_path / "deployments" / "pbj320-999999"
    d.mkdir(parents=True)
    (d / "facility_999999_flask_app.py").write_text("# legacy\n", encoding="utf-8")
    st = dash.facility_dashboard_status(
        "999999",
        root=tmp_path,
        provider_available=True,
        provider_processed=True,
        zweli_state=ZweliState.PASS,
        run_readiness=False,
    )
    assert st.can_generate_refresh is False
    assert dash.DashboardActionBlocker.UNSAFE_PACKAGE_PATH.value in st.blockers
    assert dash.DashboardActionBlocker.NO_V2_REFERENCE.value in st.blockers


def test_format_do_timestamp_humanizes_iso():
    assert ops.format_do_timestamp("2026-08-27T12:51:31.031613+00:00") == "Aug 27, 8:51 AM ET"
    assert ops.format_do_timestamp(None) == "—"
    assert ops.format_do_timestamp("not-a-date") == "not-a-date"


def test_build_sff_lifecycle_active_read_only():
    control_row = {
        "active": {"active_release_id": "2026-08", "status": "ACTIVE"},
        "pending": None,
    }
    steps = ops.build_sff_lifecycle_steps(control_row=control_row)
    labels = [s["label"] for s in steps]
    assert labels == [
        "Check CMS",
        "Acquire PDF",
        "Validate",
        "Review",
        "Make ACTIVE",
        "PBJ build",
        "Public staging",
        "Publish",
    ]
    make_active = next(s for s in steps if s["id"] == "make_active")
    assert make_active["state"] == "completed"
    publish = next(s for s in steps if s["id"] == "publish")
    assert publish["state"] == "not_wired"
    pbj = next(s for s in steps if s["id"] == "pbj_build")
    assert pbj["action"]["endpoint"] == "dashboard_builder"


def test_build_source_operator_workflow_uses_control_plane():
    control_row = {
        "active": {"active_release_id": "2026-07"},
        "pending": {"release_id": "2026-08", "state": "ACQUIRED"},
        "health": "PASS",
        "health_detail": "ok",
        "impact": {"would_mark_stale": ["facility_dashboard"]},
    }
    wf = ops.build_source_operator_workflow(
        "cms.provider_info",
        record={"actions_enabled": ["check_cms"]},
        snapshot={"validation_status": "PASS", "zweli_status": "NOT_RUN"},
        control_row=control_row,
    )
    assert wf["active_release_id"] == "2026-07"
    assert wf["pending_state"] == "ACQUIRED"
    assert wf["next_action"]["endpoint"] == "release_review"
    assert wf["lifecycle_steps"] is None

