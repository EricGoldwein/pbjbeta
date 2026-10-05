from data_ops_dashboard_jobs import parse_started_phases, progress_from_log


def test_parse_started_phases_in_order():
    log = """
--- Phase: scaffold_config ---
ok
--- Phase: extract ---
working
--- Phase: extract ---
"""
    assert parse_started_phases(log) == ["scaffold_config", "extract"]


def test_percent_only_counts_finished_steps():
    log = """
--- Phase: scaffold_config ---
--- Phase: cms_data_ready ---
--- Phase: pre_extract ---
--- Phase: extract ---
"""
    prog = progress_from_log(log, "build")
    states = {s["id"]: s["state"] for s in prog["steps"]}
    assert states["ready"] == "done"
    assert states["staffing"] == "active"
    assert states["package"] == "pending"
    assert prog["percent"] == 10
    assert "Staffing extract" in prog["label"]


def test_finished_ok_is_100():
    log = """
--- Phase: scaffold_config ---
--- Phase: extract ---
--- Phase: package ---
--- Phase: preflight ---
"""
    prog = progress_from_log(log, "build", finished=True, ok=True)
    assert prog["percent"] == 100
    assert all(s["state"] == "done" for s in prog["steps"])
    assert "publish" not in {s["id"] for s in prog["steps"]}


def test_local_plan_has_no_publish_step():
    prog = progress_from_log("", "build")
    assert [s["id"] for s in prog["steps"]] == [
        "ready",
        "staffing",
        "identity",
        "ownership",
        "package",
        "checks",
    ]


def test_deploy_plan_is_package_checks_publish():
    prog = progress_from_log("", "deploy_production")
    assert [s["id"] for s in prog["steps"]] == ["package", "checks", "publish"]
    assert prog["steps"][0]["label"] == "Confirm local dashboard"
    assert prog["steps"][2]["label"] == "Ship to Vercel"


def test_package_phase_with_nonnurse_counts_as_staffing():
    log = """
--- Phase: package ---
$ python scripts/deploy_vercel_facility.py 365865 --package
  Running create_vercel_deployment.py (smart_refresh; only stale slices rebuild)…
[nonnurse] process CY2018Q1: new quarter processed (4/37)
"""
    prog = progress_from_log(log, "build")
    states = {s["id"]: s["state"] for s in prog["steps"]}
    assert states["staffing"] == "active"
    assert states["package"] == "pending"
    assert states["ready"] == "done"
    assert "Staffing extract" in prog["label"]
    assert "4/37" in prog["label"]
    assert prog["percent"] > 10
    assert prog["percent"] < 70


def test_enrich_running_job_uses_log_not_stale_percent(tmp_path, monkeypatch):
    from data_ops_dashboard_jobs import enrich_running_job_progress, jobs_dir

    monkeypatch.setattr("data_ops_dashboard_jobs.JOBS_DIR", tmp_path)
    job_id = "abc123"
    (tmp_path / f"{job_id}.log").write_text(
        "--- Phase: package ---\n[nonnurse] process CY2018Q1: new quarter processed (4/37)\n",
        encoding="utf-8",
    )
    job = {
        "id": job_id,
        "intent": "build",
        "state": "running",
        "percent": 0,
        "label": "Package bundle",
    }
    out = enrich_running_job_progress(job)
    assert "Staffing extract" in out["label"]
    assert out["percent"] > 0


def test_reused_slices_are_done_during_package():
    log = """
--- Phase: package ---
=== Smart data refresh assessment ===
  nurse_pbj            reuse   validated against cms.pbj_nurse_staffing CY2026Q1
  nonnurse_pbj         reuse   validated against cms.pbj_non_nurse_staffing CY2026Q1
  ownership            REFRESH  artifact missing
  [OK] facility_365865_complete_data.csv: validated against cms.pbj_nurse_staffing CY2026Q1
  [OK] facility_365865_nonnurse_daily.csv: validated against cms.pbj_non_nurse_staffing CY2026Q1
Phase 3/5: Roster snapshot reuse-or-fail
[SNAPSHOT-CORE] 1387 / 2100 dates | elapsed 60.0s
"""
    prog = progress_from_log(log, "build")
    states = {s["id"]: s["state"] for s in prog["steps"]}
    notes = {s["id"]: s["note"] for s in prog["steps"]}
    assert states["staffing"] == "done"
    assert notes["staffing"] == "Reused on disk"
    assert states["package"] == "active"
    assert "1387/2100" in notes["package"]
    assert prog["percent"] >= 70
    assert prog["percent"] < 100


def test_package_phase_uses_short_title():
    log = """
--- Phase: package ---
=== Smart data refresh assessment ===
  nurse_pbj            reuse   ok
  nonnurse_pbj         reuse   ok
  [OK] facility_365865_complete_data.csv: validated
  [OK] facility_365865_nonnurse_daily.csv: validated
Phase 2/5: Incremental code sync
"""
    prog = progress_from_log(log, "build")
    notes = {s["id"]: s["note"] for s in prog["steps"]}
    assert "Copying UI" in notes["package"]
    assert "2/5" in notes["package"]
    assert "Phase 2/5" not in notes["package"]


def test_preflight_fail_sets_checks_reason():
    log = """
--- Phase: package ---
  [OK] facility_365865_complete_data.csv: validated against cms.pbj_nurse_staffing CY2026Q1
  [OK] facility_365865_nonnurse_daily.csv: validated against cms.pbj_non_nurse_staffing CY2026Q1
FAIL Citations ACTIVE-release provenance
     artifact changed since provenance was recorded
PACKAGE PREFLIGHT FAIL: 1 blocking issue(s)
preflight              failed       20.6  exit 1
"""
    prog = progress_from_log(log, "build", finished=True, ok=False)
    states = {s["id"]: s["state"] for s in prog["steps"]}
    notes = {s["id"]: s["note"] for s in prog["steps"]}
    assert states["package"] == "done"
    assert states["checks"] == "failed"
    assert "Citations" in notes["checks"]
    assert notes["checks"] != "Failed"


def test_local_job_allows_uncommitted_dashboard_code(monkeypatch, tmp_path):
    captured = {}
    script = tmp_path / "provision_premium_facility.py"
    script.write_text("# stub\n", encoding="utf-8")

    def fake_argv(**kwargs):
        captured.update(kwargs)
        return ["python", "-c", "pass"]

    class FakeThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            return None

    monkeypatch.setattr("data_ops_dashboard.provision_script_path", lambda root=None: script)
    monkeypatch.setattr("data_ops_dashboard_jobs.build_provision_argv", fake_argv)
    monkeypatch.setattr("data_ops_dashboard_jobs._infer_cold_and_scaffold", lambda ccn: (False, False))
    monkeypatch.setattr("data_ops_dashboard_jobs._write_job", lambda payload: None)
    monkeypatch.setattr("data_ops_dashboard_jobs.threading.Thread", FakeThread)
    from data_ops_dashboard_jobs import start_dashboard_job

    local = start_dashboard_job(
        ccn="365865",
        intent="build",
        access_mode="password_required",
        password="secret",
    )
    assert local["ok"] is True
    assert captured["allow_dirty_tree"] is True
    assert captured["allow_dirty_critical_files"] is True

    captured.clear()
    publish = start_dashboard_job(
        ccn="365865",
        intent="deploy_staging",
        access_mode="password_required",
        password="secret",
        confirm_ccn="365865",
        confirm_publish=True,
    )
    assert publish["ok"] is True
    assert captured["allow_dirty_tree"] is True
    assert captured["allow_dirty_critical_files"] is True
