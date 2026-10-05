"""Dashboard Builder operator actions (look up / build / deploy)."""

from __future__ import annotations

from pathlib import Path

import data_ops_dashboard as dash


def test_validate_build_requires_password():
    errors = dash.validate_dashboard_run(
        ccn="365865",
        intent="build",
        access_mode="password_required",
        password="",
        confirm_ccn="",
        confirm_publish=False,
    )
    assert dash.DashboardActionBlocker.PASSWORD_REQUIRED.value in errors


def test_validate_deploy_requires_typed_ccn_and_checkbox():
    errors = dash.validate_dashboard_run(
        ccn="365865",
        intent="deploy_production",
        access_mode="password_required",
        password="Connolly_320",
        confirm_ccn="",
        confirm_publish=False,
    )
    assert dash.DashboardActionBlocker.DEPLOY_CONFIRM_REQUIRED.value in errors

    ok = dash.validate_dashboard_run(
        ccn="365865",
        intent="deploy_production",
        access_mode="password_required",
        password="secret",
        confirm_ccn="365865",
        confirm_publish=True,
    )
    assert dash.DashboardActionBlocker.DEPLOY_CONFIRM_REQUIRED.value not in ok
    assert dash.DashboardActionBlocker.PASSWORD_REQUIRED.value not in ok


def test_build_argv_never_includes_production_or_password():
    argv = dash.build_provision_argv(
        ccn="365865",
        intent="build",
        access_mode="password_required",
        cold=True,
        scaffold_config=True,
        refresh_data=False,
        allow_dirty_tree=True,
        allow_dirty_critical_files=True,
    )
    assert "--production" not in argv
    assert "--staging" not in argv
    assert "--cold" in argv
    assert "--scaffold-config" in argv
    assert "--allow-dirty-tree" in argv
    assert "--allow-dirty-critical-files" in argv
    joined = " ".join(argv)
    assert "Connolly" not in joined
    assert "--password" not in argv


def test_production_argv_includes_acceptance_not_deploy_only():
    argv = dash.build_provision_argv(
        ccn="365865",
        intent="deploy_production",
        access_mode="password_required",
        cold=True,
        scaffold_config=True,
        refresh_data=False,
        allow_dirty_tree=False,
        allow_dirty_critical_files=True,
    )
    assert "--production" in argv
    assert "--run-api-acceptance" in argv
    assert "--run-browser-acceptance" in argv
    assert "--deploy-only" not in argv
    assert "--staging" not in argv
    assert "--allow-dirty-critical-files" in argv
    assert "--allow-dirty-tree" not in argv


def test_preview_is_dry_run():
    argv = dash.build_provision_argv(
        ccn="365865",
        intent="preview",
        access_mode="password_required",
        cold=True,
        scaffold_config=True,
        refresh_data=True,
        allow_dirty_tree=True,
    )
    assert "--dry-run" in argv
    assert "--refresh-data" in argv
    assert "--production" not in argv


def test_run_dashboard_provision_redacts_password_and_skips_deploy_flags(monkeypatch, tmp_path):
    script = tmp_path / "scripts" / "provision_premium_facility.py"
    script.parent.mkdir()
    script.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(dash, "provision_script_path", lambda root=None: script)
    monkeypatch.setattr(dash, "_infer_cold_and_scaffold", lambda ccn, root=None: (True, True))

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = "using password Connolly_320 for env"
        stderr = ""

    def runner(cmd, env, cwd):
        captured["cmd"] = cmd
        captured["env_has"] = env.get("PBJ_DASHBOARD_PASSWORD")
        return FakeProc()

    result = dash.run_dashboard_provision(
        ccn="365865",
        intent="build",
        access_mode="password_required",
        password="Connolly_320",
        runner=runner,
    )
    assert result["ok"] is True
    assert result["deployed"] is False
    assert "Connolly_320" not in result["stdout"]
    assert "<redacted>" in result["stdout"]
    assert captured["env_has"] == "Connolly_320"
    assert "--production" not in captured["cmd"]
    assert "dashboard_password" not in result
    assert "password" not in result


def test_sanitize_action_payload_strips_secrets():
    clean = dash.sanitize_action_payload(
        {"action": "run", "dashboard_password": "x", "result": {"password": "y", "ok": True}}
    )
    assert "dashboard_password" not in clean
    assert "password" not in clean["result"]
    assert clean["result"]["ok"] is True


def test_builder_template_uses_look_up_not_resolve():
    html = (Path(__file__).resolve().parents[1] / "templates" / "data_ops" / "dashboard_builder.html").read_text(
        encoding="utf-8"
    )
    assert "Look up" in html
    assert "Resolve" not in html
    assert "primary_build_label" in html
    assert "do-builder-file-notes" in html
    assert "Diagnostics" in html
    assert "Leave unchecked" not in html
    assert "Publish" in html
    assert "dashboard_password" in html
    assert "confirm_publish" in html
    assert 'name="intent" value="build"' in html
    assert "No Deploy in V0" not in html
    assert "Source gates" not in html
    assert "ZWELI_NOT_RUN" not in html
    assert "create_vercel_deployment" not in html
    assert html.count("Open local dashboard") == 1
    assert "data-do-open-local" in html
    assert "1. Build locally" in html
    assert "2. Publish to Vercel" in html
    assert "Also rebuild PBJ nurse hours from CMS" in html
    assert "allow_dirty_critical_files" not in html


def test_busy_js_preserves_submitter_before_disable():
    js = (Path(__file__).resolve().parents[1] / "static" / "data_ops" / "data_ops.js").read_text(encoding="utf-8")
    assert "data-do-submitter-keep" in js
    assert "keep.name = submitted.name" in js
    assert "Publish failed" in js
    assert 'saved.state === "error"' in js


def test_builder_view_model_uses_plain_language():
    view = dash.builder_view_model(
        {
            "ccn": "365865",
            "facility_name": None,
            "bundle_state": "NOT_GENERATED",
            "is_v2": False,
            "detail": "No local bundle yet.",
            "blockers": ["ZWELI_NOT_RUN", "MISSING_DEPLOY_DIR"],
            "can_build_local": True,
            "can_deploy": True,
            "can_run_preflight": False,
        }
    )
    assert view["state_label"] == "Not built yet"
    assert view["headline"] == "This facility"
    assert view["local_available"] is False
    assert view["primary_build_label"] == "Build local dashboard"
    assert "MISSING_DEPLOY_DIR" not in view["notes"]
    assert "ZWELI_NOT_RUN" not in " ".join(view["notes"])
    assert not any("Provider Information review has not been run" in n for n in view["notes"])
    assert view.get("local_view_url") is None
    assert view["publish_block_reason"]
    assert view["can_publish"] is False


def test_local_view_port_is_last_four():
    assert dash.local_view_port("365865") == 5865
    assert dash.local_view_url("365865") == "http://127.0.0.1:5865/"


def test_v2_builder_view_includes_local_open_url(monkeypatch):
    monkeypatch.setattr(dash, "local_view_listening", lambda ccn, timeout_s=0.4: False)
    view = dash.builder_view_model(
        {
            "ccn": "365865",
            "facility_name": "Test Home",
            "bundle_state": "GENERATED_LOCALLY",
            "is_v2": True,
            "detail": "A local V2 dashboard is already on disk.",
            "blockers": [],
            "can_build_local": True,
            "can_deploy": True,
            "can_run_preflight": True,
        }
    )
    assert view["local_view_url"] == "http://127.0.0.1:5865/"
    assert view["local_view_port"] == 5865
    assert view["prove_command"] == "python scripts/check_provider_quarter_flow.py --prove --ccn 365865"
    assert "--ccn 365865" in view["status_command"]
    assert view["local_available"] is True
    assert view["state_label"] == "Available locally"


def test_open_local_restart_does_not_reuse_listener(monkeypatch, tmp_path):
    seq = [False, True]

    def listening(ccn, timeout_s=0.4):
        if seq:
            return seq.pop(0)
        return True

    monkeypatch.setattr(dash, "local_view_listening", listening)
    monkeypatch.setattr(dash, "_detect_v2", lambda deploy_dir, ccn: True)
    monkeypatch.setattr(dash.cms_data_paths, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(dash.cms_data_paths, "facility_deploy_dir", lambda ccn, root=None: tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "start_local_v3_facility.py").write_text("# stub\n", encoding="utf-8")
    popped = {}

    def fake_popen(cmd, **kwargs):
        popped["cmd"] = cmd
        popped["mode"] = kwargs.get("env", {}).get("PBJ_QA_MODE")

        class Proc:
            pass

        return Proc()

    monkeypatch.setattr(dash.subprocess, "Popen", fake_popen)
    result = dash.ensure_local_viewer("365865", root=tmp_path, wait_s=0.1, restart=True)
    assert popped.get("mode") == "replace"
    assert result.get("started") is True


def test_open_local_without_bundle_explains_build_first(monkeypatch, tmp_path):
    monkeypatch.setattr(dash, "local_view_listening", lambda ccn, timeout_s=0.4: False)
    monkeypatch.setattr(dash, "_detect_v2", lambda deploy_dir, ccn: False)
    monkeypatch.setattr(dash.cms_data_paths, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(dash.cms_data_paths, "facility_deploy_dir", lambda ccn, root: tmp_path / "missing")
    result = dash.ensure_local_viewer("365865", root=tmp_path, wait_s=1)
    assert result["ok"] is False
    assert "Build locally" in result["detail"]


def test_password_gated_production_failure_is_plain_language():
    msg = dash.operator_failure_message(
        {
            "ok": False,
            "errors": [],
            "stdout": (
                "ERROR: password-gated premium production deploy requires "
                "--production-acceptance and --production-browser-acceptance "
                "(or explicit --allow-live-without-acceptance)\n"
                "deploy                 failed       30.7  exit 1"
            ),
            "stderr": "",
        }
    )
    assert "password-gated" in msg.lower()
    assert msg != "Build failed."


def test_private_data_root_failure_is_plain_language():
    msg = dash.operator_failure_message(
        {
            "ok": False,
            "errors": [],
            "stdout": "Unexpected error: extraction manifests require PRIVATE_DATA_ROOT: unset",
            "stderr": "",
        }
    )
    assert "extraction manifest" in msg.lower()
    assert "audit file" not in msg.lower()
    assert msg != "Build failed."


def test_private_data_root_log_noise_does_not_hide_real_error():
    msg = dash.operator_failure_message(
        {
            "ok": False,
            "errors": [],
            "stdout": (
                "Preparing external deploy staging "
                "(PRIVATE_DATA_ROOT/deploy_staging; punch data never enters repo)…\n"
                "ERROR: password-gated premium production deploy requires "
                "--production-acceptance and --production-browser-acceptance\n"
            ),
            "stderr": "",
        }
    )
    assert "password-gated" in msg.lower()
    assert "audit file" not in msg.lower()


def test_confirm_deploy_missing_private_root_is_plain_language():
    msg = dash.operator_failure_message(
        {
            "ok": False,
            "errors": [],
            "stdout": (
                "ERROR: PRIVATE_DATA_ROOT is required for --confirm-deploy so private "
                "punch/non-CMS artifacts never enter the Cursor workspace.\n"
            ),
            "stderr": "",
        }
    )
    assert "upload could not start" in msg.lower()
    assert "audit file" not in msg.lower()


def test_apply_pbjapp_provision_env_uses_pbjapp_json(tmp_path):
    priv = tmp_path / "private-root"
    priv.mkdir()
    (tmp_path / "data_paths.local.json").write_text(
        '{"private_data_root": "private-root"}',
        encoding="utf-8",
    )
    env = dash.apply_pbjapp_provision_env({"PBJ_REPO_ROOT": "wrong"}, root=tmp_path)
    assert env["PBJ_REPO_ROOT"] == str(tmp_path.resolve())
    assert env["PRIVATE_DATA_ROOT"] == str(priv.resolve())
    kept = dash.apply_pbjapp_provision_env({"PRIVATE_DATA_ROOT": "already"}, root=tmp_path)
    assert kept["PRIVATE_DATA_ROOT"] == "already"


def test_apply_pbjapp_provision_env_uses_existing_fallback_dir(tmp_path, monkeypatch):
    (tmp_path / "data_paths.local.json").write_text("{}", encoding="utf-8")
    fallback = tmp_path / "priv"
    fallback.mkdir()
    monkeypatch.setattr(dash, "_PRIVATE_DATA_FALLBACKS", (fallback,))
    env = dash.apply_pbjapp_provision_env({}, root=tmp_path)
    assert env["PRIVATE_DATA_ROOT"] == str(fallback.resolve())


def test_dirty_critical_failure_is_plain_language():
    msg = dash.operator_failure_message(
        {
            "ok": False,
            "errors": [],
            "stderr": (
                "PRECONDITION BLOCK: git working tree has dirty CRITICAL "
                "runtime/provisioning files; restore/commit them or pass "
                "--allow-dirty-critical-files"
            ),
        }
    )
    assert "PRECONDITION BLOCK" not in msg
    assert "uncommitted dashboard" in msg.lower()
    assert "blocked" in msg.lower()


def test_preflight_failure_is_plain_language():
    msg = dash.operator_failure_message(
        {
            "ok": False,
            "errors": [],
            "stdout": (
                "FAIL Citations ACTIVE-release provenance\n"
                "     artifact changed since provenance was recorded\n"
                "PACKAGE PREFLIGHT FAIL: 1 blocking issue(s)\n"
            ),
            "stderr": "",
        }
    )
    assert msg != "Build failed."
    assert "Citations" in msg
    assert "rewritten" in msg.lower()


def test_operator_failure_message_uses_traceback_exception_line():
    msg = dash.operator_failure_message(
        {
            "ok": False,
            "errors": [],
            "stdout": (
                "create_facility_vercel_package succeeded\n"
                "Traceback (most recent call last):\n"
                '  File "scripts/deploy_vercel_facility.py", line 231, '
                "in _run_v2_inline_js_check\n"
                "    print(f\"... {template.relative_to(root)}\")\n"
                "ValueError: 'D:\\\\PBJapp-data\\\\deployments\\\\pbj320-075358\\\\"
                "templates\\\\superdynamic_dashboard_v2.html' is not in the "
                "subpath of 'C:\\\\Users\\\\egold\\\\PycharmProjects\\\\PBJapp'\n"
            ),
            "stderr": "",
        }
    )
    assert msg != "Build failed."
    assert msg.startswith("ValueError:")
    assert "pbj320-075358" in msg or "relative" in msg.lower() or "subpath" in msg.lower()


def test_operator_failure_message_prefers_error_prefix_over_traceback():
    msg = dash.operator_failure_message(
        {
            "ok": False,
            "errors": [],
            "stdout": (
                "ValueError: should not win\n"
                "ERROR: real operator-facing gate failure\n"
            ),
            "stderr": "",
        }
    )
    assert msg == "real operator-facing gate failure"


def test_builder_view_thin_provider_info_needs_update(monkeypatch, tmp_path):
    deploy = tmp_path / "pbj320-365865"
    deploy.mkdir()
    (deploy / "facility_365865_provider_info_data.csv").write_text(
        "processing_month\n2026-08\n", encoding="utf-8"
    )
    (deploy / "facility_365865_complete_data.csv").write_text("x\n", encoding="utf-8")
    (deploy / "ownership").mkdir()
    (deploy / "ownership" / "NH_Ownership_facility_365865.csv").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(dash, "_pbjapp_bundle_root", lambda: tmp_path)
    monkeypatch.setattr(dash.cms_data_paths, "facility_deploy_dir", lambda ccn, root=None: deploy)
    monkeypatch.setattr(dash, "local_view_listening", lambda ccn, timeout_s=0.4: False)
    monkeypatch.setattr(
        "provider_quarter_mapping.audit_active_provider_quarter_mapping",
        lambda: {"needs_attention": False, "resolved_quarter": "2026Q1"},
    )
    view = dash.builder_view_model(
        {
            "ccn": "365865",
            "facility_name": "Test Home",
            "bundle_state": "GENERATED_LOCALLY",
            "is_v2": True,
            "detail": "A local V2 dashboard is already on disk.",
            "blockers": [],
            "can_build_local": True,
            "can_deploy": True,
            "can_run_preflight": True,
        }
    )
    assert view["local_available"] is True
    assert view["needs_update"] is True
    assert view["primary_build_label"] == "Update local dashboard"
    pi = next(row for row in view["data_status"] if row["id"] == "provider_info")
    assert pi["ok"] is False
    assert "1 month" in pi["text"]
    assert view["can_publish"] is False
    assert "Provider Information" in view["publish_block_reason"]




def test_status_and_view_use_pbj_data_root_not_pbjapp_stub(monkeypatch, tmp_path):
    """Explicit PBJapp root must not shadow PBJ_DATA_ROOT for status/diagnostics."""
    data_root = tmp_path / "data"
    pbjapp = tmp_path / "PBJapp"
    stub = pbjapp / "deployments" / "pbj320-075358"
    real = data_root / "deployments" / "pbj320-075358"
    stub.mkdir(parents=True)
    real.mkdir(parents=True)
    (real / "facility_075358_superdynamic_dashboard.py").write_text("# v2\n", encoding="utf-8")
    (real / "facility_075358_provider_info_data.csv").write_text(
        "processing_month\n" + "\n".join(f"2025-{m:02d}" for m in range(1, 7)) + "\n",
        encoding="utf-8",
    )
    (real / "facility_075358_complete_data.csv").write_text("x\n", encoding="utf-8")
    (real / "ownership").mkdir()
    (real / "ownership" / "NH_Ownership_facility_075358.csv").write_text("x\n", encoding="utf-8")
    (pbjapp / "scripts").mkdir(parents=True)
    (pbjapp / "scripts" / "provision_premium_facility.py").write_text("# stub\n", encoding="utf-8")

    monkeypatch.setenv("PBJ_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbjapp))
    monkeypatch.setattr(dash, "local_view_listening", lambda ccn, timeout_s=0.4: False)
    monkeypatch.setattr(
        "provider_quarter_mapping.audit_active_provider_quarter_mapping",
        lambda: {"needs_attention": False, "resolved_quarter": "2026Q1"},
    )

    # Simulate prior bug: passing PBJapp as explicit root must still resolve to data_root.
    st = dash.facility_dashboard_status("075358", root=pbjapp, run_readiness=False)
    assert st.is_v2 is True
    assert st.deploy_dir is not None
    assert Path(st.deploy_dir).resolve() == real.resolve()
    assert st.can_run_preflight is True
    assert "UNSAFE_PACKAGE_PATH" not in st.blockers
    assert "MISSING_DEPLOY_DIR" not in st.blockers

    view = dash.builder_view_model(st.to_dict())
    assert view["local_available"] is True
    assert view["local_view_url"] == "http://127.0.0.1:5358/"
    assert view["can_run_preflight"] is True
    pi = next(row for row in view["data_status"] if row["id"] == "provider_info")
    assert pi["ok"] is True
    staff = next(row for row in view["data_status"] if row["id"] == "staffing")
    assert staff["ok"] is True
    own = next(row for row in view["data_status"] if row["id"] == "ownership")
    assert own["ok"] is True
    # Publish gate: local V2 present unlocks can_publish path when can_deploy and no blockers
    assert view["can_publish"] is True
    assert view["publish_block_reason"] == ""


def test_builder_view_missing_dashboard(monkeypatch, tmp_path):
    missing = tmp_path / "no-bundle"
    monkeypatch.setattr(dash, "_pbjapp_bundle_root", lambda: tmp_path)
    monkeypatch.setattr(dash.cms_data_paths, "facility_deploy_dir", lambda ccn, root=None: missing)
    monkeypatch.setattr(dash, "local_view_listening", lambda ccn, timeout_s=0.4: False)
    view = dash.builder_view_model(
        {
            "ccn": "365865",
            "facility_name": "Test Home",
            "bundle_state": "NOT_GENERATED",
            "is_v2": False,
            "detail": "No local bundle yet.",
            "blockers": ["MISSING_DEPLOY_DIR"],
            "can_build_local": True,
            "can_deploy": True,
            "can_run_preflight": False,
        }
    )
    assert view["local_available"] is False
    assert view["local_view_url"] is None
    assert view["primary_build_label"] == "Build local dashboard"
    assert view["can_publish"] is False


def test_busy_js_uses_submitter_label():
    js = (Path(__file__).resolve().parents[1] / "static" / "data_ops" / "data_ops.js").read_text(encoding="utf-8")
    assert "e.submitter" in js
    assert "data-do-submitter-keep" in js
    assert "wireDashboardRunForm" in js
    assert "data-do-open-local" in js
    assert "/actions/dashboard/open-local" in js
    assert "window.open(href" in js
    assert "misses >= 5" in js
    assert "window.location.reload" in js
    assert "progressUserHidden" in js

def test_operator_failure_message_surfaces_failed_capability():
    msg = dash.operator_failure_message(
        {
            "ok": False,
            "errors": [],
            "stdout": "Artifact contract check: 075358\n",
            "stderr": (
                "FAILED CAPABILITY: nonnurse_staffing\n"
                "  expected artifact/behavior: facility_075358_nonnurse_daily.csv >= 1000 bytes\n"
                "  actual artifact/behavior: 2 bytes\n"
                "  source/builder used: indexed nonnurse daily or approved fallback / create_vercel_deployment.py\n"
                "  exact next remediation: run create_vercel_deployment.py and verify facility_{ccn}_nonnurse_daily.csv\n"
            ),
        }
    )
    assert msg != "Build failed."
    assert msg.startswith("FAILED CAPABILITY: nonnurse_staffing")
    assert "facility_075358_nonnurse_daily.csv >= 1000 bytes" in msg
    assert "2 bytes" in msg
