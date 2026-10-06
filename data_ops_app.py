"""PBJ Data Ops — Flask control-plane entrypoint.

  Local (canonical):  .\\scripts\\start_local_data_ops.ps1  →  http://127.0.0.1:8510/

  export PBJ_DATA_OPS_PASSWORD='…'
  export PBJ_DATA_OPS_SECRET='…'   # optional; session signing (separate from password)

UI → this app → cms_data_ops / data_ops_* services → canonical pipelines.
Does not deploy, does not write pbj-root, does not use Streamlit.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sys
from functools import wraps
from pathlib import Path
from typing import Any
from source_family_inventory import source_family_inventory

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cms_data_ops import (  # noqa: E402
    acquire_nurse,
    acquire_provider_info,
    approve_release_authoritative,
    build_needs_attention_queue,
    build_release_availability_context,
    build_source_operator_workflow,
    check_nurse_cms,
    check_provider_info_cms,
    check_sff_cms,
    derived_signals_payload,
    format_do_timestamp,
    format_release_month_label,
    minimal_record_for_dataset,
    overlay_control_plane_on_snapshot,
    probe_all_sources,
    recommended_next_automation,
    release_review_items,
    snapshots_with_control_plane,
)
from cms_source_registry import get_registry, get_source  # noqa: E402
from active_release_registry import load_registry  # noqa: E402
from release_control_plane import (  # noqa: E402
    control_panel_payload,
    refresh_facility_index,
    refresh_source_health,
)
from release_check import load_check_state  # noqa: E402
from data_ops_approval import (  # noqa: E402
    ApprovalError,
    acknowledge_requires_review,
    read_audit,
)
from data_ops_dashboard import (  # noqa: E402
    builder_view_model,
    check_facility_readiness,
    ensure_local_viewer,
    facility_dashboard_status,
    last_action_summary,
    operator_failure_message,
    run_dashboard_provision,
    run_preflight,
    sanitize_action_payload,
)
from data_ops_dashboard_jobs import job_public_view, read_job, start_dashboard_job  # noqa: E402

PASSWORD_ENV = "PBJ_DATA_OPS_PASSWORD"
SECRET_ENV = "PBJ_DATA_OPS_SECRET"
COOKIE_NAME = "pbj_data_ops_auth"


def _pbj_favicon_path() -> Path | None:
    configured = (os.environ.get("PBJ_REPO_ROOT") or "").strip()
    if configured:
        candidate = Path(configured).resolve() / "pbj_favicon.png"
        if candidate.is_file():
            return candidate
    sibling = (_ROOT.parent / "PBJapp" / "pbj_favicon.png").resolve()
    if sibling.is_file():
        return sibling
    return None


def _source_detail_context(source_id: str, *, theme_publication: Any | None = None) -> dict | None:
    rec = get_source(source_id)
    control = control_panel_payload()
    release_checks = load_check_state()
    check_by_dataset = {row["dataset_id"]: row for row in release_checks.get("datasets", [])}
    control_row = next((row for row in control["datasets"] if row["dataset_id"] == source_id), None)
    if rec is None and control_row is None:
        return None
    snap = next(
        (s for s in probe_all_sources(check_cms=True, theme_publication=theme_publication) if s.source_id == source_id),
        None,
    )
    snapshot = overlay_control_plane_on_snapshot(snap, control) if snap else None
    if (
        snapshot
        and source_id == "cms.health_citations"
        and theme_publication is not None
        and snapshot.get("error")
    ):
        detail = str(snapshot.get("detail") or "")
        if "probe failed" in detail.lower():
            snapshot = dict(snapshot)
            snapshot.pop("error", None)
            if snapshot.get("active_release_id"):
                snapshot["detail"] = "ACTIVE release governed; CMS discovery via theme publication."
    record = rec.to_dict() if rec is not None else minimal_record_for_dataset(source_id)
    availability = build_release_availability_context(
        source_id,
        control_row=control_row,
        check_row=check_by_dataset.get(source_id),
        snapshot=snapshot,
        record=record,
        theme_publication=theme_publication,
    )
    from website_release_readiness import FAMILIES, family_readiness
    family = next((key for key, spec in FAMILIES.items() if any(req.source_id == source_id for req in spec.required_source_evidence)), None)
    readiness = family_readiness(family, control) if family else None
    if readiness:
        evidence = next(s for s in readiness['source_evidence'] if s['source_id'] == source_id)
        observation = evidence['observation']
        availability.update({key: observation.get(key) for key in (
            'cms_dataset_version_label', 'cms_dataset_version_id', 'cms_dataset_version_modified',
            'publisher_resource_id', 'publisher_resource_version', 'publisher_file_uuid',
            'publisher_url', 'publisher_period_basis', 'publisher_sha256', 'publisher_checked_at')})
        availability.update(cms_byte_verified_current=evidence['verified'],
                            cms_release_vintage=evidence['vintage'], snapshot_date=evidence['snapshot'],
                            publisher_latest_label=format_release_month_label(evidence['vintage']) if evidence['vintage'] else 'Not observed',
                            new_release_available=observation.get('status') in {'NEWER', 'REVISED'},
                            availability_summary=evidence['status'])
    workflow = build_source_operator_workflow(
        source_id,
        record=record,
        snapshot=snapshot,
        control_row=control_row,
        release_availability=availability,
        theme_publication=theme_publication,
    )
    if family:
        if workflow.get('operator_reference'):
            workflow['operator_reference'].update(status_label=readiness['source_status'],
                                                  status_tone='current' if readiness['verified_count'] == readiness['required_count'] else 'attention')
        for layer in workflow.get('freshness_layers', []):
            if layer.get('key') == 'cms_source':
                layer.update(status_label=readiness['source_status'], detail=readiness['next_step'],
                             status='current' if readiness['verified_count'] == readiness['required_count'] else 'attention')
        if readiness['next_action'] == 'verify_sources' and readiness['release_id']:
            workflow['next_action'] = dict(label='Verify CMS source bytes', detail='Verify every required source against the exact staged website release.',
                method='post', wired=True, busy_submit=True, busy_label='Verifying CMS bytes…', endpoint='action_website_verify_sources', endpoint_args=dict(family=family, release_id=readiness['release_id']))
    ctx = {
        "record": record,
        "snapshot": snapshot,
        "control_row": control_row,
        "workflow": workflow,
        "release_availability": availability,
        "release_mechanism": check_by_dataset.get(source_id, {}).get("mechanism"),
    }
    if source_id == "cms.provider_info":
        from provider_quarter_mapping import operator_quarter_flow

        ctx["quarter_flow"] = operator_quarter_flow()
    return ctx


def _theme_publication_for_ui(check_cms: bool):
    if not check_cms:
        return None
    from cms_theme_publication import get_latest_nh_theme_publication

    return get_latest_nh_theme_publication()


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(_ROOT / "templates"),
        static_folder=str(_ROOT / "static"),
    )

    password = (os.environ.get(PASSWORD_ENV) or "").strip()
    secret = (os.environ.get(SECRET_ENV) or "").strip()
    # Fail closed: require password. Session secret separate when provided.
    app.config["DATA_OPS_PASSWORD"] = password
    app.config["DATA_OPS_AUTH_CONFIGURED"] = bool(password)
    if password and secret:
        app.secret_key = secret
    elif password:
        # Derive session key from password without storing password in cookies
        app.secret_key = hashlib.sha256(f"pbj-data-ops-session::{password}".encode()).hexdigest()
    else:
        app.secret_key = secrets.token_hex(16)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    def _auth_cookie_value(expected: str) -> str:
        return hashlib.sha256(f"pbj320-data-ops-auth::{expected}".encode()).hexdigest()

    def _is_authenticated() -> bool:
        expected = app.config["DATA_OPS_PASSWORD"]
        if not expected:
            return False
        cookie = request.cookies.get(COOKIE_NAME, "")
        want = _auth_cookie_value(expected)
        if cookie and hmac.compare_digest(cookie, want):
            return True
        return bool(session.get("data_ops_authenticated"))

    def require_auth(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not app.config["DATA_OPS_AUTH_CONFIGURED"]:
                return render_template("data_ops/locked.html"), 503
            if not _is_authenticated():
                return redirect(url_for("login", next=request.path))
            return view(*args, **kwargs)

        return wrapped

    @app.context_processor
    def inject_brand():
        favicon_path = _pbj_favicon_path()
        return {
            "brand_assets": {
                "instructions": False,
                "wordmark": False,
                "dm_sans_local": False,
                "dm_mono_local": False,
                "fonts_via": "fonts.google.com CDN (local DM Sans/Mono unavailable)",
                "mark": "PBJ320 CSS text mark (templates/partials/v2/brand_*)",
            },
            "favicon_href": url_for("favicon_ico") if favicon_path else None,
            "next_automation": recommended_next_automation(),
            "do_datetime": format_do_timestamp,
            "source_labels": {r.source_id: r.human_name for r in get_registry()},
        }

    @app.template_filter("do_datetime")
    def _do_datetime_filter(value: str | None) -> str:
        return format_do_timestamp(value)

    @app.template_filter("do_release_label")
    def _do_release_label_filter(value: str | None) -> str:
        return format_release_month_label(value) or "—"

    @app.get("/favicon.ico")
    def favicon_ico():
        favicon_path = _pbj_favicon_path()
        if not favicon_path:
            return ("", 404)
        return send_from_directory(favicon_path.parent, favicon_path.name, mimetype="image/png")

    @app.get("/login")
    def login():
        if not app.config["DATA_OPS_AUTH_CONFIGURED"]:
            return render_template("data_ops/locked.html"), 503
        if _is_authenticated():
            return redirect(url_for("sources"))
        return render_template("data_ops/login.html", error=None)

    @app.post("/login")
    def login_post():
        if not app.config["DATA_OPS_AUTH_CONFIGURED"]:
            return render_template("data_ops/locked.html"), 503
        expected = app.config["DATA_OPS_PASSWORD"]
        posted = (request.form.get("password") or "").strip()
        if posted and hmac.compare_digest(posted, expected):
            session["data_ops_authenticated"] = True
            resp = redirect(request.args.get("next") or url_for("sources"))
            resp.set_cookie(
                COOKIE_NAME,
                _auth_cookie_value(expected),
                httponly=True,
                samesite="Lax",
                max_age=60 * 60 * 12,
            )
            return resp
        return render_template("data_ops/login.html", error="Invalid password."), 401

    @app.post("/logout")
    @require_auth
    def logout():
        session.clear()
        resp = redirect(url_for("login"))
        resp.delete_cookie(COOKIE_NAME)
        return resp

    @app.get("/")
    @require_auth
    def index():
        return redirect(url_for("sources"))

    @app.get("/sources")
    @require_auth
    def sources():
        check_cms = request.args.get("check_cms", "0") != "0"
        from cms_nh_catalog import load_catalog
        theme_publication = _theme_publication_for_ui(check_cms)
        control = control_panel_payload()
        release_checks = load_check_state()
        check_by_dataset = {row["dataset_id"]: row for row in release_checks.get("datasets", [])}
        snaps = snapshots_with_control_plane(check_cms=check_cms, control=control)
        needs_attention = build_needs_attention_queue(
            control=control,
            check_by_dataset=check_by_dataset,
            snapshots=snaps,
            theme_publication=theme_publication,
        )
        from source_operator_guidance import public_update_guidance, survey_review_guidance
        public_updates = public_update_guidance(control)
        website_sources = {source['source_id']: source for family in public_updates for source in family.get('source_evidence', [])}
        reference_sources = []
        completed_reference_ids = set()
        for row in control.get("datasets", []):
            if row.get("dataset_id") == "cms.survey_summary":
                candidate = row.get("pending") or {}
                if candidate.get("state") == "VALIDATED" and (candidate.get("validation") or {}).get("status") == "PASS":
                    reference_sources.append({"source_id": row["dataset_id"], "release_id": candidate.get("release_id"), **survey_review_guidance(candidate)})
                    completed_reference_ids.add(row["dataset_id"])
        needs_attention = [item for item in needs_attention if item.get("source_id") not in completed_reference_ids]
        for item in needs_attention:
            if item.get("source_id") == "cms.survey_summary":
                candidate = next((r.get("pending") or {} for r in control.get("datasets", [])
                                  if r.get("dataset_id") == "cms.survey_summary"), {})
                guidance = survey_review_guidance(candidate)
                item["concise_state"] = guidance["label"]
                item["next_action"] = {"label": "Review saved Survey Summary", "wired": True,
                                       "read_only": True, "endpoint": "source_detail",
                                       "endpoint_args": {"source_id": "cms.survey_summary"},
                                       "detail": guidance["detail"]}
        availability_by_dataset: dict[str, dict] = {}
        for row in control.get("datasets") or []:
            dataset_id = row.get("dataset_id")
            if not dataset_id:
                continue
            rec = get_source(dataset_id)
            record_dict = rec.to_dict() if rec else minimal_record_for_dataset(dataset_id)
            availability_by_dataset[dataset_id] = build_release_availability_context(
                dataset_id,
                control_row=row,
                check_row=check_by_dataset.get(dataset_id),
                snapshot=next((s for s in snaps if s.get("source_id") == dataset_id), None),
                record=record_dict,
                theme_publication=theme_publication,
            )
        return render_template(
            "data_ops/sources.html",
            snapshots=snaps,
            check_cms=check_cms,
            theme_publication=theme_publication,
            availability_by_dataset=availability_by_dataset,
            signals=derived_signals_payload(),
            registry=[r.to_dict() for r in get_registry()],
            active_releases=load_registry().get("datasets", {}),
            control=control,
            check_by_dataset=check_by_dataset,
            needs_attention=needs_attention,
            releases_checked_at=release_checks.get("checked_at"),
            nh_catalog=load_catalog(),
            family_inventory=source_family_inventory(control, snaps, load_catalog()),
            public_updates=public_updates,
            website_sources=website_sources,
            reference_sources=reference_sources,
        )

    @app.post("/actions/control-panel/refresh")
    @require_auth
    def action_control_panel_refresh():
        try:
            refresh_source_health()
            pbjapp_root = Path(os.environ.get("PBJ_REPO_ROOT") or _ROOT)
            refresh_facility_index(pbjapp_root, ccns=("335581",))
            flash("Control-plane health and facility status refreshed", "ok")
        except Exception as exc:  # noqa: BLE001
            flash(f"Control-plane refresh failed: {exc}", "error")
        return redirect(url_for("sources", check_cms="0"))

    @app.post("/actions/control-panel/check-releases")
    @require_auth
    def action_control_panel_check_releases():
        from release_check import check_releases, production_handlers

        try:
            result = check_releases(
                acquire=False,
                external_handlers=production_handlers(),
            )
            rows = result.get("datasets") or []
            external_rows = [
                row
                for row in rows
                if row.get("mechanism") == "external recurring release"
            ]
            newer = sum(
                1 for row in external_rows if row.get("new_release_available") is True
            )
            failures = sum(
                1
                for row in external_rows
                if str(row.get("status") or "").upper() in {"ERROR", "FAILED", "CHECK_UNAVAILABLE"}
            )
            current = sum(
                1
                for row in external_rows
                if str(row.get("status") or "").upper() == "CURRENT"
            )
            other_attention = sum(
                1
                for row in rows
                if row not in external_rows and row.get("new_release_available") is True
            )
            attention_suffix = (
                f" · {other_attention} non-CMS attention" if other_attention else ""
            )
            catalog = result.get("cms_nh_catalog") or {}
            summary = catalog.get("summary") or {}
            catalog_suffix = (
                f" · NH catalog: {summary.get('datasets', 0)} datasets, "
                f"{summary.get('published', 0)} newer, {summary.get('revised', 0)} revised, "
                f"{summary.get('unchanged', 0)} unchanged, {summary.get('planned_today', 0)} planned today, "
                f"{summary.get('new_datasets', 0)} first/new observations, {summary.get('errors', 0)} errors"
                f", {summary.get('archive_errors', 0)} archive errors"
            ) if catalog else ""
            flash(
                "CMS release check finished"
                f" · {current} current · {newer} newer · {failures} errors"
                f"{attention_suffix}{catalog_suffix}",
                "error" if failures or catalog.get("status") == "ERROR" else "ok",
            )
        except Exception as exc:  # noqa: BLE001
            flash(f"CMS release check failed: {exc}", "error")
        return redirect(url_for("sources", check_cms="0"))

    @app.post("/actions/sources/<source_id>/acquire")
    @require_auth
    def action_source_acquire(source_id: str):
        from release_check import acquire_detected_source

        try:
            result = acquire_detected_source(source_id)
            flash(
                f"{source_id} {result.get('release_id') or ''} acquisition finished · "
                f"{result.get('status') or 'candidate updated'}",
                "ok",
            )
        except Exception as exc:  # noqa: BLE001
            flash(f"{source_id} acquisition failed: {exc}", "error")
        return redirect(url_for("source_detail", source_id=source_id, check_cms="0"))

    @app.get("/sources/<source_id>")
    @require_auth
    def source_detail(source_id: str):
        if source_id == "cms.survey_summary":
            from survey_summary import SOURCE_ID
            from source_operator_guidance import survey_review_guidance
            row = next((r for r in control_panel_payload()["datasets"] if r["dataset_id"] == SOURCE_ID), {})
            candidate = row.get("pending") or {}
            return render_template("data_ops/survey_summary.html", candidate=candidate, active=row.get("active"), guidance=survey_review_guidance(candidate))
        from ownership_pairing import PAIR_SOURCE_ID, build_pair_operator_context

        check_cms = request.args.get("check_cms", "1") != "0"
        if source_id == PAIR_SOURCE_ID:
            ctx = build_pair_operator_context(check_cms=check_cms)
            if ctx is None:
                flash("No pending ownership pair", "error")
                return redirect(url_for("sources"))
            ctx["panel_mode"] = "page"
            return render_template("data_ops/ownership_pair_detail.html", **ctx)
        ctx = _source_detail_context(
            source_id,
            theme_publication=_theme_publication_for_ui(check_cms),
        )
        if ctx is None:
            flash("Unknown source", "error")
            return redirect(url_for("sources"))
        return render_template("data_ops/source_detail.html", **ctx)

    @app.get("/sources/<source_id>/panel")
    @require_auth
    def source_detail_panel(source_id: str):
        if source_id == "cms.survey_summary":
            from source_operator_guidance import survey_review_guidance
            row = next((r for r in control_panel_payload()["datasets"] if r["dataset_id"] == source_id), {})
            candidate = row.get("pending") or {}
            return render_template("data_ops/partials/survey_summary_panel.html", candidate=candidate, active=row.get("active"), guidance=survey_review_guidance(candidate), panel_mode="modal")
        from ownership_pairing import PAIR_SOURCE_ID, build_pair_operator_context

        check_cms = request.args.get("check_cms", "1") != "0"
        if source_id == PAIR_SOURCE_ID:
            ctx = build_pair_operator_context(check_cms=check_cms)
            if ctx is None:
                return ("No pending ownership pair", 404)
            ctx["panel_mode"] = "modal"
            return render_template("data_ops/partials/ownership_pair_panel.html", **ctx)
        ctx = _source_detail_context(
            source_id,
            theme_publication=_theme_publication_for_ui(check_cms),
        )
        if ctx is None:
            return ("Unknown source", 404)
        ctx["panel_mode"] = "modal"
        return render_template("data_ops/partials/source_detail_panel.html", **ctx)

    @app.post("/actions/survey-summary/prepare")
    @require_auth
    def action_survey_summary_prepare():
        from survey_summary import prepare_candidate
        try:
            candidate = prepare_candidate()
            from source_operator_guidance import survey_review_guidance
            guidance = survey_review_guidance(candidate)
            flash(f"Survey Summary: {guidance['label']}. {guidance['detail']}", "ok" if candidate["state"] == "VALIDATED" else "error")
        except Exception as exc:
            flash(f"Survey Summary preparation failed: {exc}", "error")
        return redirect(url_for("source_detail", source_id="cms.survey_summary"))

    @app.post("/actions/ownership-pair/validate")
    @require_auth
    def action_ownership_pair_validate():
        from ownership_pairing import PairValidationError, validate_ownership_pair

        try:
            result = validate_ownership_pair()
            flash(f"Ownership pair {result.get('release_id')} validated", "ok")
        except PairValidationError as exc:
            flash(str(exc), "error")
        except Exception as exc:  # noqa: BLE001
            flash(f"Pair validation failed: {exc}", "error")
        return redirect(url_for("source_detail", source_id="cms.snf_ownership_pair", check_cms="0"))

    @app.post("/actions/ownership-pair/activate")
    @require_auth
    def action_ownership_pair_activate():
        from ownership_pairing import PairPromotionUnavailable, promote_ownership_pair

        try:
            result = promote_ownership_pair()
            flash(
                f"Ownership pair {result.get('release_id')} activated · next: rebuild downstream",
                "ok",
            )
        except PairPromotionUnavailable as exc:
            flash(str(exc), "error")
        except Exception as exc:  # noqa: BLE001
            flash(f"Pair activation failed: {exc}", "error")
        return redirect(url_for("sources"))

    @app.post("/actions/citation-packages/rebuild")
    @require_auth
    def action_citation_packages_rebuild():
        from citation_packages_rebuild import CitationPackagesRebuildError, rebuild_citation_packages

        ccns = [c.strip() for c in request.form.getlist("ccn") if c.strip()]
        try:
            result = rebuild_citation_packages(ccns=ccns or None)
            rebuilt = result.get("rebuilt") or []
            failed = result.get("failed") or []
            remaining = int(result.get("stale_remaining") or 0)
            if failed:
                flash(
                    f"Citation rebuild: {len(rebuilt)} ok, {len(failed)} failed, {remaining} stale remaining",
                    "error",
                )
            elif remaining:
                flash(
                    f"Citation rebuild: {len(rebuilt)} updated · {remaining} packages still stale",
                    "error",
                )
            elif rebuilt:
                flash(f"Citation packages rebuilt for {len(rebuilt)} facilit{'y' if len(rebuilt) == 1 else 'ies'}", "ok")
            else:
                flash("No stale citation packages to rebuild", "ok")
        except CitationPackagesRebuildError as exc:
            flash(str(exc), "error")
        except Exception as exc:  # noqa: BLE001
            flash(f"Citation package rebuild failed: {exc}", "error")
        return redirect(url_for("sources"))

    @app.post("/actions/ownership-downstream/rebuild")
    @require_auth
    def action_ownership_downstream_rebuild():
        from ownership_downstream_rebuild import OwnershipRebuildError, rebuild_ownership_downstream

        try:
            result = rebuild_ownership_downstream()
            remaining = result.get("stale_capabilities_remaining") or []
            if remaining:
                flash(
                    f"Ownership rebuild finished with remaining stale: {', '.join(remaining)}",
                    "error",
                )
            else:
                flash(
                    f"Ownership downstream rebuilt for {result.get('release_label') or result.get('release_id')} · inspect rebuilt state below",
                    "ok",
                )
        except OwnershipRebuildError as exc:
            flash(str(exc), "error")
        except Exception as exc:  # noqa: BLE001
            flash(f"Ownership rebuild failed: {exc}", "error")
        return redirect(url_for("source_detail", source_id="cms.snf_all_owners", check_cms="0"))

    @app.post("/actions/provider-info/sync-quarter-map")
    @require_auth
    def action_provider_quarter_map_sync():
        from provider_quarter_mapping import sync_interval_mapping_from_extract

        try:
            result = sync_interval_mapping_from_extract()
            if result.get("ok"):
                flash(result.get("detail") or "Provider Information quarter map updated", "ok")
            else:
                flash(result.get("detail") or "Could not apply the extracted quarter map", "error")
        except Exception as exc:  # noqa: BLE001
            flash(f"Quarter map update failed: {exc}", "error")
        return redirect(url_for("sources"))

    @app.get("/api/control-plane/status")
    @require_auth
    def api_control_plane_status():
        return jsonify(control_panel_payload())

    @app.get("/api/control-plane/impact/<source_id>")
    @require_auth
    def api_control_plane_impact(source_id: str):
        from release_control_plane import what_would_change

        return jsonify(what_would_change(source_id))

    @app.post("/actions/provider-info/check")
    @require_auth
    def action_pi_check():
        try:
            result = check_provider_info_cms()
            from release_check import record_check_result
            record_check_result("cms.provider_info", result["release_identity"])
            session["last_pi_action"] = {
                "action": "check_cms",
                "cms": result.get("cms"),
                "cms_is_newer": result.get("cms_is_newer"),
                "dry_run_status": (result.get("dry_run") or {}).get("status"),
            }
            flash(
                f"Provider CMS check: {result['release_identity']['status']} · {result['release_identity'].get('detail', '')}",
                "ok",
            )
        except Exception as exc:  # noqa: BLE001
            flash(f"Check failed: {exc}", "error")
        return redirect(url_for("sources"))

    @app.post("/actions/provider-info/acquire")
    @require_auth
    def action_pi_acquire():
        dry = request.form.get("dry_run") == "1"
        try:
            result = acquire_provider_info(dry_run=dry)
            ar = result.get("acquire_report") or {}
            session["last_pi_action"] = {
                "action": "acquire_process",
                "status": ar.get("status"),
                "dry_run": dry,
            }
            flash(f"Acquire {'dry-run ' if dry else ''}status: {ar.get('status')}", "ok")
        except Exception as exc:  # noqa: BLE001
            flash(f"Acquire failed: {exc}", "error")
        return redirect(url_for("sources"))

    @app.post("/actions/provider-info/stage-pbj320")
    @require_auth
    def action_pi_stage_pbj320():
        from pbj320_stage_provider_info import ProviderInfoStageError, stage_provider_info_for_pbj320

        try:
            result = stage_provider_info_for_pbj320()
            status = str(result.get("status") or "")
            manifest = result.get("manifest") or {}
            if status == "NO_MATERIAL_DIFF":
                flash(
                    f"PBJ320 Stage: no material diff — destination already matches staged manifest for {result.get('active_release_id')}",
                    "ok",
                )
            else:
                changed = int(result.get("artifacts_changed") or len(manifest.get("artifacts") or []))
                idempotent = bool(manifest.get("idempotent_no_material_diff"))
                detail = "no file changes" if idempotent else f"{changed} destination artifact(s) updated"
                flash(
                    f"PBJ320 Stage complete · Provider Information {result.get('active_release_id')} · "
                    f"{detail} · gates PASS",
                    "ok",
                )
            return redirect(
                url_for(
                    "pi_stage_manifest",
                    release_id=str(result.get("active_release_id") or ""),
                )
            )
        except ProviderInfoStageError as exc:
            flash(f"PBJ320 Stage failed: {exc}", "error")
        except Exception as exc:  # noqa: BLE001
            flash(f"PBJ320 Stage failed: {exc}", "error")
        return redirect(url_for("source_detail", source_id="cms.provider_info"))

    @app.post("/actions/provider-info/refresh-stage-manifest")
    @require_auth
    def action_pi_refresh_stage_manifest():
        from pbj320_stage_provider_info import ProviderInfoStageError, refresh_provider_info_stage_manifest

        try:
            result = refresh_provider_info_stage_manifest()
            flash(
                f"PBJ320 Stage manifest refreshed for {result.get('active_release_id')} "
                f"(on-disk destinations unchanged)",
                "ok",
            )
            return redirect(
                url_for(
                    "pi_stage_manifest",
                    release_id=str(result.get("active_release_id") or ""),
                )
            )
        except ProviderInfoStageError as exc:
            flash(f"PBJ320 Stage manifest refresh failed: {exc}", "error")
        except Exception as exc:  # noqa: BLE001
            flash(f"PBJ320 Stage manifest refresh failed: {exc}", "error")
        return redirect(url_for("source_detail", source_id="cms.provider_info"))

    @app.get("/website-releases/<family>/<release_id>/panel")
    @require_auth
    def website_release_review_panel(family: str, release_id: str):
        from flask import abort
        from website_release_review import FAMILIES, website_review_context

        if family not in FAMILIES or not release_id or any(c not in '0123456789-' for c in release_id):
            abort(404)
        context = website_review_context(family, release_id, control_panel_payload())
        if context is None:
            return '<h2 id="do-source-panel-title">Website candidate unavailable</h2><p>No staged manifest exists for this release. Return to Sources to review the current readiness.</p><button type="button" data-do-modal-close>Close</button>', 200
        return render_template("data_ops/partials/website_release_review.html", **context)

    @app.post("/actions/website-releases/<family>/<release_id>/verify-sources")
    @require_auth
    def action_website_verify_sources(family: str, release_id: str):
        from flask import abort
        from website_release_readiness import FAMILIES, verify_family_sources
        if family not in FAMILIES or not release_id or any(c not in '0123456789-' for c in release_id):
            abort(404)
        try:
            result = verify_family_sources(family, release_id, control_panel_payload())
            flash(f"{result['label']}: {result['verified_count']}/{result['required_count']} source bytes verified. {result['next_step']}",
                  'ok' if result['verified_count'] == result['required_count'] else 'error')
        except Exception as exc:
            flash(f"Source verification unavailable: {exc}", 'error')
        return redirect(url_for('sources', check_cms='0'))

    @app.get("/provider-info/stage-manifest/<release_id>")
    @require_auth
    def pi_stage_manifest(release_id: str):
        import cms_data_paths
        from pbj320_publication_contract import (
            merge_destination_layers_for_display,
            publication_state_summary,
        )
        from pbj320_publish_provider_info import (
            ProviderInfoPublishError,
            build_publish_review_context,
            load_publication_record,
        )
        from pbj320_stage_provider_info import (
            load_stage_manifest,
            pi_stage_manifest_publishable,
            pi_stage_manifest_stale_detail,
        )

        control_root = cms_data_paths.repo_root()
        manifest = load_stage_manifest(release_id.strip(), root=control_root)
        if manifest is None:
            flash(f"No stage manifest for {release_id}", "error")
            return redirect(url_for("source_detail", source_id="cms.provider_info"))
        stage_publishable = pi_stage_manifest_publishable(manifest, root=control_root)
        stage_stale = (
            str(manifest.get("status") or "") == "STAGED" and not stage_publishable
        )
        stage_stale_detail = pi_stage_manifest_stale_detail(manifest, root=control_root)
        publish_review = None
        publish_blocked: str | None = None
        if stage_publishable:
            try:
                publish_review = build_publish_review_context(
                    release_id=release_id.strip(), root=control_root
                )
            except ProviderInfoPublishError as exc:
                publish_blocked = str(exc)
        publication = load_publication_record(release_id.strip(), root=control_root)
        display_layers = merge_destination_layers_for_display(manifest, publication)
        pub_summary = publication_state_summary(display_layers, publication)
        return render_template(
            "data_ops/pi_stage_manifest.html",
            manifest=manifest,
            release_id=release_id.strip(),
            publish_review=publish_review,
            publish_blocked=publish_blocked,
            publication=publication,
            stage_publishable=stage_publishable,
            stage_stale=stage_stale,
            stage_stale_detail=stage_stale_detail,
            display_layers=display_layers,
            pub_summary=pub_summary,
        )

    @app.post("/actions/provider-info/verify-production")
    @require_auth
    def action_pi_verify_production():
        from pbj320_verify_production import ProviderInfoVerifyError, verify_provider_info_production

        release_id = (request.form.get("release_id") or "").strip() or "2026-08"
        dry_run = (request.form.get("dry_run") or "").strip().lower() in {"1", "true", "yes", "on"}
        try:
            result = verify_provider_info_production(release_id, dry_run=dry_run)
            if result.get("all_pass"):
                flash(
                    f"Production verification PASS · Provider Information {release_id} · "
                    f"{len(result.get('checks') or [])} checks",
                    "ok",
                )
            else:
                failed = [c for c in (result.get("checks") or []) if c.get("result") != "PASS"]
                flash(
                    f"Production verification incomplete · {len(failed)} check(s) failed",
                    "error",
                )
        except ProviderInfoVerifyError as exc:
            flash(f"Production verification failed: {exc}", "error")
        except Exception as exc:  # noqa: BLE001
            flash(f"Production verification failed: {exc}", "error")
        return redirect(url_for("pi_stage_manifest", release_id=release_id))

    @app.post("/actions/provider-info/publish-pbj320")
    @require_auth
    def action_pi_publish_pbj320():
        from pbj320_publish_provider_info import ProviderInfoPublishError, publish_provider_info_for_pbj320

        release_id = (request.form.get("release_id") or "").strip()
        publish_base_sha = (request.form.get("publish_base_sha") or "").strip() or None
        confirm = (request.form.get("confirm") or "").strip().lower() in {"1", "true", "yes", "on"}
        if not confirm:
            flash("Publish refused: explicit confirmation required", "error")
            return redirect(url_for("pi_stage_manifest", release_id=release_id or "2026-08"))
        try:
            result = publish_provider_info_for_pbj320(
                release_id=release_id or None,
                confirm=True,
                push=True,
                publish_base_sha=publish_base_sha,
            )
            if result.get("push_succeeded"):
                flash(
                    f"PBJ320 Publish complete · Provider Information {result.get('release_id')} · "
                    f"commit {str(result.get('commit_sha') or '')[:12]}… pushed · deploy UNKNOWN",
                    "ok",
                )
            else:
                flash(
                    f"PBJ320 Publish committed locally · {result.get('release_id')} · push did not succeed",
                    "error",
                )
            return redirect(url_for("pi_stage_manifest", release_id=str(result.get("release_id") or release_id)))
        except ProviderInfoPublishError as exc:
            flash(f"PBJ320 Publish failed: {exc}", "error")
        except Exception as exc:  # noqa: BLE001
            flash(f"PBJ320 Publish failed: {exc}", "error")
        return redirect(url_for("pi_stage_manifest", release_id=release_id or "2026-08"))

    @app.post("/actions/health-citations/validate")
    @require_auth
    def action_citations_validate():
        release_id = (request.form.get("release_id") or request.args.get("release_id") or "").strip()
        if not release_id:
            flash("Missing release_id for Health Citations validation", "error")
            return redirect(url_for("sources"))
        try:
            from health_citations_acquire import prepare_health_citations_validated_candidate

            result = prepare_health_citations_validated_candidate(release_id)
            candidate = (result.get("candidate") or {}).get("release_id") or release_id
            flash(
                f"Health Citations {candidate} validated — ready for activation review",
                "ok",
            )
        except Exception as exc:  # noqa: BLE001
            flash(f"Validate failed: {exc}", "error")
            return redirect(url_for("sources"))
        return redirect(
            url_for(
                "release_review",
                source_id="cms.health_citations",
                release_id=release_id,
            )
        )

    @app.post("/actions/nurse/check")
    @require_auth
    def action_nurse_check():
        try:
            result = check_nurse_cms()
            from release_check import record_check_result
            record_check_result("cms.pbj_nurse_staffing", result["release_identity"])
            session["last_nurse_action"] = {
                "action": "check_cms",
                "cms": result.get("cms"),
                "cms_is_newer": result.get("cms_is_newer"),
                "dry_run_status": (result.get("dry_run") or {}).get("status"),
            }
            flash(
                f"Nurse reporting quarter {result['cms']['quarter_label']} · {result['release_identity']['status']}",
                "ok",
            )
        except Exception as exc:  # noqa: BLE001
            flash(f"Nurse check failed: {exc}", "error")
        return redirect(url_for("sources"))

    @app.post("/actions/sff/check")
    @require_auth
    def action_sff_check():
        try:
            result = check_sff_cms()
            from release_check import record_check_result
            record_check_result("cms.sff_pdf_list", result["release_identity"])
            cms = result.get("cms") or {}
            flash(
                f"SFF CMS posting {cms.get('posting_label') or cms.get('release_id')} · {result['status']}",
                "ok",
            )
        except Exception as exc:  # noqa: BLE001
            flash(f"SFF check failed: {exc}", "error")
        return redirect(request.form.get("next") or url_for("sources"))

    @app.post("/actions/sff/stage-detected")
    @require_auth
    def action_sff_stage_detected():
        from sff_release import stage_detected_candidate

        try:
            result = stage_detected_candidate()
            validation = result.get("validation") or {}
            flash(
                f"SFF {result.get('release_id')} staged · validation {validation.get('status')}",
                "ok" if validation.get("status") == "PASS" else "error",
            )
        except Exception as exc:  # noqa: BLE001
            flash(f"SFF staging failed: {exc}", "error")
        return redirect(url_for("source_detail", source_id="cms.sff_pdf_list", check_cms="0"))

    @app.post("/actions/nurse/acquire")
    @require_auth
    def action_nurse_acquire():
        dry = request.form.get("dry_run") == "1"
        try:
            result = acquire_nurse(dry_run=dry)
            ar = result.get("acquire_report") or {}
            session["last_nurse_action"] = {
                "action": "acquire_process",
                "status": ar.get("status"),
                "lifecycle": ar.get("lifecycle"),
                "dry_run": dry,
            }
            flash(
                f"Nurse acquire {'dry-run ' if dry else ''}status: {ar.get('status')}",
                "ok",
            )
        except Exception as exc:  # noqa: BLE001
            flash(f"Nurse acquire failed: {exc}", "error")
        return redirect(url_for("sources"))

    @app.post("/actions/nurse/dismiss-candidate")
    @require_auth
    def action_nurse_dismiss_candidate():
        from operator_freshness import audit_nurse_staffing_candidate_state
        from release_control_plane import control_panel_payload, discard_candidate

        control = control_panel_payload()
        row = next(
            (r for r in control.get("datasets") or [] if r.get("dataset_id") == "cms.pbj_nurse_staffing"),
            None,
        )
        audit = audit_nurse_staffing_candidate_state(control_row=row)
        if not audit.get("is_redundant_reacquisition"):
            flash("Nurse candidate is not a safe redundant re-acquisition to dismiss.", "error")
            return redirect(url_for("sources"))
        removed = discard_candidate(
            "cms.pbj_nurse_staffing",
            reason=audit.get("kind") or "redundant_same_quarter_reacquisition",
        )
        if removed:
            flash(f"Dismissed redundant CY2026Q1 re-acquisition candidate.", "ok")
        else:
            flash("No nurse candidate to dismiss.", "error")
        return redirect(url_for("sources"))

    @app.get("/release-review")
    @require_auth
    def release_review():
        check_cms = request.args.get("check_cms", "1") != "0"
        focus_source_id = (request.args.get("source_id") or "").strip() or None
        focus_release_id = (request.args.get("release_id") or "").strip() or None
        theme_publication = _theme_publication_for_ui(check_cms)
        control = control_panel_payload()
        items = release_review_items(
            check_cms=check_cms,
            control=control,
            theme_publication=theme_publication,
            focus_source_id=focus_source_id,
            focus_release_id=focus_release_id,
        )
        return render_template(
            "data_ops/release_review.html",
            items=items,
            audit=read_audit(limit=50),
            control=control,
            focus_source_id=focus_source_id,
            focus_release_id=focus_release_id,
        )

    @app.post("/actions/acknowledge")
    @require_auth
    def action_acknowledge():
        source_id = (request.form.get("source_id") or "").strip()
        release_id = (request.form.get("release_id") or "").strip()
        note = (request.form.get("note") or "").strip()
        try:
            acknowledge_requires_review(source_id, release_id, note=note)
            flash(f"Acknowledged {source_id} {release_id}", "ok")
        except Exception as exc:  # noqa: BLE001
            flash(str(exc), "error")
        return redirect(
            url_for(
                "release_review",
                source_id=source_id or None,
                release_id=release_id or None,
            )
        )

    @app.post("/actions/approve")
    @require_auth
    def action_approve():
        from release_review_policy import assert_promotion_eligible, post_activation_operator_target

        source_id = (request.form.get("source_id") or "").strip()
        release_id = (request.form.get("release_id") or "").strip()
        note = (request.form.get("note") or "").strip()
        _ = request.form.get("zweli_status")
        try:
            assert_promotion_eligible(source_id, release_id)
            approve_release_authoritative(source_id, release_id, note=note)
            target = post_activation_operator_target(source_id, release_id=release_id)
            flash(target["flash"], "ok")
            return redirect(url_for(target["redirect_endpoint"], **target["redirect_args"]))
        except ApprovalError as exc:
            flash(str(exc), "error")
        except Exception as exc:  # noqa: BLE001
            flash(str(exc), "error")
        return redirect(
            url_for(
                "release_review",
                source_id=source_id or None,
                release_id=release_id or None,
            )
        )

    @app.get("/dashboard-builder")
    @require_auth
    def dashboard_builder():
        ccn = (request.args.get("ccn") or "").strip()
        status = None
        pi = None
        if ccn:
            snaps = probe_all_sources(check_cms=False, run_zweli=True)
            pi = next((s for s in snaps if s.source_id == "cms.provider_info"), None)
            status = facility_dashboard_status(
                ccn,
                zweli_state=pi.zweli_status if pi else None,
                provider_release_id=pi.release_id if pi else None,
                structural_ok=(pi.structural_status != "FAIL") if pi else True,
                provider_processed=bool(pi and pi.processed),
                provider_available=bool(pi and pi.local_raw_present),
            ).to_dict()
            status = builder_view_model(status)
        last_action = session.pop("last_dash_action", None)
        return render_template(
            "data_ops/dashboard_builder.html",
            ccn=ccn,
            view=status,
            last_action=last_action,
            last_summary=last_action_summary(last_action) if last_action else "",
        )

    @app.post("/actions/dashboard/readiness")
    @require_auth
    def action_dash_readiness():
        ccn = (request.form.get("ccn") or "").strip()
        result = check_facility_readiness(ccn)
        session["last_dash_action"] = {"action": "readiness", "result": result}
        flash(
            "Readiness OK" if result.get("ok") else f"Readiness exit {result.get('exit_code')}",
            "ok" if result.get("ok") else "error",
        )
        return redirect(url_for("dashboard_builder", ccn=ccn))

    @app.post("/actions/dashboard/run")
    @require_auth
    def action_dash_run():
        ccn = (request.form.get("ccn") or "").strip()
        intent = (request.form.get("intent") or "").strip()
        access_mode = (request.form.get("access_mode") or "password_required").strip()
        password = (request.form.get("dashboard_password") or "").strip()
        confirm_ccn = (request.form.get("confirm_ccn") or "").strip()
        confirm_publish = (request.form.get("confirm_publish") or "") == "1"
        refresh_data = (request.form.get("refresh_data") or "") == "1"
        allow_dirty_local = True
        wants_json = (
            request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or (request.accept_mimetypes["application/json"] >= request.accept_mimetypes["text/html"])
        )
        if wants_json:
            started = start_dashboard_job(
                ccn=ccn,
                intent=intent,
                access_mode=access_mode,
                password=password,
                confirm_ccn=confirm_ccn,
                confirm_publish=confirm_publish,
                refresh_data=refresh_data,
            )
            if not started.get("ok"):
                return jsonify(started), 400
            return jsonify(started)
        result = run_dashboard_provision(
            ccn=ccn,
            intent=intent,
            access_mode=access_mode,
            password=password,
            confirm_ccn=confirm_ccn,
            confirm_publish=confirm_publish,
            refresh_data=refresh_data,
            allow_dirty_tree=allow_dirty_local,
            allow_dirty_critical_files=allow_dirty_local,
        )
        session["last_dash_action"] = sanitize_action_payload(
            {"action": "run", "result": result}
        )
        if result.get("ok"):
            if result.get("dry_run"):
                flash("Preview plan finished (no package, no deploy).", "ok")
            elif result.get("deployed"):
                flash("Build and deploy finished.", "ok")
            else:
                flash("Local build finished (not deployed).", "ok")
        else:
            errors = result.get("errors") or []
            flash(
                "; ".join(str(e) for e in errors)
                if errors
                else (result.get("operator_message") or operator_failure_message(result)),
                "error",
            )
        return redirect(url_for("dashboard_builder", ccn=ccn))

    @app.post("/actions/dashboard/open-local")
    @require_auth
    def action_dash_open_local():
        ccn = (request.form.get("ccn") or request.args.get("ccn") or "").strip()
        result = ensure_local_viewer(ccn, restart=(request.form.get("restart") or "") == "1")
        wants_json = (
            request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or (request.accept_mimetypes["application/json"] >= request.accept_mimetypes["text/html"])
        )
        if wants_json:
            return jsonify(sanitize_action_payload(result))
        if result.get("ok") and result.get("url"):
            return redirect(str(result["url"]))
        flash(result.get("detail") or "Could not open the local dashboard.", "error")
        return redirect(url_for("dashboard_builder", ccn=ccn))

    @app.get("/actions/dashboard/job/<job_id>")
    @require_auth
    def action_dash_job(job_id: str):
        job = read_job(job_id)
        if not job:
            return jsonify({"ok": False, "error": "unknown job"}), 404
        return jsonify(job_public_view(job))

    @app.post("/actions/dashboard/preflight")
    @require_auth
    def action_dash_preflight():
        ccn = (request.form.get("ccn") or "").strip()
        result = run_preflight(ccn)
        session["last_dash_action"] = {"action": "preflight", "result": result}
        flash(
            "Preflight OK" if result.get("ok") else "Preflight failed",
            "ok" if result.get("ok") else "error",
        )
        return redirect(url_for("dashboard_builder", ccn=ccn))

    return app


app = create_app()


def main() -> int:
    port = int(os.environ.get("PBJ_DATA_OPS_PORT") or "8510")
    if not (os.environ.get(PASSWORD_ENV) or "").strip():
        print(
            f"ERROR: set {PASSWORD_ENV} before starting Data Ops (fail closed).",
            file=sys.stderr,
        )
        return 1
    url = f"http://127.0.0.1:{port}/"
    print(f"Data Ops listening: {url}", flush=True)
    app.run(host="127.0.0.1", port=port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
