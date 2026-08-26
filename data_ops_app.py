"""PBJ Data Ops — Flask control-plane entrypoint.

  export PBJ_DATA_OPS_PASSWORD='…'
  export PBJ_DATA_OPS_SECRET='…'   # optional; session signing (separate from password)
  python data_ops_app.py

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

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
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
    check_nurse_cms,
    check_provider_info_cms,
    derived_signals_payload,
    probe_all_sources,
    recommended_next_automation,
    release_review_items,
)
from cms_source_registry import get_registry, get_source  # noqa: E402
from data_ops_approval import (  # noqa: E402
    ApprovalError,
    acknowledge_requires_review,
    read_audit,
)
from data_ops_dashboard import (  # noqa: E402
    COLD_NEW_V2_PATH,
    EXISTING_V2_REFRESH_PATH,
    check_facility_readiness,
    facility_dashboard_status,
    refresh_existing_v2_runtime,
    run_preflight,
)

PASSWORD_ENV = "PBJ_DATA_OPS_PASSWORD"
SECRET_ENV = "PBJ_DATA_OPS_SECRET"
COOKIE_NAME = "pbj_data_ops_auth"


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
        return {
            "brand_assets": {
                "instructions": False,
                "wordmark": False,
                "dm_sans_local": False,
                "dm_mono_local": False,
                "fonts_via": "fonts.google.com CDN (local DM Sans/Mono unavailable)",
                "mark": "PBJ320 CSS text mark (templates/partials/v2/brand_*)",
            },
            "next_automation": recommended_next_automation(),
        }

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
        check_cms = request.args.get("check_cms", "1") != "0"
        snaps = [s.to_dict() for s in probe_all_sources(check_cms=check_cms)]
        return render_template(
            "data_ops/sources.html",
            snapshots=snaps,
            check_cms=check_cms,
            signals=derived_signals_payload(),
            registry=[r.to_dict() for r in get_registry()],
        )

    @app.get("/sources/<source_id>")
    @require_auth
    def source_detail(source_id: str):
        rec = get_source(source_id)
        if rec is None:
            flash("Unknown source", "error")
            return redirect(url_for("sources"))
        snap = next(
            (s for s in probe_all_sources(check_cms=True) if s.source_id == source_id),
            None,
        )
        return render_template(
            "data_ops/source_detail.html",
            record=rec.to_dict(),
            snapshot=snap.to_dict() if snap else None,
        )

    @app.post("/actions/provider-info/check")
    @require_auth
    def action_pi_check():
        try:
            result = check_provider_info_cms()
            session["last_pi_action"] = {
                "action": "check_cms",
                "cms": result.get("cms"),
                "cms_is_newer": result.get("cms_is_newer"),
                "dry_run_status": (result.get("dry_run") or {}).get("status"),
            }
            flash(
                f"CMS {result['cms']['data_vintage_label']} · newer={result['cms_is_newer']}",
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

    @app.post("/actions/nurse/check")
    @require_auth
    def action_nurse_check():
        try:
            result = check_nurse_cms()
            session["last_nurse_action"] = {
                "action": "check_cms",
                "cms": result.get("cms"),
                "cms_is_newer": result.get("cms_is_newer"),
                "dry_run_status": (result.get("dry_run") or {}).get("status"),
            }
            flash(
                f"Nurse CMS {result['cms']['quarter_label']} · newer={result['cms_is_newer']}",
                "ok",
            )
        except Exception as exc:  # noqa: BLE001
            flash(f"Nurse check failed: {exc}", "error")
        return redirect(url_for("sources"))

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

    @app.get("/release-review")
    @require_auth
    def release_review():
        items = release_review_items(check_cms=True)
        return render_template(
            "data_ops/release_review.html",
            items=items,
            audit=read_audit(limit=50),
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
        return redirect(url_for("release_review"))

    @app.post("/actions/approve")
    @require_auth
    def action_approve():
        source_id = (request.form.get("source_id") or "").strip()
        release_id = (request.form.get("release_id") or "").strip()
        note = (request.form.get("note") or "").strip()
        # Form may display Zweli status but must never be authoritative.
        _ = request.form.get("zweli_status")
        try:
            approve_release_authoritative(source_id, release_id, note=note)
            flash(f"Approved {source_id} {release_id}", "ok")
        except ApprovalError as exc:
            flash(str(exc), "error")
        except Exception as exc:  # noqa: BLE001
            flash(str(exc), "error")
        return redirect(url_for("release_review"))

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
        return render_template(
            "data_ops/dashboard_builder.html",
            ccn=ccn,
            status=status,
            existing_refresh_path=EXISTING_V2_REFRESH_PATH,
            cold_new_path=COLD_NEW_V2_PATH,
            last_action=session.pop("last_dash_action", None),
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

    @app.post("/actions/dashboard/refresh")
    @require_auth
    def action_dash_refresh():
        ccn = (request.form.get("ccn") or "").strip()
        snaps = probe_all_sources(check_cms=False, run_zweli=True)
        pi = next((s for s in snaps if s.source_id == "cms.provider_info"), None)
        st = facility_dashboard_status(
            ccn,
            zweli_state=pi.zweli_status if pi else None,
            provider_release_id=pi.release_id if pi else None,
            structural_ok=(pi.structural_status != "FAIL") if pi else True,
            provider_processed=bool(pi and pi.processed),
            provider_available=bool(pi and pi.local_raw_present),
            run_readiness=False,
        )
        if not st.can_generate_refresh:
            session["last_dash_action"] = {
                "action": "refresh",
                "ok": False,
                "blockers": st.blockers,
                "detail": st.detail,
            }
            flash(
                st.detail or ("Blocked: " + ", ".join(st.blockers)),
                "error",
            )
            return redirect(url_for("dashboard_builder", ccn=ccn))
        result = refresh_existing_v2_runtime(ccn)
        session["last_dash_action"] = {"action": "refresh", "result": result}
        flash(
            "Refresh OK" if result.get("ok") else result.get("detail") or result.get("error") or "Failed",
            "ok" if result.get("ok") else "error",
        )
        return redirect(url_for("dashboard_builder", ccn=ccn))

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
    app.run(host="127.0.0.1", port=port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
