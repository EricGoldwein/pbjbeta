"""PBJ Data Ops v0 — password-protected internal CMS source status page.

Auth: set environment variable ``PBJ_DATA_OPS_PASSWORD`` (never commit credentials).
Fail closed when unset. Provider Info actions call PR #63 acquire machinery only.
"""

from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cms_data_ops import (  # noqa: E402
    acquire_provider_info,
    check_provider_info_cms,
    probe_all_sources,
    recommended_next_automation,
)
from cms_source_registry import get_registry  # noqa: E402

st.set_page_config(
    page_title="PBJ Data Ops | 320",
    page_icon="pbj_images/pbj_favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PASSWORD_ENV = "PBJ_DATA_OPS_PASSWORD"


def _expected_password() -> str:
    return (os.environ.get(PASSWORD_ENV) or "").strip()


def _require_auth() -> None:
    expected = _expected_password()
    if not expected:
        st.error(
            f"PBJ Data Ops is locked. Set environment variable `{PASSWORD_ENV}` "
            "(do not commit credentials)."
        )
        st.stop()

    if st.session_state.get("data_ops_authenticated") is True:
        return

    st.title("PBJ Data Ops")
    st.caption("Internal operations — password required.")
    with st.form("data_ops_login"):
        pwd = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Unlock")
    if submitted:
        if pwd and hmac.compare_digest(pwd, expected):
            st.session_state.data_ops_authenticated = True
            st.rerun()
        st.error("Invalid password.")
    st.stop()


def _status_color(status: str) -> str:
    return {
        "CURRENT": "#1b7f4e",
        "CMS_NEWER": "#b45309",
        "LOCAL_RAW_ONLY": "#1d4ed8",
        "PROCESSING_REQUIRED": "#7c3aed",
        "READY_FOR_HANDOFF": "#0f766e",
        "UNKNOWN": "#6b7280",
        "ERROR": "#b91c1c",
    }.get(status, "#6b7280")


def _fmt(val) -> str:
    if val is None or val == "":
        return "—"
    return str(val)


_require_auth()

st.title("PBJ Data Ops")
st.caption(
    "Canonical CMS source registry + local/CMS status. "
    "PBJapp is the data factory. No deploy, git push, or pbj-root writes from this page."
)

col_a, col_b = st.columns([1, 3])
with col_a:
    check_cms = st.checkbox("Query CMS metastore (Provider Info / SFF)", value=True)
with col_b:
    if st.button("Refresh status"):
        st.session_state.pop("data_ops_snapshots", None)
        st.session_state.pop("data_ops_pi_action", None)

if "data_ops_snapshots" not in st.session_state:
    with st.spinner("Probing sources…"):
        st.session_state.data_ops_snapshots = [
            s.to_dict() for s in probe_all_sources(check_cms=check_cms)
        ]

snapshots = st.session_state.data_ops_snapshots
registry = {r.source_id: r for r in get_registry()}

# Compact table
rows = []
for s in snapshots:
    rows.append(
        {
            "source": s["human_name"],
            "CMS latest": _fmt(s.get("cms_latest")),
            "PBJapp latest": _fmt(s.get("pbjapp_latest")),
            "format": ", ".join(s.get("formats") or []),
            "cadence": s.get("cadence") or "—",
            "status": s.get("status"),
            "last checked": _fmt(s.get("last_checked")),
            "last successful local processing": _fmt(
                s.get("last_successful_local_processing")
            ),
            "automation": s.get("automation_level"),
        }
    )

st.subheader("Source status")
st.dataframe(rows, use_container_width=True, hide_index=True)

st.subheader("Per-source detail")
for s in snapshots:
    color = _status_color(s["status"])
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 2, 2])
        with c1:
            st.markdown(
                f"**{s['human_name']}**  \n"
                f"`{s['source_id']}` · "
                f"<span style='color:{color};font-weight:600'>{s['status']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(s.get("detail") or "")
        with c2:
            st.write(f"CMS: **{_fmt(s.get('cms_latest'))}**")
            st.write(f"PBJapp: **{_fmt(s.get('pbjapp_latest'))}**")
        with c3:
            st.write(f"Format: {_fmt(', '.join(s.get('formats') or []))}")
            st.write(f"Cadence: {_fmt(s.get('cadence'))}")
            st.write(f"Checked: {_fmt(s.get('last_checked'))}")
            st.write(f"Processed: {_fmt(s.get('last_successful_local_processing'))}")

        if s["source_id"] == "cms.provider_info":
            a1, a2, a3 = st.columns(3)
            with a1:
                if st.button("Refresh / check CMS", key="pi_check"):
                    try:
                        result = check_provider_info_cms()
                        st.session_state.data_ops_pi_action = result
                        st.session_state.data_ops_snapshots = [
                            x.to_dict() for x in probe_all_sources(check_cms=True)
                        ]
                        st.success(
                            f"CMS: {result['cms']['data_vintage_label']} · "
                            f"newer={result['cms_is_newer']} · "
                            f"dry_run={result['dry_run'].get('status')}"
                        )
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Check failed: {exc}")
            with a2:
                if st.button("Acquire / process current", key="pi_acq"):
                    try:
                        result = acquire_provider_info(dry_run=False)
                        st.session_state.data_ops_pi_action = result
                        st.session_state.data_ops_snapshots = [
                            x.to_dict() for x in probe_all_sources(check_cms=True)
                        ]
                        ar = result.get("acquire_report") or {}
                        st.success(f"Acquire status: {ar.get('status')}")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Acquire failed: {exc}")
            with a3:
                if st.button("Dry-run acquire", key="pi_dry"):
                    try:
                        result = acquire_provider_info(dry_run=True)
                        st.session_state.data_ops_pi_action = result
                        st.info(
                            f"Dry-run: {(result.get('acquire_report') or {}).get('status')}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Dry-run failed: {exc}")
        else:
            st.caption("Read-only in Data Ops v0.")

        rec = registry.get(s["source_id"])
        if rec and rec.automation_notes:
            with st.expander("Registry / audit notes"):
                st.write(rec.automation_notes)
                if rec.notes:
                    st.write(rec.notes)
                st.write("Evidence:")
                for e in rec.evidence:
                    st.code(e, language=None)

if st.session_state.get("data_ops_pi_action"):
    with st.expander("Last Provider Info action result", expanded=False):
        st.json(st.session_state.data_ops_pi_action)

nxt = recommended_next_automation()
st.subheader("Suggested next automation")
st.write(f"**{nxt['human_name']}** (`{nxt['source_id']}`)")
st.write(nxt["why"])

if st.button("Lock page"):
    st.session_state.data_ops_authenticated = False
    st.session_state.pop("data_ops_snapshots", None)
    st.rerun()
