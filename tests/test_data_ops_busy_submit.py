"""Tests for long-running Data Ops form submit feedback."""

from __future__ import annotations

import pytest


def test_render_post_action_form_busy_submit_markup() -> None:
    from data_ops_app import create_app
    from flask import render_template_string

    app = create_app()
    action = {
        "label": "Stage for PBJ320",
        "endpoint": "action_pi_stage_pbj320",
        "wired": True,
        "method": "post",
        "busy_submit": True,
        "busy_label": "Staging Provider Information…",
        "busy_detail": "Running validation and build gates.",
    }
    with app.test_request_context("/"):
        html = render_template_string(
            "{% from 'data_ops/partials/source_detail_macros.html' import render_workflow_action %}"
            "{{ render_workflow_action(action) }}",
            action=action,
        )
    assert "data-do-busy-form" in html
    assert "data-do-busy-submit" in html
    assert "Staging Provider Information" in html
    assert "Running validation and build gates" in html
    assert "/actions/provider-info/stage-pbj320" in html.replace("\\", "/")


def test_wire_busy_forms_blocks_duplicate_submit() -> None:
    """JS guard: second submit attempt is prevented while busy flag is set."""
    js_path = pytest.importorskip("pathlib").Path(__file__).resolve().parents[1] / "static" / "data_ops" / "data_ops.js"
    text = js_path.read_text(encoding="utf-8")
    assert "data-do-busy-active" in text
    assert "e.preventDefault()" in text
    assert "wireBusyForms" in text
