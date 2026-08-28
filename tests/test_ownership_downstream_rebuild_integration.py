"""Integration: ownership downstream rebuild against live registry (local only)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PBJ = ROOT.parent / "PBJapp"
sys.path.insert(0, str(ROOT))


@pytest.mark.integration
def test_live_rebuild_ownership_downstream_when_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(ROOT / "state" / "active_releases.json"))
    monkeypatch.setenv("PBJ_REPO_ROOT", str(PBJ))

    from ownership_downstream_rebuild import audit_ownership_downstream_stale, rebuild_ownership_downstream

    pre = audit_ownership_downstream_stale(root=ROOT, pbj_root=PBJ)
    if not pre.get("is_stale"):
        pytest.skip("ownership downstream already current")

    result = rebuild_ownership_downstream(root=ROOT, pbj_root=PBJ)
    post = result.get("post_audit") or audit_ownership_downstream_stale(root=ROOT, pbj_root=PBJ)

    assert result.get("release_id") == "2026-07-31"
    assert post.get("is_stale") is False
    assert (post.get("stale_capabilities") or []) == []

    policy = __import__("json").loads((PBJ / "ownership" / "ownership_release_policy.json").read_text(encoding="utf-8"))
    assert policy.get("active_release_date") == "2026-07-31"
    lookup = PBJ / "ownership" / "_derived" / "cms_snf_ownership_ccn_bridge" / "release_2026-07-31_lookup.json"
    assert lookup.is_file()
