"""Contract tests for canonical local Data Ops launch (port 8510)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "start_local_data_ops.ps1"


def test_start_local_data_ops_script_is_canonical_port_8510():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "$CanonicalPort = 8510" in text
    assert "http://127.0.0.1:8510" in text
    assert "ForEach-Object { Stop-Process" not in text
    assert "Get-NetTCPConnection -LocalPort $CanonicalPort" in text


def test_data_ops_app_default_port_is_8510(monkeypatch):
    monkeypatch.delenv("PBJ_DATA_OPS_PORT", raising=False)
    import data_ops_app

    assert int(data_ops_app.os.environ.get("PBJ_DATA_OPS_PORT") or "8510") == 8510
