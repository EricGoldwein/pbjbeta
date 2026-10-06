"""Provider Information processing-month → PBJ quarter mapping from extracts."""

from __future__ import annotations

import json
from pathlib import Path

from prov_info_quarter_map import get_quarter_from_processing_month
from provider_quarter_mapping import (
    audit_active_provider_quarter_mapping,
    mapping_row_from_extract,
    sync_interval_mapping_from_extract,
)


def test_august_2026_maps_to_q1_2026() -> None:
    assert get_quarter_from_processing_month("2026-08") == "Q1 2026"


def test_mapping_row_uses_interval_extract(tmp_path: Path) -> None:
    pi = tmp_path / "provider_info"
    pi.mkdir()
    (pi / "NH_DataCollectionIntervals_Aug2026.csv").write_text(
        "Measure Code,Data Collection Period From Date,Data Collection Period Through Date,Processing Date\n"
        "STAFFING_LEVELS,01/01/2026,03/31/2026,20260801\n",
        encoding="utf-8",
    )
    row = mapping_row_from_extract("2026-08", root=tmp_path)
    assert row is not None
    assert row["processing_month"] == "08-2026"
    assert row["interval_staffing_level_quarter"] == "Q1 2026"


def test_sync_writes_interval_json(tmp_path: Path) -> None:
    pi = tmp_path / "provider_info"
    pi.mkdir()
    (pi / "NH_DataCollectionIntervals_Aug2026.csv").write_text(
        "Measure Code,Data Collection Period From Date,Data Collection Period Through Date,Processing Date\n"
        "STAFFING_LEVELS,01/01/2026,03/31/2026,20260801\n",
        encoding="utf-8",
    )
    result = sync_interval_mapping_from_extract("2026-08", root=tmp_path)
    assert result["ok"] is True
    path = tmp_path / "static" / "data" / "interval_quarter_mapping.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    months = [r["processing_month"] for r in data["rows"]]
    assert months[0] == "08-2026"
    assert data["rows"][0]["interval_staffing_level_quarter"] == "Q1 2026"


def test_needs_attention_when_shipped_json_missing_active_month(tmp_path: Path) -> None:
    registry = tmp_path / "state" / "active_releases.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "cms.provider_info": {
                        "active_release_id": "2026-08",
                        "status": "ACTIVE",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    json_path = tmp_path / "static" / "data" / "interval_quarter_mapping.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"rows": [{"processing_month": "06-2026"}]}), encoding="utf-8")
    audit = audit_active_provider_quarter_mapping(root=tmp_path)
    assert audit["needs_attention"] is True
    assert audit["release_id"] == "2026-08"

    from cms_data_ops import _provider_quarter_mapping_attention_item

    item = _provider_quarter_mapping_attention_item(root=tmp_path)
    assert item is not None
    assert item["source_id"] == "cms.provider_info.quarter_map"
    assert item["next_action"]["endpoint"] == "action_provider_quarter_map_sync"


def test_operator_flow_flags_thin_facility_slice(tmp_path: Path) -> None:
    deploy = tmp_path / "deployments" / "pbj320-365865"
    deploy.mkdir(parents=True)
    (deploy / "facility_365865_provider_info_data.csv").write_text(
        "processing_date\n2026-08-01\n",
        encoding="utf-8",
    )
    from provider_quarter_mapping import operator_quarter_flow

    flow = operator_quarter_flow(ccn="365865", root=tmp_path)
    assert flow["history"]["ready"] is False
    assert "1 month" in flow["history"]["detail"]
    assert flow["prove_command"].endswith("--prove")


def test_quarter_flow_script_lists_real_tests() -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "check_provider_quarter_flow.py"
    spec = importlib.util.spec_from_file_location("check_provider_quarter_flow", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ops = Path(__file__).resolve().parents[1]
    for rel in mod.DATA_OPS_PROVE:
        assert (ops / rel).is_file(), rel
    app = mod._pbjapp_root()
    if app is not None:
        for rel in mod.PBJAPP_PROVE:
            assert (app / rel).is_file(), rel
