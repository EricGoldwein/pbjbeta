"""Bounded read-only observations of existing adjacent-source artifacts.

Presence is not semantic validation or an operational release. No downloads,
recursive data scan, lifecycle changes, or assertions about Git merge state.
"""
import json
import os
from pathlib import Path

from data_path_resolver import resolve_data_path


def adjacent_source_observations(data_root: Path | None = None) -> dict:
    data_root = data_root or Path(os.environ.get("PBJ_DATA_ROOT") or resolve_data_path("provider_info").path.parent)
    result = {}
    penalties = sorted((data_root / "Penalties").glob("NH_Penalties_*.csv"))
    if penalties:
        result["penalties"] = {"evidence": "; ".join(p.name for p in penalties), "health": "RAW_PRESENT_UNVALIDATED",
                               "next_action": "Validate the retained Penalties release against its source contract"}
    receipts = sorted((data_root / "cms" / "hcris" / "pilot_outputs").glob("*/*/validation_receipt.json"))
    if receipts:
        receipt = receipts[-1]
        try:
            data = json.loads(receipt.read_text())
            limitations = data.get("unresolved_or_unproven") or []
            result["hcris"] = {"evidence": f"{data.get('pilot_id', 'Pilot receipt')} · {data.get('generated_at', 'undated')}",
                               "health": data.get("production_status") or "RECEIPT_PRESENT_STATUS_UNKNOWN",
                               "next_action": "Review pilot limitation: " + str(limitations[0]) if limitations else "Review the existing pilot receipt before operational integration"}
        except (ValueError, OSError):
            result["hcris"] = {"evidence": str(receipt), "health": "UNREADABLE_RECEIPT", "next_action": "Inspect the existing pilot receipt"}
    nppes = data_root / "clinicians" / "_sources" / "nppes"
    if nppes.is_dir():
        result["npi_nppes"] = {"evidence": "NPPES source directory exists; validation not established", "health": "RAW_LOCATION_PRESENT",
                               "next_action": "Inspect the existing NPPES receipt before acquisition"}
    else:
        result["npi_nppes"] = {"evidence": "No NPPES artifact at the canonical clinicians source location",
                               "health": "NOT_OBSERVED_IN_RUNTIME", "next_action": "Verify any external NPPES acquisition receipt before adding a pipeline"}
    return result
