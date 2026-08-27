"""Whitelisted manual-source staging. Uploading can never make data ACTIVE."""

from __future__ import annotations

import csv
import re
import tempfile
from pathlib import Path
from typing import BinaryIO

from active_release_registry import sha256_file
from release_control_plane import ReleaseState, record_candidate

MAX_BYTES = 20 * 1024 * 1024
MACPAC = "macpac.state_staffing_standards"
SFF = "cms.sff_pdf_list"


def _copy_limited(stream: BinaryIO, destination: Path) -> None:
    total = 0
    with destination.open("wb") as output:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk: break
            total += len(chunk)
            if total > MAX_BYTES: raise ValueError("upload exceeds 20 MB limit")
            output.write(chunk)


def stage_upload(dataset_id: str, release_id: str, filename: str, stream: BinaryIO, *, root: Path | None = None) -> dict:
    control_root = (root or Path(__file__).resolve().parent).resolve()
    safe_name = Path(filename or "").name
    if dataset_id == SFF:
        if not re.fullmatch(r"20\d{2}-\d{2}", release_id) or Path(safe_name).suffix.lower() != ".pdf":
            raise ValueError("SFF requires a YYYY-MM release and PDF")
        (control_root / "_scratch").mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=control_root / "_scratch") as handle:
            temp = Path(handle.name)
        try:
            _copy_limited(stream, temp)
            from sff_release import stage_pdf
            return stage_pdf(release_id, source_pdf=temp, root=control_root)
        finally:
            temp.unlink(missing_ok=True)
    if dataset_id != MACPAC:
        raise ValueError("dataset is not approved for manual staging")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", release_id) or Path(safe_name).suffix.lower() != ".csv":
        raise ValueError("MACPAC requires a safe version and CSV")
    destination_dir = control_root / "manual" / MACPAC / release_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "macpac_state_staffing_standards.csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", dir=destination_dir) as handle:
        temp = Path(handle.name)
    try:
        _copy_limited(stream, temp)
        with temp.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"State", "Total_Estimated_Staffing_Requirements", "Min_Staffing", "Max_Staffing", "Value_Type", "Is_Federal_Minimum", "Display_Text"}
            missing = sorted(required - set(reader.fieldnames or []))
            rows = list(reader)
        states = [str(row.get("State") or "").strip() for row in rows]
        errors = ([f"missing columns: {missing}"] if missing else []) + (["empty CSV"] if not rows else []) + (["blank or duplicate state"] if any(not s for s in states) or len(states) != len(set(states)) else [])
        validation = {"status": "PASS" if not errors else "FAIL", "row_count": len(rows), "errors": errors}
        if errors:
            record_candidate(dataset_id, release_id, ReleaseState.FAILED, validation=validation, root=control_root)
            return {"dataset_id": dataset_id, "release_id": release_id, "validation": validation}
        temp.replace(destination)
        record_candidate(dataset_id, release_id, ReleaseState.VALIDATED, source_path=destination, validation=validation, metadata={"original_filename": safe_name, "upload_hash": sha256_file(destination), "review_required": True}, root=control_root)
        return {"dataset_id": dataset_id, "release_id": release_id, "validation": validation, "hash": sha256_file(destination)}
    finally:
        temp.unlink(missing_ok=True)
