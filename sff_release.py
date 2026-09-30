"""Governed CMS Special Focus Facility posting lifecycle.

CMS does not expose a stable data-api dataset for this posting. Acquisition is
explicit/monitored and this module never promotes a release.
"""

from __future__ import annotations

import csv
import json
import re
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from active_release_registry import get_active_release, registry_path, sha256_file
from release_control_plane import ReleaseState, load_candidates, record_candidate

DATASET_ID = "cms.sff_pdf_list"
OFFICIAL_AUGUST_2026_URL = "https://www.cms.gov/files/document/sff-posting-candidate-list-august-2026.pdf"
SFF_POSTING_URL_TEMPLATE = "https://www.cms.gov/files/document/sff-posting-candidate-list-{month_slug}-{year}.pdf"
SFF_POSTING_UPDATED_RE = re.compile(r"Updated\s+([A-Za-z]+)\s+(20\d{2})", re.I)
_MONTH_SLUGS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_MONTH_NAME_TO_NUM = {name: index for index, name in enumerate(_MONTH_SLUGS, start=1)}
CATEGORIES = {"Table A": "CURRENT_SFF", "Table B": "GRADUATED", "Table C": "NO_LONGER_PARTICIPATING", "Table D": "SFF_CANDIDATE"}
PBJ_TABLE_FILES = {"Table A": "sff_table_a.csv", "Table B": "sff_table_b.csv", "Table C": "sff_table_c.csv", "Table D": "sff_table_d.csv"}
USPS = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR VI GU MP AS".split())
CCN_RE = re.compile(r"^[0-9A-Z]{6}$")


def sff_posting_url_candidates(*, months_back: int = 8, anchor: datetime | None = None) -> list[tuple[str, str]]:
    """Return ``(url, url_release_id)`` pairs newest-first for recent CMS posting slugs."""
    now = anchor or datetime.now(timezone.utc)
    year, month = now.year, now.month
    out: list[tuple[str, str]] = []
    for _ in range(max(1, months_back)):
        slug = _MONTH_SLUGS[month - 1]
        url_release_id = f"{year}-{month:02d}"
        url = SFF_POSTING_URL_TEMPLATE.format(month_slug=slug, year=year)
        out.append((url, url_release_id))
        month -= 1
        if month < 1:
            month = 12
            year -= 1
    return out


def parse_sff_posting_updated_label(pdf_bytes: bytes) -> tuple[str, str] | None:
    """Parse ``Updated August 2026`` from CMS SFF posting PDF bytes or metadata."""
    for pattern in (
        SFF_POSTING_UPDATED_RE,
        re.compile(r"/Title\s*\([^)]*Updated\s+([A-Za-z]+)\s+(20\d{2})", re.I),
    ):
        match = pattern.search(pdf_bytes.decode("latin-1", errors="ignore"))
        if not match:
            continue
        month_num = _MONTH_NAME_TO_NUM.get(match.group(1).lower())
        if not month_num:
            continue
        year = int(match.group(2))
        release_id = f"{year}-{month_num:02d}"
        label = f"{match.group(1).title()} {year}"
        return release_id, label
    return None


def _fetch_sff_posting_pdf(
    url: str,
    *,
    fetch_bytes: Any | None = None,
) -> tuple[int, bytes | None]:
    if fetch_bytes is not None:
        payload = fetch_bytes(url)
        if not payload or not payload.startswith(b"%PDF-"):
            return 404, None
        return 200, payload
    req = urllib.request.Request(url, headers={"User-Agent": "PBJ-data-ops/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            status = getattr(response, "status", 200) or 200
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 404, 410}:
            return exc.code, None
        raise
    if status in {404, 410} or not payload.startswith(b"%PDF-"):
        return 404, None
    return status, payload


def discover_latest_cms_sff_posting(
    *,
    fetch_bytes: Any | None = None,
    months_back: int = 8,
    anchor: datetime | None = None,
) -> dict[str, Any]:
    """Discover the newest CMS-posted SFF PDF by slug probe + ``Updated …`` label."""
    discovered: list[dict[str, Any]] = []
    for url, url_release_id in sff_posting_url_candidates(months_back=months_back, anchor=anchor):
        status, payload = _fetch_sff_posting_pdf(url, fetch_bytes=fetch_bytes)
        if status != 200 or not payload:
            continue
        parsed = parse_sff_posting_updated_label(payload)
        if parsed:
            release_id, posting_label = parsed
        else:
            release_id, posting_label = url_release_id, url_release_id
        discovered.append(
            {
                "source_url": url,
                "url_release_id": url_release_id,
                "release_id": release_id,
                "posting_label": posting_label,
            }
        )
    if not discovered:
        raise RuntimeError("no CMS SFF posting PDF discovered at monitored slug URLs")
    discovered.sort(key=lambda row: (row["release_id"], row["url_release_id"]), reverse=True)
    best = discovered[0]
    return {
        "release_id": best["release_id"],
        "posting_label": best["posting_label"],
        "source_url": best["source_url"],
        "url_release_id": best["url_release_id"],
        "candidates": discovered,
    }


def check_sff_cms(
    *,
    fetch_bytes: Any | None = None,
    root: Path | None = None,
    months_back: int = 8,
    record_detection: bool = True,
) -> dict[str, Any]:
    """Compare ACTIVE ``cms.sff_pdf_list`` against the newest CMS-posted SFF PDF."""
    control_root = (root or Path(__file__).resolve().parent).resolve()
    active = get_active_release(DATASET_ID, registry_path(control_root)) or {}
    active_id = active.get("active_release_id")
    cms = discover_latest_cms_sff_posting(fetch_bytes=fetch_bytes, months_back=months_back)
    cms_release_id = str(cms["release_id"])
    cms_is_newer = not active_id or cms_release_id > str(active_id)
    if cms_is_newer and record_detection:
        record_candidate(
            DATASET_ID,
            cms_release_id,
            ReleaseState.DETECTED,
            metadata={
                "source_url": cms["source_url"],
                "posting_period": cms_release_id,
                "posting_label": cms["posting_label"],
                "url_release_id": cms["url_release_id"],
                "discovery": "monitored CMS monthly SFF slug probe",
            },
            root=control_root,
        )
    return {
        "action": "check_cms",
        "cms": {
            "release_id": cms_release_id,
            "posting_label": cms["posting_label"],
            "source_url": cms["source_url"],
            "url_release_id": cms["url_release_id"],
            "candidates_found": len(cms.get("candidates") or []),
        },
        "active_release_id": active_id,
        "cms_is_newer": cms_is_newer,
        "status": "CURRENT" if active_id and not cms_is_newer else ("NEWER_AVAILABLE" if cms_is_newer else "UNKNOWN"),
    }


def _rows_from_pdf(pdf_path: Path) -> list[dict[str, str]]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required to parse CMS SFF postings") from exc
    rows: list[dict[str, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            tables = page.find_tables()
            if not tables:
                continue
            table = tables[0]
            extracted = table.extract()
            title = str((extracted[0] or [""])[0] or "")
            table_key = next((key for key in CATEGORIES if title.startswith(key)), None)
            if not table_key:
                continue
            # CMS supplies vertical column rules but no horizontal row rules.
            # Provider numbers are stable row anchors; bucket words on that
            # baseline into the PDF's actual columns, preserving blank cells.
            # The title row spans the first logical column, so pdfplumber reports
            # that column as the whole table. Remaining column left edges are
            # accurate; prepend the table's left edge to recover provider number.
            bounds = [table.bbox[0]] + [col.bbox[0] for col in table.columns[1:]] + [table.bbox[2]]
            words = page.extract_words(x_tolerance=1, y_tolerance=1)
            anchors = [word for word in words if CCN_RE.fullmatch(word["text"].upper()) and any(ch.isdigit() for ch in word["text"]) and bounds[0] - 2 <= word["x0"] < bounds[1] and word["top"] > 75]
            for anchor in anchors:
                cells: list[list[str]] = [[] for _ in range(len(bounds) - 1)]
                center_y = (anchor["top"] + anchor["bottom"]) / 2
                for word in words:
                    word_y = (word["top"] + word["bottom"]) / 2
                    if abs(word_y - center_y) > 2.2:
                        continue
                    word_x = (word["x0"] + word["x1"]) / 2
                    for index in range(len(bounds) - 1):
                        if bounds[index] <= word_x < bounds[index + 1]:
                            cells[index].append(word["text"])
                            break
                values = [" ".join(parts).strip() for parts in cells]
                if len(values) < 8:
                    raise RuntimeError(f"SFF table geometry changed on page {page_number}")
                row = {"ccn": values[0].upper(), "facility_name": values[1], "address": values[2], "city": values[3], "state": values[4].upper(), "zip": values[5], "phone": values[6], "category": CATEGORIES[table_key], "source_table": table_key, "source_page": str(page_number), "status_date": "", "survey_criteria": "", "months_in_status": ""}
                if table_key == "Table A":
                    row["status_date"], row["survey_criteria"], row["months_in_status"] = values[7], values[8], values[9]
                elif table_key in {"Table B", "Table C"}:
                    row["status_date"], row["months_in_status"] = values[7], values[8]
                else:
                    row["months_in_status"] = values[7]
                rows.append(row)
    if not rows:
        raise RuntimeError("no governed SFF tables were parsed")
    return rows


def _provider_ccns(root: Path) -> set[str]:
    active = get_active_release("cms.provider_info", registry_path(root))
    if not active:
        return set()
    from urllib.parse import unquote, urlparse
    raw = unquote(urlparse(str(active.get("source_uri") or "")).path)
    if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    source = Path(raw)
    if not source.is_file():
        return set()
    with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        normalized = {re.sub(r"[^a-z0-9]", "", name.lower()): name for name in (reader.fieldnames or [])}
        field = next((normalized[key] for key in ("federalprovidernumber", "providernumber", "ccn") if key in normalized), None)
        return {str(row.get(field) or "").strip().upper() for row in reader} if field else set()


def validate_rows(rows: list[dict[str, str]], *, provider_ccns: set[str] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    invalid_ccns = sorted({row["ccn"] for row in rows if not CCN_RE.fullmatch(row["ccn"])})
    invalid_states = sorted({row["state"] for row in rows if row["state"] not in USPS})
    categories = Counter(row["category"] for row in rows)
    duplicate_pairs = [key for key, count in Counter((row["category"], row["ccn"]) for row in rows).items() if count > 1]
    if invalid_ccns: errors.append(f"invalid CCNs: {invalid_ccns[:5]}")
    if invalid_states: errors.append(f"invalid states: {invalid_states}")
    if set(categories) != set(CATEGORIES.values()): errors.append("one or more required CMS status categories are absent")
    if duplicate_pairs: errors.append(f"duplicate CCN within category: {duplicate_pairs[:5]}")
    required_fields = ("ccn", "facility_name", "address", "city", "state", "zip", "phone", "months_in_status", "source_page")
    blank_required = sorted({field for field in required_fields if any(not row[field] for row in rows)})
    if blank_required: errors.append(f"blank required fields: {blank_required}")
    bad_dates = [row["ccn"] for row in rows if row["status_date"] and not re.fullmatch(r"\d{2}/\d{2}/\d{4}", row["status_date"])]
    if bad_dates: errors.append(f"invalid status dates: {bad_dates[:5]}")
    bad_months = [row["ccn"] for row in rows if not row["months_in_status"].isdigit() or not 0 <= int(row["months_in_status"]) <= 200]
    if bad_months: errors.append(f"invalid months-in-status: {bad_months[:5]}")
    bad_criteria = [row["ccn"] for row in rows if row["category"] == "CURRENT_SFF" and row["survey_criteria"] not in {"", "Met", "Not Met"}]
    if bad_criteria: errors.append(f"invalid survey criteria: {bad_criteria[:5]}")
    incomplete_surveys = [row["ccn"] for row in rows if row["category"] == "CURRENT_SFF" and bool(row["status_date"]) != bool(row["survey_criteria"])]
    if incomplete_surveys: errors.append(f"inspection/criteria mismatch: {incomplete_surveys[:5]}")
    known = provider_ccns or set()
    unique_ccns = {row["ccn"] for row in rows}
    unmatched = unique_ccns - known if known else set()
    return {"status": "PASS" if not errors else "FAIL", "validated_at": datetime.now(timezone.utc).isoformat(), "row_count": len(rows), "unique_ccn_count": len(unique_ccns), "category_counts": dict(sorted(categories.items())), "cross_category_ccn_count": len(rows) - len(unique_ccns), "provider_info_mapping": {"reference_available": bool(known), "matched": len(unique_ccns & known), "unmatched": len(unmatched), "unmatched_ccns": sorted(unmatched)}, "errors": errors}


def _write_pbj_table_contract(rows: list[dict[str, str]], release_dir: Path) -> list[dict[str, str]]:
    """Write the established pbj-root sff_table_[a-d].csv handoff contract."""
    artifacts = []
    for table, filename in PBJ_TABLE_FILES.items():
        subset = [row for row in rows if row["source_table"] == table]
        headers = ["Provider Number", "Facility Name", "Address", "City", "State", "Zip", "Phone Number"]
        if table == "Table A": headers += ["Most Recent Inspection", "Met Survey Criteria", "Months as an SFF"]
        elif table == "Table B": headers += ["Date of Graduation", "Months as an SFF"]
        elif table == "Table C": headers += ["Date of Termination", "Months as an SFF"]
        else: headers += ["Months as an SFF Candidate"]
        path = release_dir / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers); writer.writeheader()
            for row in subset:
                out = {"Provider Number": row["ccn"], "Facility Name": row["facility_name"], "Address": row["address"], "City": row["city"], "State": row["state"], "Zip": row["zip"], "Phone Number": row["phone"]}
                if table == "Table A": out.update({"Most Recent Inspection": row["status_date"], "Met Survey Criteria": row["survey_criteria"], "Months as an SFF": row["months_in_status"]})
                elif table == "Table B": out.update({"Date of Graduation": row["status_date"], "Months as an SFF": row["months_in_status"]})
                elif table == "Table C": out.update({"Date of Termination": row["status_date"], "Months as an SFF": row["months_in_status"]})
                else: out["Months as an SFF Candidate"] = row["months_in_status"]
                writer.writerow(out)
        artifacts.append({"role": filename, "filename": filename, "source_uri": path.as_uri(), "hash": sha256_file(path), "rows": len(subset)})
    return artifacts


def stage_pdf(release_id: str, *, source_url: str | None = None, source_pdf: Path | None = None, root: Path | None = None, fetch_bytes=None) -> dict[str, Any]:
    """Acquire, normalize and register a VALIDATED candidate; never promote."""
    control_root = (root or Path(__file__).resolve().parent).resolve()
    if not re.fullmatch(r"20\d{2}-\d{2}", release_id): raise ValueError("SFF release_id must be YYYY-MM")
    if not source_url and not source_pdf: raise ValueError("provide a CMS source URL or source_pdf")
    release_dir = control_root / "sff" / "releases" / release_id
    release_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = release_dir / f"cms_sff_posting_{release_id}.pdf"
    if source_pdf:
        if source_url and not str(source_url).startswith("https://www.cms.gov/"): raise ValueError("SFF provenance URL must use https://www.cms.gov/")
        payload, provenance_url = Path(source_pdf).read_bytes(), source_url
    else:
        if not str(source_url).startswith("https://www.cms.gov/"): raise ValueError("SFF acquisition is restricted to an https://www.cms.gov/ URL")
        if fetch_bytes: payload = fetch_bytes(source_url)
        else:
            req = urllib.request.Request(source_url, headers={"User-Agent": "PBJ-data-ops/1.0"})
            with urllib.request.urlopen(req, timeout=180) as response: payload = response.read()
        provenance_url = source_url
    if not payload.startswith(b"%PDF-") or len(payload) < 10_000: raise RuntimeError("source is not a plausible CMS PDF")
    parsed_posting = parse_sff_posting_updated_label(payload)
    if parsed_posting is None:
        raise RuntimeError("CMS SFF PDF is missing an embedded 'Updated Month Year' identity")
    if parsed_posting[0] != release_id:
        raise RuntimeError(
            f"CMS SFF PDF identity {parsed_posting[0]} does not match candidate {release_id}"
        )
    if pdf_path.exists() and pdf_path.read_bytes() != payload: raise RuntimeError("refusing to overwrite a differing staged SFF PDF")
    if not pdf_path.exists(): pdf_path.write_bytes(payload)
    rows = _rows_from_pdf(pdf_path)
    validation = validate_rows(rows, provider_ccns=_provider_ccns(control_root))
    normalized = release_dir / f"cms_sff_posting_{release_id}.csv"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=release_dir, suffix=".tmp") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows); temp = Path(handle.name)
    temp.replace(normalized)
    pbj_handoff = _write_pbj_table_contract(rows, release_dir)
    evidence = {"dataset_id": DATASET_ID, "release_id": release_id, "source_url": provenance_url, "source_pdf": pdf_path.name, "source_pdf_hash": sha256_file(pdf_path), "normalized_hash": sha256_file(normalized), "pbj_handoff": pbj_handoff, "validation": validation}
    (release_dir / "validation.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state = ReleaseState.VALIDATED if validation["status"] == "PASS" else ReleaseState.FAILED
    record_candidate(DATASET_ID, release_id, state, source_path=normalized if state == ReleaseState.VALIDATED else None, validation=validation, metadata={"source_url": provenance_url, "source_pdf_uri": pdf_path.as_uri(), "source_pdf_hash": evidence["source_pdf_hash"], "posting_period": release_id, "parser": "sff_release.py:v1", "pbj_handoff": pbj_handoff}, root=control_root)
    return evidence


def stage_detected_candidate(*, root: Path | None = None, fetch_bytes=None) -> dict[str, Any]:
    """Stage the governed DETECTED SFF candidate using its recorded CMS URL."""
    control_root = (root or Path(__file__).resolve().parent).resolve()
    candidate = (load_candidates(control_root).get("datasets") or {}).get(DATASET_ID) or {}
    state = str(candidate.get("state") or "").upper()
    if state != ReleaseState.DETECTED.value:
        raise RuntimeError(f"SFF candidate must be DETECTED before staging; found {state or 'none'}")
    release_id = str(candidate.get("release_id") or "")
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    source_url = str((metadata or {}).get("source_url") or "")
    if not source_url.startswith("https://www.cms.gov/"):
        raise RuntimeError("detected SFF candidate is missing a trusted cms.gov source URL")
    return stage_pdf(
        release_id,
        source_url=source_url,
        root=control_root,
        fetch_bytes=fetch_bytes,
    )
