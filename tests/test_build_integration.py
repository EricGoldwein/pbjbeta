"""End-to-end build tests against real source PDFs (read in place; never
copied into this worktree). Scoped to a handful of publications via a
monkeypatched source list so the suite stays fast while still exercising the
real parser, real governed-release lookup, and real Provider Info file.
"""

import csv
import json
from pathlib import Path

import pytest

import sff_history.build as build_module
from sff_history.paths import PublicationSource, archive_extracted_dir, pbj_data_ops_root

ARCHIVE_DIR = archive_extracted_dir()
JULY_2026 = ARCHIVE_DIR / "SFF Posting with Candidate List -  July 2026.pdf"
MARCH_2023 = ARCHIVE_DIR / "SFF Posting with Candidate List - March 2023.pdf"
GOVERNED_ROOT = pbj_data_ops_root()

pytestmark = pytest.mark.skipif(
    not (JULY_2026.is_file() and MARCH_2023.is_file() and (GOVERNED_ROOT / "state" / "active_releases.json").is_file()),
    reason="real archive/governed-release sources not staged locally",
)


def _scoped_sources() -> list[PublicationSource]:
    from sff_history.paths import _governed_active_source  # noqa: SLF001 - test-only read

    governed = _governed_active_source()
    assert governed is not None and governed.release_id == "2026-08"
    return [
        PublicationSource(release_id="2023-03", pdf_path=MARCH_2023, source_kind="archive"),
        PublicationSource(release_id="2026-07", pdf_path=JULY_2026, source_kind="archive"),
        governed,
    ]


def _run_build(tmp_path: Path, monkeypatch, name: str) -> dict:
    monkeypatch.setattr(build_module, "discover_publication_sources", _scoped_sources)
    out_dir = tmp_path / name
    return build_module.build_canonical_dataset(out_dir=out_dir)


def test_build_is_idempotent(tmp_path, monkeypatch):
    manifest_a = _run_build(tmp_path, monkeypatch, "run_a")
    manifest_b = _run_build(tmp_path, monkeypatch, "run_b")

    # Every data artifact must be byte-identical across runs...
    hashes_a = dict(manifest_a["file_sha256"])
    hashes_b = dict(manifest_b["file_sha256"])
    # ...except publications.csv, which legitimately carries a per-run
    # built_at processing timestamp (audit metadata, not derived data) that
    # is expected to differ. Compare its rows with that column stripped.
    hashes_a.pop("publications.csv")
    hashes_b.pop("publications.csv")
    assert hashes_a == hashes_b

    def _rows_without_built_at(run_name: str) -> list[dict]:
        with (tmp_path / run_name / "publications.csv").open(encoding="utf-8") as f:
            return [{k: v for k, v in row.items() if k != "built_at"} for row in csv.DictReader(f)]

    assert _rows_without_built_at("run_a") == _rows_without_built_at("run_b")

    assert manifest_a["publication_ids"] == manifest_b["publication_ids"]
    assert manifest_a["observation_count"] == manifest_b["observation_count"]
    assert manifest_a["derived_change_count"] == manifest_b["derived_change_count"]
    assert manifest_a["derived_graduation_event_count"] == manifest_b["derived_graduation_event_count"]
    assert manifest_a["derived_interval_count"] == manifest_b["derived_interval_count"]


def test_isolated_march_2023_produces_no_cross_month_derivation(tmp_path, monkeypatch):
    # 2023-03 is not calendar-adjacent to 2026-07/2026-08 in this scoped run,
    # so it must never be compared against them for OBSERVED_CHANGE/interval
    # purposes even though both are PASS publications.
    manifest = _run_build(tmp_path, monkeypatch, "isolated")
    out_dir = tmp_path / "isolated"
    with (out_dir / "derived_changes.csv").open(encoding="utf-8") as f:
        changes = list(csv.DictReader(f))
    with (out_dir / "derived_intervals.csv").open(encoding="utf-8") as f:
        intervals = list(csv.DictReader(f))
    assert all("2023-03" not in (c["from_publication_id"], c["to_publication_id"]) for c in changes)
    assert all("2023-03" not in i["publication_ids"].split(";") for i in intervals)


def test_known_graduation_end_to_end(tmp_path, monkeypatch):
    manifest = _run_build(tmp_path, monkeypatch, "grad")
    out_dir = tmp_path / "grad"

    with (out_dir / "observations.csv").open(encoding="utf-8") as f:
        obs = list(csv.DictReader(f))
    july_row = next(r for r in obs if r["ccn"] == "045421" and r["publication_id"] == "2026-07")
    august_row = next(r for r in obs if r["ccn"] == "045421" and r["publication_id"] == "2026-08")
    assert july_row["normalized_category"] == "CURRENT_SFF"
    assert august_row["normalized_category"] == "GRADUATED"
    assert august_row["explicit_status_date"] == "07/16/2026"

    with (out_dir / "derived_changes.csv").open(encoding="utf-8") as f:
        changes = list(csv.DictReader(f))
    change = next(c for c in changes if c["ccn"] == "045421")
    assert change["from_categories"] == "CURRENT_SFF"
    assert change["to_categories"] == "GRADUATED"
    assert change["asserted_date"] == ""

    with (out_dir / "derived_graduation_events.csv").open(encoding="utf-8") as f:
        events = list(csv.DictReader(f))
    event = next(e for e in events if e["ccn"] == "045421")
    assert event["event_date"] == "2026-07-16"


def test_multi_table_membership_end_to_end(tmp_path, monkeypatch):
    manifest = _run_build(tmp_path, monkeypatch, "multi")
    out_dir = tmp_path / "multi"
    with (out_dir / "observations.csv").open(encoding="utf-8") as f:
        obs = list(csv.DictReader(f))
    matches = [r for r in obs if r["ccn"] == "045143" and r["publication_id"] == "2026-08"]
    assert {r["normalized_category"] for r in matches} == {"CURRENT_SFF", "SFF_CANDIDATE"}


def test_provider_info_disagreement_recorded_without_becoming_a_change(tmp_path, monkeypatch):
    manifest = _run_build(tmp_path, monkeypatch, "recon")
    out_dir = tmp_path / "recon"
    with (out_dir / "reconciliation_provider_info.csv").open(encoding="utf-8") as f:
        recon = list(csv.DictReader(f))
    row = next(r for r in recon if r["ccn"] == "045421" and r["publication_id"] == "2026-08")
    assert row["reconciliation_note"] == "DISAGREE_VALUE"
    # No derived_changes row is keyed off Provider Info at all -- confirmed
    # architecturally in test_provider_info_check.py; here we only confirm
    # the July->August PDF-only OBSERVED_CHANGE still carries no asserted
    # date despite the disagreement existing in the same build.
    with (out_dir / "derived_changes.csv").open(encoding="utf-8") as f:
        changes = list(csv.DictReader(f))
    change = next(c for c in changes if c["ccn"] == "045421")
    assert change["asserted_date"] == ""
