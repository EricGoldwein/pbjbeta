from sff_history.coverage import build_coverage, gapless_runs
from sff_history.publications import Publication

ERA = "era3b_ccn_2023_03_plus"
PARSER = "sff_history.pdf_parser:v1"


def _pub(publication_id: str, status: str = "PASS") -> Publication:
    return Publication(
        publication_id=publication_id,
        publication_period=publication_id,
        updated_date=None,
        updated_date_precision="unknown",
        updated_label_raw="",
        updated_date_period_mismatch=False,
        source_filename=f"{publication_id}.pdf",
        source_kind="archive",
        sha256="0" * 64,
        era_id=ERA,
        parser_version=PARSER,
        page_count=10,
        row_count=1,
        validation_status=status,
    )


def test_known_gap_shows_up_as_missing_month():
    # Reproduces the 2023-11 -> [gap: 2023-12] -> 2024-01 pattern confirmed
    # in the archive (SFF_ARCHIVE_AUDIT.md S4).
    pubs = [_pub("2023-11"), _pub("2024-01")]
    rows = build_coverage(pubs, start=(2023, 11), end=(2024, 1))
    by_month = {r.year_month: r for r in rows}
    assert by_month["2023-11"].present is True
    assert by_month["2023-12"].present is False
    assert by_month["2024-01"].present is True
    assert by_month["2024-01"].gap_before is True  # preceding month (Dec) is missing
    assert by_month["2023-11"].gap_before is False


def test_gapless_runs_splits_at_missing_month():
    pubs = [_pub("2023-10"), _pub("2023-11"), _pub("2024-01"), _pub("2024-02")]
    runs = gapless_runs(pubs)
    ids = [[p.publication_id for p in run] for run in runs]
    assert ids == [["2023-10", "2023-11"], ["2024-01", "2024-02"]]


def test_fail_publication_treated_as_absent_for_run_purposes():
    pubs = [_pub("2023-10"), _pub("2023-11", status="FAIL"), _pub("2023-12")]
    runs = gapless_runs(pubs)
    ids = [[p.publication_id for p in run] for run in runs]
    assert ids == [["2023-10"], ["2023-12"]]
