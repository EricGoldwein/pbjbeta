# SFF continuation fix and September revalidation

The governed parser now inherits established Table A/B/C/D context only after checking the expected column headers, ordered column geometry and CCN row anchors. Inherited geometry must match the preceding section. Unrelated/ambiguous tables, multiple tables after section establishment, and CCN anchors without table geometry fail explicitly. Text is never concatenated across pages.

Regression proof: hiding September page 5's title preserves 654 rows, including its 38 rows from 315125 through 535022. Hiding Table D page 11's title preserves its 50 rows from 075228 through 165580. A substituted unrelated table fails explicitly. The August/September parsing and pending-candidate workflow suite passed 69 tests.

August: A=87, B=126, C=1, D=441, total=655. September: A=87, B=125, C=2, D=440, total=654, unique CCNs=639, cross-category CCNs=15. All nine September alphanumeric CCNs are unchanged.

September revalidation used the already-downloaded official PDF after an exact row-for-row comparison with the existing candidate and matching four-table category/CCN multisets. The PDF, normalized CSV and all four table hashes remained unchanged. Evidence records `sff_release.py:v2-continuation` and the parser source hash. The candidate remains VALIDATED; August ACTIVE and its artifacts, and every other candidate, remain unchanged.

Browser verification with the current August ACTIVE / September VALIDATED registry shows Review READY / NEXT, Make ACTIVE NOT YET, and Release Review as the next recommended action. Existing layout/styles are retained. No activation, public rebuild or publication occurred.

## PBJ-root follow-up

The preceding read-only audit ran current pbj-root `scripts/sff/extract_sff_posting.py` on September and obtained 645 rows, omitting all nine alphanumeric CCNs. Governed Data Ops produced the independently verified complete 654-row dataset. Recommend eventually consuming reviewed, governed Data Ops four-table outputs in pbj-root instead of maintaining an independent PDF parser, with a separate integration change and provenance checks. No pbj-root code was modified here.
