"""Orchestrator: build the canonical SFF history layer end to end.

Deterministic and idempotent: given the same source PDFs, re-running this
produces byte-identical CSV output (rows are always written in a fixed sort
order, never insertion order, and every ID is derived from stable inputs —
publication_id from the in-PDF date, observation_id from
publication_id+table+ccn+sequence). Safe to re-run at any time; it never
mutates a source PDF or any governed release in pbj-data-ops/pbj-root.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .coverage import build_coverage, coverage_to_rows
from .derive import derive_changes, derive_graduation_events, derive_intervals
from .observations import OBSERVATION_FIELDS, Observation, build_observations, validate_observations
from .paths import discover_publication_sources, output_dir, provider_info_normalized_csv
from .provider_info_check import RECONCILIATION_FIELDS, load_provider_info_ccn_status, reconcile
from .publications import PUBLICATION_FIELDS, Publication, build_publication
from .schema import ERA_3B, PARSER_VERSION


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def build_canonical_dataset(*, out_dir: Path | None = None) -> dict[str, Any]:
    """Run the full Phase-1 build. Returns a manifest dict (also written to
    ``build_manifest.json``).
    """
    target = out_dir or output_dir()
    sources = discover_publication_sources()
    if not sources:
        raise RuntimeError("no Era-3b publication sources discovered")

    publications: list[Publication] = []
    observations_by_pub: dict[str, list[Observation]] = {}

    for source in sources:
        publication, raw_rows = build_publication(source)
        if publication.validation_status == "PASS":
            observations = build_observations(
                publication.publication_id, raw_rows, era_id=ERA_3B, parser_version=PARSER_VERSION
            )
            field_validation = validate_observations(observations)
            if field_validation["status"] != "PASS":
                publication.validation_status = "FAIL"
                publication.validation_errors.extend(field_validation["errors"])
            publication.validation_warnings.extend(field_validation.get("warnings") or [])
            observations_by_pub[publication.publication_id] = observations
        else:
            observations_by_pub[publication.publication_id] = []
        publications.append(publication)

    publications.sort(key=lambda p: p.publication_id)

    publication_rows = [p.to_row() for p in publications]
    observation_rows: list[dict[str, Any]] = []
    for publication_id in sorted(observations_by_pub):
        for obs in sorted(observations_by_pub[publication_id], key=lambda o: o.observation_id):
            observation_rows.append(obs.to_row())

    changes = derive_changes(publications, observations_by_pub)
    changes.sort(key=lambda r: r["change_id"])
    events = derive_graduation_events(observations_by_pub)
    events.sort(key=lambda r: r["event_id"])
    intervals = derive_intervals(publications, observations_by_pub)
    intervals.sort(key=lambda r: r["interval_id"])

    pass_publications = [p for p in publications if p.validation_status == "PASS"]
    if pass_publications:
        start = (2023, 3)
        latest_id = max(p.publication_id for p in publications)
        end = tuple(int(x) for x in latest_id.split("-"))  # type: ignore[assignment]
        coverage_rows = coverage_to_rows(build_coverage(publications, start=start, end=end))
    else:
        coverage_rows = []

    reconciliation_rows: list[dict[str, Any]] = []
    for publication in pass_publications:
        provider_csv = provider_info_normalized_csv(publication.publication_id)
        if provider_csv is None:
            continue
        provider_status = load_provider_info_ccn_status(provider_csv)
        reconciliation_rows.extend(
            reconcile(publication.publication_id, observations_by_pub[publication.publication_id], provider_status)
        )
    reconciliation_rows.sort(key=lambda r: (r["publication_id"], r["ccn"]))

    file_hashes: dict[str, str] = {}
    file_hashes["publications.csv"] = _write_csv(target / "publications.csv", PUBLICATION_FIELDS, publication_rows)
    file_hashes["observations.csv"] = _write_csv(target / "observations.csv", OBSERVATION_FIELDS, observation_rows)
    file_hashes["derived_changes.csv"] = _write_csv(
        target / "derived_changes.csv",
        ["change_id", "ccn", "from_publication_id", "to_publication_id", "change_kind", "from_categories", "to_categories", "gapless", "derived_from_observation_ids", "asserted_date", "method"],
        changes,
    )
    file_hashes["derived_graduation_events.csv"] = _write_csv(
        target / "derived_graduation_events.csv",
        ["event_id", "ccn", "event_kind", "event_date", "event_date_raw", "publication_id", "observation_id"],
        events,
    )
    file_hashes["derived_intervals.csv"] = _write_csv(
        target / "derived_intervals.csv",
        ["interval_id", "ccn", "normalized_category", "start_publication_id", "end_publication_id", "publication_ids", "months_counter_consistency", "derived_from_observation_ids"],
        intervals,
    )
    file_hashes["publication_coverage.csv"] = _write_csv(
        target / "publication_coverage.csv",
        ["year_month", "present", "publication_id", "validation_status", "gap_before"],
        coverage_rows,
    )
    file_hashes["reconciliation_provider_info.csv"] = _write_csv(
        target / "reconciliation_provider_info.csv", RECONCILIATION_FIELDS, reconciliation_rows
    )

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "era_id": ERA_3B,
        "parser_version": PARSER_VERSION,
        "publication_count": len(publications),
        "publication_pass_count": len(pass_publications),
        "publication_fail_count": len(publications) - len(pass_publications),
        "observation_count": len(observation_rows),
        "derived_change_count": len(changes),
        "derived_graduation_event_count": len(events),
        "derived_interval_count": len(intervals),
        "reconciliation_row_count": len(reconciliation_rows),
        "publication_ids": [p.publication_id for p in publications],
        "failed_publications": [
            {"publication_id": p.publication_id, "source_filename": p.source_filename, "errors": p.validation_errors}
            for p in publications
            if p.validation_status != "PASS"
        ],
        "file_sha256": file_hashes,
    }
    (target / "build_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = build_canonical_dataset()
    print(json.dumps(result, indent=2, sort_keys=True))
