from __future__ import annotations

import json
from pathlib import Path


def _write_registry(path: Path, *, nurse_hash: str = "nurse-hash") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "cms.pbj_nurse_staffing": {
                        "dataset_id": "cms.pbj_nurse_staffing",
                        "active_release_id": "CY2026Q1",
                        "hash": nurse_hash,
                        "status": "ACTIVE",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_rebuilt_candidate_records_release_hash_timestamp_and_output_hash(
    tmp_path: Path, monkeypatch
) -> None:
    import active_release_registry as arr
    import derived_provenance as provenance
    import release_control_plane as rcp

    registry = tmp_path / "state" / "active_releases.json"
    _write_registry(registry)
    artifact = tmp_path / "national_quarterly_metrics.csv"
    artifact.write_text("CY_Qtr,value\n2026Q1,1\n", encoding="utf-8")
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: registry)
    monkeypatch.setattr(provenance, "registry_path", lambda _root=None: registry)
    monkeypatch.setattr(rcp, "registry_path", lambda _root=None: registry)

    candidate = provenance.record_validated_derived_candidate(
        "pbj.benchmarks.national",
        artifact,
        builder="test-builder",
        root=tmp_path,
    )

    assert candidate["state"] == "VALIDATED"
    metadata = candidate["metadata"]
    assert metadata["built_at"]
    assert metadata["output_artifact_hash"]
    assert metadata["upstream_provenance"]["cms.pbj_nurse_staffing"] == {
        "release_id": "CY2026Q1",
        "source_hash": "nurse-hash",
    }


def test_derived_state_becomes_stale_when_active_hash_changes(
    tmp_path: Path, monkeypatch
) -> None:
    import active_release_registry as arr
    import release_check

    registry = tmp_path / "state" / "active_releases.json"
    output_hash = "derived-hash"
    registry.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "datasets": {
            "cms.pbj_nurse_staffing": {
                "active_release_id": "CY2026Q1",
                "hash": "nurse-hash-a",
            },
            "pbj.benchmarks.national": {
                "active_release_id": "sha256:derived",
                "hash": output_hash,
                "metadata": {
                    "output_artifact_hash": output_hash,
                    "upstream_provenance": {
                        "cms.pbj_nurse_staffing": {
                            "release_id": "CY2026Q1",
                            "source_hash": "nurse-hash-a",
                        }
                    },
                },
            },
        },
    }
    registry.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: registry)
    monkeypatch.setattr(release_check, "registry_path", lambda _root=None: registry)

    current = release_check.derived_state(
        "pbj.benchmarks.national", ("cms.pbj_nurse_staffing",), root=tmp_path
    )
    assert current["status"] == "CURRENT"

    payload["datasets"]["cms.pbj_nurse_staffing"]["hash"] = "nurse-hash-b"
    registry.write_text(json.dumps(payload), encoding="utf-8")
    stale = release_check.derived_state(
        "pbj.benchmarks.national", ("cms.pbj_nurse_staffing",), root=tmp_path
    )
    assert stale["status"] == "STALE"
    assert stale["new_release_available"] is True


def test_release_only_provenance_is_unknown_without_hash(
    tmp_path: Path, monkeypatch
) -> None:
    import active_release_registry as arr
    import release_check

    registry = tmp_path / "state" / "active_releases.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "cms.pbj_nurse_staffing": {
                        "active_release_id": "CY2026Q1",
                        "hash": "nurse-hash",
                    },
                    "pbj.benchmarks.national": {
                        "active_release_id": "sha256:derived",
                        "hash": "derived-hash",
                        "metadata": {
                            "upstream_releases": {
                                "cms.pbj_nurse_staffing": "CY2026Q1"
                            }
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: registry)
    monkeypatch.setattr(release_check, "registry_path", lambda _root=None: registry)
    result = release_check.derived_state(
        "pbj.benchmarks.national", ("cms.pbj_nurse_staffing",), root=tmp_path
    )
    assert result["status"] == "UNKNOWN"
    assert result["provenance_missing"] is True


def test_unreadable_provenance_is_unknown(tmp_path: Path, monkeypatch) -> None:
    import active_release_registry as arr
    import release_check

    registry = tmp_path / "state" / "active_releases.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "cms.pbj_nurse_staffing": {
                        "active_release_id": "CY2026Q1",
                        "hash": "nurse-hash",
                    },
                    "pbj.benchmarks.national": {
                        "active_release_id": "sha256:derived",
                        "hash": "derived-hash",
                        "metadata": {
                            "output_artifact_hash": "derived-hash",
                            "upstream_provenance": "not-an-object",
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: registry)
    monkeypatch.setattr(release_check, "registry_path", lambda _root=None: registry)

    result = release_check.derived_state(
        "pbj.benchmarks.national", ("cms.pbj_nurse_staffing",), root=tmp_path
    )
    assert result["status"] == "UNKNOWN"
