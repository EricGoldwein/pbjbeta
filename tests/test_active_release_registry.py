from pathlib import Path

import pytest

from active_release_registry import ActiveReleaseError, get_active_release, promote_release


def test_promote_validated_source(tmp_path: Path) -> None:
    source, registry = tmp_path / "source.csv", tmp_path / "active.json"
    source.write_text("CCN\n335581\n", encoding="utf-8")
    promote_release("cms.provider_info", "2026-08", source, validated_at="2026-08-26T00:00:00+00:00", path=registry)
    assert get_active_release("cms.provider_info", registry)["active_release_id"] == "2026-08"


def test_unvalidated_and_bad_hash_fail(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("x\n", encoding="utf-8")
    with pytest.raises(ActiveReleaseError, match="validated ACTIVE"):
        promote_release("cms.provider_info", "candidate", source, validated_at=None, path=tmp_path / "r.json")
    with pytest.raises(ActiveReleaseError, match="hash"):
        promote_release("cms.provider_info", "candidate", source, validated_at="now", source_hash="0" * 64, path=tmp_path / "r.json")


def test_promoted_source_set_is_hash_validated_and_role_addressable(tmp_path: Path) -> None:
    primary = tmp_path / "ProviderInfoNorm.csv"
    ownership = tmp_path / "NH_Ownership.csv"
    primary.write_text("ccn\n335581\n", encoding="utf-8")
    ownership.write_text("CMS Certification Number (CCN)\n335581\n", encoding="utf-8")
    registry = tmp_path / "active.json"
    record = promote_release(
        "cms.provider_info", "2026-08", primary,
        validated_at="2026-08-26T00:00:00Z", path=registry,
        metadata={"source_set": [{"role": "nh_ownership", "source_path": str(ownership)}]},
    )
    member = record["metadata"]["source_set"][0]
    assert member["role"] == "nh_ownership"
    assert member["hash"]
    assert "source_path" not in member
