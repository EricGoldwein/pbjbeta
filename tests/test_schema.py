from sff_history.schema import is_valid_ccn


def test_alphanumeric_ccn_is_valid():
    # Real CCNs confirmed in the archive audit's July/August 2026 reconciliation
    # (SFF_CURRENT_RECONCILIATION.md S2): CMS's CCN space is not purely numeric.
    for ccn in ("15E064", "17E210", "04E262", "34A002"):
        assert is_valid_ccn(ccn)


def test_purely_numeric_ccn_is_valid():
    assert is_valid_ccn("015009")


def test_wrong_length_ccn_is_invalid():
    assert not is_valid_ccn("15E06")
    assert not is_valid_ccn("15E0645")


def test_lowercase_ccn_is_invalid_unnormalized():
    # CCNs are opaque strings compared exactly; this module never uppercases
    # or otherwise "cleans" a CCN on the caller's behalf.
    assert not is_valid_ccn("15e064")
