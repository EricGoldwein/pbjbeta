import json
from source_evidence import adjacent_source_observations


def test_presence_and_receipt_are_observations_not_active_or_merge_claims(tmp_path):
    penalties = tmp_path / 'Penalties'
    penalties.mkdir()
    (penalties / 'NH_Penalties_Aug2026.csv').write_text('fixture')
    pilot = tmp_path / 'cms' / 'hcris' / 'pilot_outputs' / 'contract' / 'release'
    pilot.mkdir(parents=True)
    (pilot / 'validation_receipt.json').write_text(json.dumps({'pilot_id': 'pilot', 'production_status': 'NON_PRODUCTION_RAW_CONTRACT_PILOT'}))
    result = adjacent_source_observations(tmp_path)
    assert result['penalties']['health'] == 'RAW_PRESENT_UNVALIDATED'
    assert result['hcris']['health'] == 'NON_PRODUCTION_RAW_CONTRACT_PILOT'
    assert result['npi_nppes']['health'] == 'NOT_OBSERVED_IN_RUNTIME'
    assert all('active' not in row for row in result.values())
