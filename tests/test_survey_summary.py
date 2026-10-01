import csv
import hashlib
import io
import json

import pytest

from survey_summary import CCN, SCHEMA, SOURCE_ID, _retain, prepare_candidate, validate_csv


def raw_csv(rows=None):
    row = {c: '0' if c.startswith(('Count of ', 'Total Number')) else '' for c in SCHEMA['columns']}
    row.update({CCN: '01A193', 'Inspection Cycle': '1', 'Provider Name': 'Fixture', 'Processing Date': '2026-08-01',
                'Health Survey Date': '2025-01-01', 'Fire Safety Survey Date': '2025-01-01'})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=SCHEMA['columns'])
    writer.writeheader()
    writer.writerows(rows or [row])
    return stream.getvalue().encode()


def metadata():
    return {'identifier': 'tbry-pc2d', 'title': 'Survey Summary', 'theme': ['Nursing homes including rehab services'],
            'modified': '2026-08-01', 'released': '2026-08-26',
            'distribution': [{'downloadURL': 'https://data.cms.gov/resources/test_123/NH_SurveySummary_Aug2026.csv', 'mediaType': 'text/csv'}]}


def test_key_is_cycle_not_survey_date_and_alpha_ccns_are_preserved():
    row = next(csv.DictReader(io.StringIO(raw_csv().decode())))
    second = {**row, 'Inspection Cycle': '2'}
    receipt = validate_csv(raw_csv([row, second]), modified='2026-08-01')
    assert receipt['status'] == 'PASS'
    assert receipt['duplicate_key_rows'] == 0
    assert receipt['ccn_health_survey_date_duplicate_rows'] == 1
    assert receipt['foreign_keys'] == []
    assert validate_csv(raw_csv([row, row]), modified='2026-08-01')['errors']['duplicate_key'] == 1


@pytest.mark.parametrize('column,value,error', [(CCN, '1193', 'ccn'), ('Inspection Cycle', '4', 'cycle'),
    ('Health Survey Date', '2027-01-01', 'date_range'), ('Processing Date', '2026-07-01', 'processing_date'),
    ('Total Number of Health Deficiencies', '-1', 'count')])
def test_invalid_source_values_fail_closed(column, value, error):
    row = next(csv.DictReader(io.StringIO(raw_csv().decode())))
    row[column] = value
    assert error in validate_csv(raw_csv([row]), modified='2026-08-01')['errors']


def test_schema_drift_and_empty_file_fail():
    assert validate_csv(raw_csv().replace(b'Provider Name', b'Unexpected Name'), modified='2026-08-01')['status'] == 'FAIL'
    assert validate_csv(b'', modified='2026-08-01')['status'] == 'FAIL'


def test_fire_missing_values_require_absent_fire_survey_date():
    row = next(csv.DictReader(io.StringIO(raw_csv().decode())))
    row['Total Number of Fire Safety Deficiencies'] = ''
    assert validate_csv(raw_csv([row]), modified='2026-08-01')['status'] == 'FAIL'
    row['Fire Safety Survey Date'] = ''
    assert validate_csv(raw_csv([row]), modified='2026-08-01')['status'] == 'PASS'


def test_candidate_retains_bytes_metadata_receipt_and_never_activates(tmp_path, monkeypatch):
    monkeypatch.delenv('PBJ_ACTIVE_RELEASE_REGISTRY', raising=False)
    registry = tmp_path / 'state' / 'active_releases.json'
    registry.parent.mkdir(exist_ok=True)
    original_registry = json.dumps({'schema_version': 1, 'datasets': {'fixture': {'active_release_id': 'existing'}}}).encode()
    registry.write_bytes(original_registry)
    raw = raw_csv()
    result = prepare_candidate(root=tmp_path, fetch=lambda _: metadata(), download=lambda _: raw)
    assert result['state'] == 'VALIDATED'
    assert result['hash'] == hashlib.sha256(raw).hexdigest()
    receipt_path = result['validation']['receipt_path']
    assert json.loads(open(receipt_path).read())['sha256'] == result['hash']
    assert registry.read_bytes() == original_registry
    assert result['metadata']['review_ready'] is True
    repeated = prepare_candidate(root=tmp_path, fetch=lambda _: metadata(), download=lambda _: raw)
    assert repeated['release_id'] == result['release_id']
    assert len(list((tmp_path / 'state' / 'source_artifacts' / SOURCE_ID).glob('*/source.csv'))) == 1


def test_invalid_candidate_is_retained_but_not_review_ready(tmp_path, monkeypatch):
    monkeypatch.delenv('PBJ_ACTIVE_RELEASE_REGISTRY', raising=False)
    result = prepare_candidate(root=tmp_path, fetch=lambda _: metadata(), download=lambda _: b'bad\n')
    assert result['state'] == 'FAILED'
    assert result['metadata']['review_ready'] is False


def test_immutable_artifacts_refuse_overwrite(tmp_path):
    path = tmp_path / 'source.csv'
    _retain(path, b'first')
    _retain(path, b'first')
    with pytest.raises(ValueError, match='Immutable'):
        _retain(path, b'second')


def test_survey_probe_uses_candidate_validation_not_unmodeled_status(tmp_path, monkeypatch):
    monkeypatch.setenv('PBJ_ACTIVE_RELEASE_REGISTRY', str(tmp_path / 'state' / 'active_releases.json'))
    prepare_candidate(root=tmp_path, fetch=lambda _: metadata(), download=lambda _: raw_csv())
    from cms_data_ops import probe_source
    snapshot = probe_source(SOURCE_ID, root=tmp_path, check_cms=False)
    assert snapshot.status == 'READY_FOR_HANDOFF'
    assert snapshot.structural_status == 'PASS'
    assert snapshot.local_raw_present
