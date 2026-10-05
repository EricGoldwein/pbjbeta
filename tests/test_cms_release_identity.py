"""Shared CMS identity regression: read-only assessment, no date equality shortcut."""
import hashlib
import json
from pathlib import Path

import pytest

from cms_release_identity import assess_raw_identity, active_raw_record, RAW_PRIMARY_SOURCES

SOURCES = ["cms.provider_info", "cms.pbj_nurse_staffing", "cms.pbj_non_nurse_staffing",
           "cms.health_citations", "cms.snf_all_owners", "cms.snf_enrollments", "cms.sff_pdf_list"]
CATALOG_FIXTURE=json.loads((Path(__file__).parent/'fixtures/cms_nh_release_metadata_20261005.json').read_text())['datasets']


@pytest.mark.parametrize('metadata',CATALOG_FIXTURE,ids=lambda d:d['identifier'])
def test_every_official_catalog_member_inherits_separate_publication_metadata(metadata):
    from cms_nh_catalog import normalize_dataset, compare_observation
    current=normalize_dataset(metadata,metadata['identifier'])
    assert current['cms_release_vintage']==metadata['released'][:7]
    assert current['processing_modified_date']==metadata['modified']
    assert current['snapshot_date'] is None
    assert current['resources'][0]['resource_id']
    assert current['resources'][0]['version']
    assert current['byte_check']=='NOT_PERFORMED'
    assert compare_observation(current,dict(current),'2026-10-05')[0]=='METADATA_UNCHANGED'


def fixture(root, source):
    raw = root / 'immutable.csv'; raw.write_bytes(b'raw official bytes\n')
    normalized = root / 'normalized.csv'; normalized.write_bytes(b'normalized bytes\n')
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    path = raw if source in RAW_PRIMARY_SOURCES else normalized
    active_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {'cms_release_vintage': '2026-08'}
    if source == 'cms.sff_pdf_list':
        metadata.update(source_pdf_uri=raw.as_uri(), source_pdf_hash=sha)
    elif source not in RAW_PRIMARY_SOURCES:
        metadata['immutable_raw_source'] = {'source_uri': raw.as_uri(), 'hash': sha,
                                            'normalized_sha256': active_sha}
    active = {'active_release_id': '2026-07-31', 'hash': active_sha,
              'source_uri': path.as_uri(), 'metadata': metadata}
    current = {'release_id': '2026-07-31', 'cms_release_vintage': '2026-08',
               'snapshot_date': '2026-07-31', 'publisher_url': 'https://data.cms.gov/current.csv',
               'publisher_filename': 'same_2026.07.31.csv', 'publisher_file_uuid': 'same-id',
               'publisher_modified': '2026-08-17'}
    return active, current, raw


@pytest.mark.parametrize('source', SOURCES)
@pytest.mark.parametrize('case,status', [('identical','CURRENT'), ('same_date_revision','REVISED'), ('new_vintage','NEWER')])
def test_shared_rule_for_every_operational_adapter(tmp_path, source, case, status):
    active, current, raw = fixture(tmp_path, source)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    if case == 'new_vintage': current['cms_release_vintage'] = '2026-09'
    payload = raw.read_bytes() if case == 'identical' else b'changed official bytes\n'
    result = assess_raw_identity(source, active=active, current=current, fetch_bytes=lambda _: payload)
    assert result['status'] == status
    assert result['cms_release_vintage'] == current['cms_release_vintage']
    assert result['snapshot_date'] == '2026-07-31'
    assert result['new_release_available'] == (status != 'CURRENT')
    assert result['publisher_sha256'] == hashlib.sha256(payload).hexdigest()
    assert result['active_raw_sha256'] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('source', ['cms.provider_info','cms.pbj_nurse_staffing','cms.pbj_non_nurse_staffing'])
def test_transformed_active_without_bound_raw_is_unknown_even_matching_ids(tmp_path, source):
    active, current, raw = fixture(tmp_path, source)
    active['metadata'].pop('immutable_raw_source')
    result = assess_raw_identity(source, active=active, current=current, fetch_bytes=lambda _: raw.read_bytes())
    assert result['status'] == 'UNKNOWN'
    assert result['new_release_available'] is None


def test_provider_existing_manifest_binds_raw_to_normalized_active(tmp_path):
    active, current, raw = fixture(tmp_path, 'cms.provider_info')
    active['metadata'].pop('immutable_raw_source')
    manifest = tmp_path / 'release_manifest.json'
    manifest.write_text(json.dumps({'source_members': [{'source_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
        'normalized_outputs': [{'path': str(tmp_path/'normalized.csv'), 'sha256':active['hash']},
                               {'path':str(raw), 'sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}]}]}))
    active['metadata']['validation_evidence'] = str(manifest)
    assert active_raw_record('cms.provider_info', active)['source_uri'] == raw.as_uri()
    result = assess_raw_identity('cms.provider_info', active=active, current=current, fetch_bytes=lambda _: raw.read_bytes())
    assert result['status'] == 'CURRENT'
    active['hash'] = 'wrong'
    assert active_raw_record('cms.provider_info', active) is None


def test_byte_fetch_failure_and_tampered_raw_never_current(tmp_path):
    active, current, raw = fixture(tmp_path, 'cms.provider_info')
    def fail(_): raise OSError('publisher unavailable')
    assert assess_raw_identity('cms.provider_info', active=active, current=current, fetch_bytes=fail)['status'] == 'ERROR'
    raw.write_bytes(b'tampered')
    result=assess_raw_identity('cms.provider_info', active=active, current=current, fetch_bytes=lambda _: raw.read_bytes())
    assert result['status'] == 'UNKNOWN'


def test_resource_version_token_is_preserved_without_inventing_a_date(tmp_path):
    active,current,raw=fixture(tmp_path,'cms.provider_info')
    current['publisher_url']='https://data.cms.gov/provider-data/sites/default/files/resources/resource_1786724150/same.csv'
    result=assess_raw_identity('cms.provider_info',active=active,current=current,fetch_bytes=lambda _:raw.read_bytes())
    assert result['status']=='CURRENT'
    assert result['publisher_resource_id']=='resource'
    assert result['publisher_resource_version']=='1786724150'
    assert result['cms_release_vintage']=='2026-08'
    assert 'acquired_at' not in result


def test_metadata_catalog_does_not_claim_current_and_publication_is_separate():
    from cms_nh_catalog import normalize_dataset, compare_observation, coverage_for, THEME
    current=normalize_dataset({'identifier':'tbry-pc2d','theme':[THEME],'released':'2026-08-26',
        'modified':'2026-07-31','distribution':[{'downloadURL':'https://data.cms.gov/raw_2026.07.31.csv'}]}, 'tbry-pc2d')
    assert current['cms_release_vintage']=='2026-08'
    assert current['processing_modified_date']=='2026-07-31'
    assert compare_observation(current,dict(current),'2026-10-05')[0]=='METADATA_UNCHANGED'
    revised={**current,'publisher_sha256':'remote'}
    prior={**current,'publisher_sha256':'previous'}
    assert compare_observation(revised,prior,'2026-10-05')[0]=='REVISED'
    newer={**current,'released':'2026-09-26','cms_release_vintage':'2026-09'}
    assert compare_observation(newer,current,'2026-10-05')[0]=='NEWER'
    assert coverage_for('tbry-pc2d')['active_lifecycle']=='Review only; activation and publication unavailable'


def test_same_period_revision_overrides_theme_and_pending_date_suppression(tmp_path):
    from cms_data_ops import build_release_availability_context
    active,current,raw=fixture(tmp_path,'cms.provider_info')
    check=assess_raw_identity('cms.provider_info',active=active,current=current,fetch_bytes=lambda _:b'revised')
    context=build_release_availability_context('cms.provider_info',root=tmp_path,
        control_row={'active':active,'pending':{'release_id':active['active_release_id'],'state':'DETECTED'}},check_row=check)
    assert context['new_release_available'] is True
    assert context['cms_byte_verified_current'] is False
    assert context['availability_summary']=='CMS revision available'
    assert context['cms_release_vintage']=='2026-08'
    assert context['snapshot_date']=='2026-07-31'


def test_old_date_only_current_check_is_not_rendered_as_verified(tmp_path):
    from cms_data_ops import build_release_availability_context
    active,current,raw=fixture(tmp_path,'cms.provider_info')
    context=build_release_availability_context('cms.provider_info',root=tmp_path,
        control_row={'active':active},check_row={'status':'CURRENT','release_id':active['active_release_id'],'new_release_available':False})
    assert context['cms_byte_verified_current'] is False
    assert context['availability_summary']=='CMS bytes not verified'
    assert context['unchanged_in_latest_publication'] is False


def test_selected_local_active_does_not_create_publisher_current_badge():
    from cms_data_ops import overlay_control_plane_on_snapshot
    control={'datasets':[{'dataset_id':'cms.provider_info','active':{'active_release_id':'2026-08','status':'ACTIVE'},'pending':None}]}
    snapshot=overlay_control_plane_on_snapshot({'source_id':'cms.provider_info','status':'UNKNOWN'},control)
    assert snapshot['status']=='ACTIVE'


@pytest.mark.parametrize('source,check_fn', [('cms.provider_info','check_provider_info_cms'),
    ('cms.pbj_nurse_staffing','check_nurse_cms'), ('cms.health_citations','check_health_citations_cms')])
@pytest.mark.parametrize('changed', [False, True])
def test_production_handlers_forward_byte_evidence_without_acquiring(tmp_path, monkeypatch, source, check_fn, changed):
    from release_check import production_handlers
    active,current,raw=fixture(tmp_path,source)
    evidence=assess_raw_identity(source,active=active,current=current,
        fetch_bytes=lambda _:b'changed' if changed else raw.read_bytes())
    module='health_citations_acquire' if source=='cms.health_citations' else 'cms_data_ops'
    monkeypatch.setattr(module+'.'+check_fn, lambda **_: {'cms':{},'release_identity':evidence})
    def forbidden(**_): raise AssertionError('acquisition is forbidden in a read-only check')
    monkeypatch.setattr('cms_data_ops.acquire_provider_info',forbidden)
    monkeypatch.setattr('cms_data_ops.acquire_nurse',forbidden)
    result=production_handlers()[source](False)
    assert result['status']==('REVISED' if changed else 'CURRENT')
    assert result['publisher_sha256']==evidence['publisher_sha256']


def test_pbj_adapter_records_version_date_without_calling_it_release_vintage(tmp_path):
    from generic_cms_csv import CsvFeed, detect
    feed=CsvFeed('cms.pbj_non_nurse_staffing','product',r'PBJ.*\.csv$',tmp_path,(),False,
                 cms_product_path='/quality-of-care/pbj',cms_product_name='PBJ',version_date_basis='reporting_period')
    def fetch(url):
        if '/slug?' in url: return {'data':{'uuid':'product','current_dataset':{'uuid':'version'}}}
        if '/jsonapi/' in url: return {'data':[{'id':'version','attributes':{'field_dataset_version':'2026-01-01','field_last_updated_date':'2026-07-29'}}]}
        assert '/dataset/version/resources' in url
        return {'data':[{'type':'Primary','file_name':'PBJ_dailynonnursestaffing_CY2026Q1.csv','file_url':'https://data.cms.gov/current.csv','file_uuid':'file'}]}
    found=detect(feed,fetch_json=fetch)
    assert found['cms_release_vintage'] is None
    assert found['snapshot_date']=='CY2026Q1'
    assert found['dataset_version_label']=='2026-01-01'
    assert found['dataset_version_modified']=='2026-07-29'
    assert found['dataset_version_id']=='version'


@pytest.mark.parametrize('changed',[False,True])
def test_sff_pdf_same_posting_date_byte_check_is_read_only(tmp_path, monkeypatch, changed):
    import sff_release
    from active_release_registry import promote_release
    from release_control_plane import load_candidates
    active,current,raw=fixture(tmp_path,'cms.sff_pdf_list')
    registry=tmp_path/'state/active_releases.json'
    promote_release('cms.sff_pdf_list','2026-08',tmp_path/'normalized.csv',validated_at='test',path=registry,metadata=active['metadata'])
    payload=b'changed PDF' if changed else raw.read_bytes()
    monkeypatch.setattr(sff_release,'discover_latest_cms_sff_posting',lambda **_: {
        'release_id':'2026-08','posting_label':'August 2026','source_url':'https://www.cms.gov/current.pdf',
        'url_release_id':'2026-08','publisher_sha256':hashlib.sha256(payload).hexdigest(),
        'posting_date_verified':True,'candidates':[]})
    before=registry.read_bytes()
    result=sff_release.check_sff_cms(root=tmp_path,record_detection=False)
    assert result['status']==('REVISED' if changed else 'CURRENT')
    assert registry.read_bytes()==before
    assert load_candidates(tmp_path)['datasets']=={}


def test_single_check_record_does_not_change_active_or_candidates(tmp_path):
    from release_check import record_check_result, load_check_state
    active,current,raw=fixture(tmp_path,'cms.provider_info')
    evidence=assess_raw_identity('cms.provider_info',active=active,current=current,fetch_bytes=lambda _:raw.read_bytes())
    protected={p:p.read_bytes() for p in (tmp_path/'state').iterdir()}
    record_check_result('cms.provider_info',evidence,root=tmp_path)
    assert load_check_state(tmp_path)['datasets'][0]['status']=='CURRENT'
    assert all(p.read_bytes()==b for p,b in protected.items())


def test_survey_same_metadata_changed_bytes_retains_review_only_candidates(tmp_path,monkeypatch):
    import csv, io
    from survey_summary import SCHEMA, CCN, prepare_candidate
    # Keep Windows immutable metadata/receipt paths below MAX_PATH.
    root=tmp_path.parent/'survey'
    root.mkdir(exist_ok=True)
    (root/'state').mkdir(exist_ok=True)
    short_registry=root/'state/active_releases.json'
    short_registry.write_bytes((tmp_path/'state/active_releases.json').read_bytes())
    monkeypatch.setenv('PBJ_ACTIVE_RELEASE_REGISTRY',str(short_registry))
    metadata={'identifier':'tbry-pc2d','title':'Survey Summary','theme':['Nursing homes including rehab services'],
              'modified':'2026-08-01','released':'2026-08-26',
              'distribution':[{'downloadURL':'https://data.cms.gov/NH_SurveySummary_Aug2026.csv'}]}
    row={c:'0' if c.startswith(('Count of ','Total Number')) else '' for c in SCHEMA['columns']}
    row.update({CCN:'01A193','Inspection Cycle':'1','Provider Name':'Fixture','Processing Date':'2026-08-01',
                'Health Survey Date':'2025-01-01','Fire Safety Survey Date':'2025-01-01'})
    def payload(name):
        out=io.StringIO(); writer=csv.DictWriter(out,fieldnames=SCHEMA['columns']);writer.writeheader()
        writer.writerow({**row,'Provider Name':name});return out.getvalue().encode()
    registry=tmp_path/'state/active_releases.json';before=registry.read_bytes()
    first=prepare_candidate(root=root,fetch=lambda _:metadata,download=lambda _:payload('First'))
    repeat=prepare_candidate(root=root,fetch=lambda _:metadata,download=lambda _:payload('First'))
    revised=prepare_candidate(root=root,fetch=lambda _:metadata,download=lambda _:payload('Revision'))
    assert first['state']==repeat['state']==revised['state']=='VALIDATED'
    assert first['release_id']==repeat['release_id']
    assert first['release_id']!=revised['release_id']
    assert first['hash']!=revised['hash']
    assert revised['metadata']['cms_release_vintage']=='2026-08'
    assert revised['metadata']['processing_modified_date']=='2026-08-01'
    assert revised['metadata']['acquired_at']
    assert registry.read_bytes()==before
    assert short_registry.read_bytes()==before
    assert not (root/'state/pbj320_stages').exists()
    assert not (root/'state/pbj320_publication_receipts').exists()
