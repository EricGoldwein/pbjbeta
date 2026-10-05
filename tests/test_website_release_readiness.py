import hashlib
import json

import pytest

from test_website_release_review import staged_candidate
from website_release_readiness import FAMILIES, family_readiness, verify_family_sources, observe_required_source, evidence_path


def observations(root):
    return {r['dataset_id']: r for r in json.loads((root / 'state/release_checks.json').read_text())['datasets']}


@pytest.mark.parametrize('family', FAMILIES)
def test_only_verification_evidence_is_written_and_bound_to_exact_candidate(tmp_path, family):
    release, control, _ = staged_candidate(tmp_path, family)
    good = observations(tmp_path)
    (tmp_path / 'state/release_checks.json').write_text('{}')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert family_readiness(family, control)['next_action'] == 'verify_sources'
    calls = []
    def observer(req, **kwargs):
        calls.append(req.source_id)
        return good[req.source_id]
    result = verify_family_sources(family, release, control, observer=observer)
    assert result['publication_ready']
    assert calls == [req.source_id for req in FAMILIES[family].required_source_evidence]
    assert all(p.read_bytes() == value for p, value in before.items())
    record = json.loads(evidence_path(family, release).read_text())
    assert record['stage_manifest_sha256'] == result['stage_sha']
    assert all(s['comparison_result'] == 'MATCH' for s in record['sources'].values())
    for source in record['sources'].values():
        assert source['publisher_checked_at'] and source['publisher_url'] and source['staged_sha256']


@pytest.mark.parametrize('failure', ['exception', 'revised', 'metadata'])
def test_one_unverifiable_member_or_same_vintage_revision_fails_closed(tmp_path, failure):
    release, control, _ = staged_candidate(tmp_path, 'cms.snf_ownership_pair')
    good = observations(tmp_path)
    def observer(req, **kwargs):
        evidence = dict(good[req.source_id])
        if req.role == 'Enrollments':
            if failure == 'exception':
                raise OSError('CMS unavailable')
            if failure == 'revised':
                evidence.update(status='REVISED', publisher_sha256='changed-same-vintage')
            if failure == 'metadata':
                evidence.pop('publisher_sha256')
        return evidence
    result = verify_family_sources('cms.snf_ownership_pair', release, control, observer=observer)
    assert result['verified_count'] == 1 and not result['publication_ready']
    assert result['next_action'] == 'verify_sources'
    if failure == 'revised':
        assert result['source_status'] == 'Publisher bytes differ'
        assert result['source_evidence'][1]['vintage'] == '2026-08'


def receipt(root, family, release, path, state):
    record = dict(stage_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  commit_sha='candidate-commit', committed_at='observed')
    if state in {'pushed', 'deployed', 'verified', 'old', 'flag_only'}:
        record.update(push_succeeded=True, push_timestamp='observed')
    if state == 'old':
        record['stage_manifest_sha256'] = 'other-candidate'
    if state == 'deployed':
        record['deployment_observation'] = dict(commit_sha='candidate-commit', observed_at='observed', status='SUCCESS')
    if state in {'verified', 'flag_only'}:
        record.update(production_verified=True, production_verified_at='observed')
    if state == 'verified':
        record['production_verification_checks'] = [dict(expected=hashlib.sha256(b'{}').hexdigest(), actual=hashlib.sha256(b'{}').hexdigest(), result='PASS', target='https://www.pbj320.com/fixture.json', verification_timestamp='observed')]
    destination = root / 'state/pbj320_publications' / family / f'{release}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record))


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('state,label,action', [
    ('staged', 'Not published', 'review'),
    ('committed', 'Committed locally, not pushed', 'review'),
    ('pushed', 'Pushed; deployment not verified', 'review'),
    ('deployed', 'Deployment observed; provenance not verified', 'review'),
    ('verified', 'Live deployment verified', 'none'),
    ('old', 'Not published', 'review'),
    ('flag_only', 'Pushed; deployment not verified', 'review'),
])
def test_publication_layers_never_collapse(tmp_path, family, state, label, action):
    release, control, path = staged_candidate(tmp_path, family)
    if state != 'staged':
        receipt(tmp_path, family, release, path, state)
    row = family_readiness(family, control)
    assert row['publication']['label'] == label
    assert row['next_action'] == action
    if state == 'old':
        assert row['publication']['receipt_notice']
        assert not row['publication']['pushed'] and not row['publication']['production_verified']


def test_stage_flags_do_not_establish_commit_push_or_deployment(tmp_path):
    release, control, path = staged_candidate(tmp_path, 'cms.sff_pdf_list')
    manifest = json.loads(path.read_text())
    manifest['destination_layers'] = dict(canonical='CURRENT', committed='YES', pushed='YES', deployed='YES', production_verified='YES')
    path.write_text(json.dumps(manifest))
    row = family_readiness('cms.sff_pdf_list', control)
    assert row['publication']['label'] == 'Not published'
    assert not row['publication']['production_verified']


def test_tampered_cached_destination_and_changed_active_block_candidate(tmp_path):
    release, control, _ = staged_candidate(tmp_path, 'cms.sff_pdf_list')
    from pbj320_publication import stage_artifact_cache_path
    (stage_artifact_cache_path('cms.sff_pdf_list', release) / 'sff_public_json.json').write_text('tampered')
    assert not family_readiness('cms.sff_pdf_list', control)['candidate_valid']
    control['datasets'][0]['active']['hash'] = 'new-active'
    row = family_readiness('cms.sff_pdf_list', control)
    assert not row['publication_ready'] and row['verified_count'] == 0


def test_sff_observer_uses_official_current_link_and_raw_pdf_binding(tmp_path, monkeypatch):
    from active_release_registry import registry_path
    pdf = b'%PDF-1.7 /Title (SFF Updated September 2026)'
    normalized = tmp_path / 'normalized.csv'; normalized.write_bytes(b'normalized')
    raw = tmp_path / 'source.pdf'; raw.write_bytes(pdf)
    sha = hashlib.sha256(normalized.read_bytes()).hexdigest()
    raw_sha = hashlib.sha256(pdf).hexdigest()
    registry_path(tmp_path).write_text(json.dumps(dict(schema_version=1, datasets={'cms.sff_pdf_list': dict(status='ACTIVE', active_release_id='2026-09',
        hash=sha, source_uri=normalized.as_uri(), metadata=dict(source_pdf_uri=raw.as_uri(), source_pdf_hash=raw_sha))})))
    url = 'https://www.cms.gov/files/document/sff-posting-candidate-list-september-2026.pdf'
    calls = []
    def fetch(link):
        calls.append(link)
        return pdf if link == url else ('<a href="' + url + '">Current posting</a>').encode()
    requirement = FAMILIES['cms.sff_pdf_list'].required_source_evidence[0]
    result = observe_required_source(requirement, root=tmp_path, fetch_bytes=fetch)
    assert result['status'] == 'CURRENT'
    assert result['active_raw_sha256'] == raw_sha and result['active_hash'] == sha
    assert result['publisher_identity_basis'] == 'official_page_link'
    assert len(calls) == 2
    with pytest.raises(ValueError, match='no unique current posting'):
        observe_required_source(requirement, root=tmp_path, fetch_bytes=lambda _: b'No deterministic posting link')


def test_supporting_deploy_generated_artifact_is_not_a_missing_commit_cache(tmp_path):
    release, control, path = staged_candidate(tmp_path, 'cms.snf_ownership_pair')
    manifest = json.loads(path.read_text())
    manifest['artifacts'].append(dict(destination_id='owner_profile_index', publication_class='deploy_generated', path='Render generated owner index'))
    path.write_text(json.dumps(manifest))
    assert family_readiness('cms.snf_ownership_pair', control)['publication_ready']


def test_non_cms_cached_observation_cannot_satisfy_byte_gate(tmp_path):
    _, control, _ = staged_candidate(tmp_path, 'cms.snf_ownership_pair')
    checks = observations(tmp_path)
    checks['cms.snf_all_owners']['publisher_url'] = 'https://example.com/copied.csv'
    (tmp_path / 'state/release_checks.json').write_text(json.dumps(dict(datasets=list(checks.values()))))
    row = family_readiness('cms.snf_ownership_pair', control)
    assert row['verified_count'] == 1 and not row['publication_ready']
