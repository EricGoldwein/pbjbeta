"""Review-only source lifecycle is enforced before any approval or ACTIVE write."""
import json

import pytest

import cms_data_ops as ops
import data_ops_app
from active_release_registry import ActiveReleaseError, promote_release
from data_ops_approval import ApprovalError
from release_control_plane import promote_candidate
from release_review_policy import _promote_permitted, assert_promotion_eligible, evaluate_governed_candidate_review


SOURCE = 'cms.survey_summary'


def test_validated_reference_has_no_activation_card():
    candidate = {'source_id': SOURCE, 'release_id': '2026-09', 'state': 'VALIDATED',
                 'validation': {'status': 'PASS'}}
    assert not _promote_permitted(candidate)
    assert evaluate_governed_candidate_review(candidate, snap=None, active_release_id=None) is None


def test_all_direct_activation_paths_fail_before_writes(tmp_path):
    tmp_path = tmp_path / 'isolated'
    tmp_path.mkdir()
    state = tmp_path / 'state'
    state.mkdir()
    payload = {'datasets': {SOURCE: {'release_id': '2026-09', 'state': 'VALIDATED',
                                    'validation': {'status': 'PASS'}}}}
    candidates = state / 'release_candidates.json'
    candidates.write_text(json.dumps(payload))
    before = candidates.read_bytes()
    with pytest.raises(ApprovalError, match='review-only'):
        assert_promotion_eligible(SOURCE, '2026-09', root=tmp_path)
    with pytest.raises(ApprovalError, match='review-only'):
        ops.approve_release_authoritative(SOURCE, '2026-09', root=tmp_path)
    with pytest.raises(ActiveReleaseError, match='review-only'):
        promote_candidate(SOURCE, root=tmp_path)
    with pytest.raises(ActiveReleaseError, match='review-only'):
        promote_release(SOURCE, '2026-09', tmp_path / 'unused.csv', validated_at='now', root=tmp_path)
    assert candidates.read_bytes() == before
    assert not (state / 'active_releases.json').exists()
    assert list(state.iterdir()) == [candidates]


def test_forged_approval_post_is_rejected(monkeypatch):
    monkeypatch.setenv('PBJ_DATA_OPS_PASSWORD', 'test-only')
    called = []
    monkeypatch.setattr(data_ops_app, 'approve_release_authoritative', lambda *a, **kw: called.append(a))
    client = data_ops_app.create_app().test_client()
    with client.session_transaction() as session:
        session['data_ops_authenticated'] = True
    response = client.post('/actions/approve', data={'source_id': SOURCE, 'release_id': '2026-09'})
    assert response.status_code == 302
    assert not called
    with client.session_transaction() as session:
        assert 'review-only' in session['_flashes'][-1][1]


@pytest.mark.parametrize('source_id', ['cms.provider_info', 'cms.pbj_nurse_staffing',
                                     'cms.health_citations', 'cms.sff_pdf_list',
                                     'cms.snf_all_owners', 'cms.snf_enrollments'])
def test_other_validated_sources_retain_activation_policy(source_id):
    assert _promote_permitted({'source_id': source_id, 'state': 'VALIDATED'})


def test_reference_not_in_release_review_html(monkeypatch):
    monkeypatch.setenv('PBJ_DATA_OPS_PASSWORD', 'test-only')
    monkeypatch.setattr(data_ops_app, 'release_review_items', lambda **kwargs: [])
    client = data_ops_app.create_app().test_client()
    with client.session_transaction() as session:
        session['data_ops_authenticated'] = True
    html = client.get('/release-review?source_id=cms.survey_summary').get_data(as_text=True)
    assert '/actions/approve' not in html
    assert 'requires no activation' in html
