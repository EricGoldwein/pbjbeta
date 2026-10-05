import pytest
import data_ops_app
from test_website_release_review import staged_candidate


@pytest.mark.parametrize('family', ['cms.snf_ownership_pair', 'cms.sff_pdf_list'])
@pytest.mark.parametrize('verified', [False, True])
def test_source_headline_and_cms_layer_share_candidate_evidence(tmp_path, monkeypatch, family, verified):
    _, control, _ = staged_candidate(tmp_path, family)
    if not verified:
        (tmp_path / 'state/release_checks.json').write_text('{}')
    source = control['datasets'][0]['dataset_id']
    monkeypatch.setattr(data_ops_app, 'control_panel_payload', lambda: control)
    monkeypatch.setattr(data_ops_app, 'probe_all_sources', lambda **kwargs: [])
    monkeypatch.setattr(data_ops_app, 'build_release_availability_context', lambda *args, **kwargs: {})
    monkeypatch.setattr(data_ops_app, 'build_source_operator_workflow', lambda *args, **kwargs: dict(
        operator_reference=dict(status_label='Up to date'),
        freshness_layers=[dict(key='cms_source', status_label='Needs attention')],
        next_action=dict(label='Old action'), release_availability=kwargs['release_availability']))
    context = data_ops_app._source_detail_context(source)
    expected = 'Byte verified current' if verified else 'Verification required'
    workflow = context['workflow']
    assert workflow['operator_reference']['status_label'] == expected
    assert workflow['freshness_layers'][0]['status_label'] == expected
    assert context['release_availability']['cms_byte_verified_current'] is verified
    if verified:
        assert context['release_availability']['cms_release_vintage'] == '2026-08'
        assert context['release_availability']['publisher_sha256']
    else:
        assert workflow['next_action']['label'] == 'Verify CMS source bytes'
        assert workflow['next_action']['method'] == 'post' and workflow['next_action']['wired']
