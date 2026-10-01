from source_family_inventory import source_family_inventory


def test_unintegrated_families_do_not_claim_active_lifecycle():
    rows = {r['source_id']: r for r in source_family_inventory({}, observations={})}
    for key in ('penalties', 'hcris', 'npi_nppes'):
        assert rows[key]['governance_status'] == 'NO_GOVERNED_RELEASE'
        assert rows[key]['implementation_status'] == 'NO_REGISTERED_ADAPTER'
        assert rows[key]['health'] == 'UNKNOWN'
    for key in ('penalties', 'hcris', 'npi_nppes'):
        assert rows[key]['workflow_source_id'] is None
        assert not any(k in rows[key] for k in ('active', 'active_release_id'))


def test_inventory_covers_required_sources_and_uses_real_release_health():
    rows = {r['source_id']: r for r in source_family_inventory({'datasets': [
        {'dataset_id': 'cms.snf_all_owners', 'active': {'active_release_id': '2026-07-31'}, 'health': 'CURRENT'}]})}
    assert rows['cms.snf_all_owners']['evidence'] == '2026-07-31'
    assert rows['cms.snf_all_owners']['health'] == 'CURRENT'
    assert {'cms.provider_info', 'cms.health_citations', 'cms.sff_pdf_list', 'cms.snf_enrollments',
            'cms.pbj_nurse_staffing', 'cms.pbj_non_nurse_staffing', 'cms.pbj_employee_ein_detail'} <= rows.keys()
    assert rows['cms.provider_info']['evidence'] == 'No governed release evidence'


def test_candidate_receipt_and_health_drive_inventory_not_static_maturity():
    control = {'datasets': [{'dataset_id': 'cms.survey_summary', 'health': 'MISSING',
                            'pending': {'state': 'VALIDATED', 'release_id': '2026-08-sha', 'validation': {'status': 'PASS'}}}]}
    rows = {r['source_id']: r for r in source_family_inventory(control, observations={})}
    row = rows['cms.survey_summary']
    assert (row['governance_status'], row['implementation_status'], row['health']) == ('GOVERNED', 'VALIDATED', 'VALIDATION_PASS')
    assert row['evidence'] == '2026-08-sha'
    assert 'explicit approval' in row['next_action']
    control['datasets'][0]['pending']['validation']['status'] = 'FAIL'
    row = next(r for r in source_family_inventory(control, observations={}) if r['source_id'] == 'cms.survey_summary')
    assert row['health'] == 'VALIDATION_FAIL'
    assert row['next_action'] == 'Inspect failed validation receipt'


def test_graph_placeholder_and_catalog_do_not_imply_governance():
    rows = {r['source_id']: r for r in source_family_inventory(
        {'datasets': [{'dataset_id': 'cms.provider_info', 'health': 'MISSING'}]},
        catalog={'datasets': [{'stable_id': 'g6vv-u9sr', 'released': '2026-08-26'}]}, observations={})}
    assert rows['cms.provider_info']['governance_status'] == 'REGISTERED_ONLY'
    assert 'catalog observation only' in rows['penalties']['evidence']
    assert rows['penalties']['governance_status'] == 'NO_GOVERNED_RELEASE'
