from pathlib import Path
from cms_data_ops import _probe_chain, _chain_sort_key
from cms_source_registry import get_source
from release_source_catalog import BY_ID, UpdateMechanism
from release_check import production_handlers


def test_chain_uses_publisher_check_and_existing_generic_adapter():
    assert BY_ID['cms.chain_performance'].mechanism == UpdateMechanism.EXTERNAL_RECURRING
    handler = production_handlers()['cms.chain_performance']
    feed = handler.__defaults__[0]
    assert feed.automatic_validation is False
    assert ('Chain', 'Chain Name') in feed.required_column_groups
    assert get_source('cms.chain_performance').acquisition_implementation is not None


def test_chain_probe_includes_current_and_history_filename_patterns(tmp_path):
    own=tmp_path/'ownership'; (own/'chain_history_source').mkdir(parents=True)
    old=own/'Nursing_Home_Chain_Performance_Measures_Jul_2026.csv'
    new=own/'chain_history_source'/'2026-09-09.csv'
    old.write_text('Chain ID,Chain Name\n1,A\n');new.write_text('Chain ID,Chain Name\n1,A\n')
    assert _chain_sort_key(new) > _chain_sort_key(old)
    same_month=own/'Nursing_Home_Chain_Performance_Measures_Sep_2026.csv'
    same_month.write_text('Chain,Chain ID\nA,1\n')
    assert _chain_sort_key(new) > _chain_sort_key(same_month)
    snap=_probe_chain(get_source('cms.chain_performance'),tmp_path)
    assert snap.pbjapp_latest == 'September 2026 file date'
    assert 'ACTIVE' not in snap.status
