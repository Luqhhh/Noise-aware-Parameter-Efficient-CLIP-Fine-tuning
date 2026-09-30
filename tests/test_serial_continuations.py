"""Freeze funding decisions and serial single-checkpoint commands before DEV results."""
import copy
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('serial',Path(__file__).resolve().parents[1]/'scripts/run_serial_continuations.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def evidence(net=10,macro=.1):
    group=dict(rows=100,net=net,delta_macro_pp=macro)
    slices={name:copy.deepcopy(group) for name in ('all','tail','head','few_support')}
    pair=dict(slices=slices)
    return dict(status='development_complete',epochs=4,
                paired=dict(last_ema=copy.deepcopy(pair),ema_swa_2_4=copy.deepcopy(pair)),
                trajectory=[dict(raw=copy.deepcopy(pair),ema=copy.deepcopy(pair)) for _ in range(4)])


def test_unanimous_mature_evidence_can_fund_fixed_full():
    result=module.full_support(evidence())
    assert result['eligible'] and result['status']=='supports_fixed_full'
    assert result['platform_gain_known'] is False and result['assessment']


def test_small_original_label_drop_is_retained_for_review_not_rejected():
    result=module.full_support(evidence(net=-1))
    assert not result['eligible'] and result['status']=='evidence_requires_review'
    assert result['candidate_rejected'] is False


def test_sole_micro_gain_does_not_fund_full():
    assert not module.full_support(evidence(macro=-.01))['eligible']


def test_tail_or_sparse_or_raw_disagreement_holds_full():
    for slice_name in ('tail','few_support'):
        x=evidence();x['paired']['ema_swa_2_4']['slices'][slice_name]['net']=-1
        assert not module.full_support(x)['eligible']
    x=evidence();x['trajectory'][-1]['raw']['slices']['all']['net']=-1
    assert not module.full_support(x)['eligible']


def test_each_job_has_one_train_then_one_single_checkpoint_infer():
    job=dict(output='/run/wft',plan='/run/wft/prepared/plan.json')
    commands=module.stage_commands(job)
    assert [stage for stage,_ in commands]==['train','infer']
    assert '/run/wft/run/training/selected.pt' in commands[1][1]
    assert all('--execute' in argv for _,argv in commands)
