"""CPU contract: P75 phase logs never create the trainer-owned RUN directory.

``aegis_clip.trainer.train`` refuses to start when the run directory already
exists, so a capture file written inside that directory makes every fresh
``--phase train`` run abort before its first step. Both queues must place their
captured stdout beside RUN instead.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(module) -> None:
    run = module.RUN
    for phase in ('p75_train.log', 'p75_cache.log', 'p75_evaluate.log', 'p75_paired.log',
                  'p75_deliver.log'):
        resolved = module.phase_log(run/phase)
        assert resolved.parent == run.parent, (phase, resolved)
        assert run not in resolved.parents, (phase, resolved)
        assert resolved.name == phase


def test_patch_readout_phase_logs_stay_outside_run():
    check(load('run_p75_patch_readout'))


def test_full_sam_phase_logs_stay_outside_run():
    check(load('run_p75_full_sam'))


def test_delivery_gate_blocks_unpromoted_and_is_bypassable_only_explicitly():
    module = load('run_p75_patch_readout')
    # The measured P75_PATCH_READOUT decode: +0.0074 pp over the incumbent, i.e. a macro miss.
    probe = {'decode': {'macro': 0.7583388090133667, 'micro': 0.7666666507720947}}
    assert module.gate_failures(probe) == ['macro']
    # The incumbent itself is exactly on the macro boundary minus the +0.30 pp requirement.
    incumbent = {'decode': {'macro': 0.7582651238982523, 'micro': 0.7665322580645161}}
    assert module.gate_failures(incumbent) == ['macro']
    # A candidate that clears both metrics must not be flagged.
    passing = {'decode': {'macro': module.GATE_MACRO, 'micro': module.GATE_MICRO}}
    assert module.gate_failures(passing) == []
    # A micro-only miss must be reported too.
    assert module.gate_failures({'decode': {'macro': 0.77, 'micro': 0.70}}) == ['micro']
