"""Bind frozen ND-CW evidence to a V2 pair, without starting CUDA."""
from __future__ import annotations

import json
from pathlib import Path
import yaml

from v2.plan import ROOT, STAGES, dump, require, sha
from . import io


def verify_bundle(path, require_gate=True):
    path = Path(path).resolve()
    meta = json.loads(path.read_text())
    require(meta['data_version'] == '20260921' and meta['total_images'] == 148695, 'Foreign ND-CW manifest')
    require(not require_gate or meta['training_allowed'] is True, 'A0 scale/coverage/independence gates did not pass')
    io.verify_files(meta['inputs'])
    require(sha(meta['freeze_file']) == meta['freeze_sha256'] and sha(meta['audit_report']) == meta['audit_report_sha256'],
            'Changed frozen thresholds/audit')
    for name, digest in meta['files'].items():
        require(sha(path.parent / name) == digest, f'Changed ND-CW artifact: {name}')
    return path, meta


def attach_weights(plan, output, settings):
    """Called by v2.plan.prepare; all supervision remains outside the sampler."""
    from v2.runtime import records_and_split
    bundle, meta = verify_bundle(settings['manifest'])
    require(settings['arm'] in ('control', 'treatment'), 'Unknown paired arm')
    records, split = records_and_split(output)
    require(len(records) == meta['total_images'], 'ND-CW/V2 row count mismatch')
    plan['inputs'].update(meta['inputs'])
    plan['inputs'].update({str(bundle.parent / n): h for n, h in meta['files'].items()})
    plan['inputs'].update({str(bundle): sha(bundle), meta['freeze_file']: meta['freeze_sha256'],
                          meta['audit_report']: meta['audit_report_sha256']})
    for stage in STAGES:
        template = ROOT / 'configs/v2_ndcw' / (stage + '.yaml')
        template_cfg = yaml.safe_load(template.read_text())
        require(template_cfg == yaml.safe_load((ROOT / 'configs/v2/stages' / (stage + '.yaml')).read_text()),
                'ND-CW template changed a frozen V2 hyperparameter')
        plan['inputs'][str(template)] = sha(template)
        name = 'control_weights.csv' if settings['arm'] == 'control' else (
            'ndcw_weights.csv' if stage == 'full_576' else 'ndcw_dev_weights.csv')
        sidecar = bundle.parent / name
        weights = io.load_sidecar(sidecar, records, expected_sha=meta['files'][name])
        if stage != 'full_576':
            require(bool((weights[split['val']] == 1).all()), 'Dev weights must not attenuate holdout supervision')
        if settings['arm'] == 'control':
            require(bool((weights == 1).all()), 'Control sidecar must be all ones')
        path = output / plan['stages'][stage]['config']
        cfg = json.loads(path.read_text())
        cfg['ndcw'] = dict(sidecar=str(sidecar), sha256=sha(sidecar),
                           manifest=str(bundle), arm=settings['arm'], loss_only=True)
        dump(path, cfg)
        plan['stages'][stage]['config_sha256'] = sha(path)
    groups = json.loads((bundle.parent / 'evaluation_groups.json').read_text())
    expected_val = {'train/' + records[i]['relative_path'] for i in split['val']}
    require(set(groups['val_paths']) == expected_val, 'Paired evaluation groups use a foreign split')
    for path in Path(__file__).parent.glob('*.py'):
        plan['inputs'][str(path)] = sha(path)


def prepare_pair(manifest, output):
    from v2.plan import prepare
    bundle, meta = verify_bundle(manifest)
    output = io.fresh(output)
    original = json.loads((ROOT / 'configs/v2/recipe.json').read_text())
    for arm in ('control', 'treatment'):
        recipe = dict(original, experiment_id='NDCW_N1_' + arm.upper() + '_20260930',
                      ndcw=dict(manifest=str(bundle), arm=arm))
        path = output / (arm + '_recipe.json')
        dump(path, recipe)
        prepare(path, output / arm)
    record = dict(status='prepared_no_training', manifest=str(bundle), manifest_sha256=sha(bundle),
                  arms={a: dict(plan=str(output / a / 'plan.json'), plan_sha256=sha(output / a / 'plan.json')) for a in ('control', 'treatment')},
                  allowed_stage='s1_384', probe_required=True,
                  paired_policy='same split/seed/official init/sampler/order/augmentation/LR/EMA; ND loss weights only')
    dump(output / 'pair.json', record)
    return record
