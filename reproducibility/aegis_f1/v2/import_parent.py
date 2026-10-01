"""Import a completed S1 by retaining its exact config, manifest and checkpoint.

Later configs use the destination resources. Existing checkpoint guards remain
unchanged; an imported stage cannot be trained again in the destination plan.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
from pathlib import Path, PureWindowsPath
import shutil
import subprocess

from .plan import ROOT, STAGES, dump, json_read, require, sha, verify_prepared
from .runtime import read_checkpoint

BASELINE = 'bde43f7f1fc9f21c2172c7f295babf0368bbb455'


def canonical_config(config):
    value = copy.deepcopy(config)
    for section, keys in {'paths': ('project_root',), 'data': ('train_dir', 'num_workers',),
                          'model': ('official_checkpoint',),
                          'local_replay': ('micro_batch_size',)}.items():
        for key in keys:
            value[section].pop(key, None)
    return value


def manifest_rows(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['relative_path'] = row['relative_path'].replace('\\', '/')
    return rows


def windows_key(value):
    return PureWindowsPath(value).as_posix().casefold()


def shared_ast(source):
    tree = ast.parse(source)
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in ('WeightAverage', 'using_weights')}


def validate_bundle(bundle, destination_plan):
    bundle = Path(bundle)
    manifest = json_read(bundle / 'MANIFEST.json')
    sources = {}
    for entry in manifest['files']:
        relative = Path(entry['bundle_path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe bundle manifest path')
        path = bundle / relative
        require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'Bundle asset changed')
        require(entry.get('match') is not False, 'Export contains a mismatching source asset')
        if entry.get('original_path'):
            key = windows_key(entry['original_path'])
            require(key not in sources, 'Ambiguous original asset path')
            sources[key] = path
    for line in (bundle / 'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split(None, 1)
        relative = Path(name.lstrip('*'))
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe checksum path')
        require(sha(bundle / relative) == digest, 'Bundle checksum failed')
    source_plan = json_read(bundle / 'plan/plan.json')
    require(sha(bundle / 'plan/plan.json') == manifest['plan_sha256'], 'Exported plan hash mismatch')
    require(source_plan['experiment_id'] == destination_plan['experiment_id'] and
            source_plan['data_version'] == destination_plan['data_version'] and
            source_plan['strategy_revision'] == destination_plan['strategy_revision'], 'Wrong experiment/phase')
    official = destination_plan['recipe']['official_checkpoint']
    for name, digest in source_plan['inputs'].items():
        path = Path(official) if windows_key(name) == windows_key(source_plan['recipe']['official_checkpoint']) else sources.get(windows_key(name))
        require(path is not None and sha(path) == digest, 'Original frozen input cannot be verified: ' + name)
    for stage in STAGES:
        relative = source_plan['stages'][stage]['config'].replace('\\', '/')
        path = bundle / 'plan' / relative
        require(path.resolve().is_relative_to((bundle / 'plan').resolve()), 'Unsafe source configuration path')
        require(sha(path) == source_plan['stages'][stage]['config_sha256'], 'Original stage config changed')
    for path in sorted((bundle / 'code_bindings').rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(bundle / 'code_bindings').as_posix()
        local = relative if relative.startswith('configs/') else 'reproducibility/aegis_f1/' + relative
        baseline = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASELINE + ':' + local])
        source = path.read_bytes().replace(b'\r\n', b'\n')
        if relative == 'aegis_clip/v1_strategy.py':
            require(shared_ast(source.decode()) == shared_ast(baseline.decode()), 'EMA implementation differs')
            require(shared_ast(source.decode()) == shared_ast((ROOT / local).read_text()), 'Current EMA implementation differs')
        else:
            require(source == baseline.replace(b'\r\n', b'\n'), 'Unrecognized source implementation: ' + relative)
    payload = read_checkpoint(bundle / 'run/best.pt', source_plan, 's1_384')
    source_config = json_read(bundle / 'plan/configs/s1_384.json')
    require(payload['config'] == source_config, 'Checkpoint embeds another training config')
    require(source_plan['stages']['s1_384']['total_updates'] == 13930 and
            source_plan['stages']['s1_384']['epochs'] == 10, 'Unexpected source endpoint')
    return source_plan, payload


def attach(bundle, plan_path):
    bundle, plan_path = Path(bundle).resolve(), Path(plan_path).resolve()
    plan = verify_prepared(plan_path)
    workspace = plan_path.parent
    require(not (workspace / 'runs').exists() and not plan.get('imported_stages'), 'Import only into a fresh workspace')
    source_plan, payload = validate_bundle(bundle, plan)
    require(manifest_rows(bundle / 'plan/train_manifest.csv') == manifest_rows(workspace / 'train_manifest.csv'), 'Training records differ')
    require(json_read(bundle / 'plan/split.json') == json_read(workspace / 'split.json'), 'Frozen split differs')
    for stage in STAGES:
        source = json_read(bundle / 'plan/configs' / (stage + '.json'))
        target = json_read(workspace / plan['stages'][stage]['config'])
        require(canonical_config(source) == canonical_config(target), 'Training hyperparameters differ: ' + stage)
        source_spec = {k: v for k, v in source_plan['stages'][stage].items() if k not in ('config', 'config_sha256')}
        target_spec = {k: v for k, v in plan['stages'][stage].items() if k not in ('config', 'config_sha256')}
        require(source_spec == target_spec, 'Stage schedule differs: ' + stage)
    for name in ('train_manifest.csv', 'split.json'):
        shutil.copyfile(bundle / 'plan' / name, workspace / name)
        plan['inputs'][str(workspace / name)] = sha(workspace / name)
    s1_config = workspace / plan['stages']['s1_384']['config']
    shutil.copyfile(bundle / 'plan/configs/s1_384.json', s1_config)
    plan['stages']['s1_384']['config_sha256'] = sha(s1_config)
    run = workspace / 'runs/s1_384'
    run.mkdir(parents=True, exist_ok=False)
    for name in ('best.pt', 'best.binding.json', 'status.json', 'history.json'):
        (run / name).symlink_to(bundle / 'run' / name)
    receipt = dict(status='completed_s1_imported', original_checkpoint_preserved=True,
                   source_plan_sha256=sha(bundle / 'plan/plan.json'),
                   source_checkpoint_plan_sha256=payload['binding']['plan_sha256'],
                   source_plan_hash_matches_checkpoint=sha(bundle / 'plan/plan.json') == payload['binding']['plan_sha256'],
                   source_checkpoint_sha256=sha(bundle / 'run/best.pt'),
                   source_stage_config_sha256=sha(s1_config), source_manifest_sha256=sha(workspace / 'train_manifest.csv'),
                   source_epoch=payload['epoch'], source_updates=payload['global_step'], chosen=payload['metrics']['chosen'],
                   completed_epochs=payload['binding']['completed_epochs'], source_metrics=payload['metrics'],
                   baseline_commit=BASELINE, source_bundle=str(bundle),
                   destination=str(workspace), training_started=False,
                   note='The exported plan and checkpoint-recorded plan hashes differ. Both are retained. Original frozen inputs, stage config, manifest, completed endpoint and checkpoint sidecar are validated; no checkpoint metadata is rewritten.')
    receipt_path = workspace / 's1_import.json'
    dump(receipt_path, receipt)
    plan['inputs'][str(receipt_path)] = sha(receipt_path)
    for path in (bundle / 'plan/plan.json', bundle / 'run/best.pt', bundle / 'run/best.binding.json', bundle / 'run/status.json', bundle / 'MANIFEST.json'):
        plan['inputs'][str(path)] = sha(path)
    plan['imported_stages'] = {'s1_384': {'receipt': str(receipt_path), 'sha256': sha(receipt_path)}}
    dump(plan_path, plan)
    verified = verify_prepared(plan_path)
    read_checkpoint(run / 'best.pt', verified, 's1_384')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    print(__import__('json').dumps(attach(args.bundle, args.plan), indent=2))


if __name__ == '__main__':
    main()
