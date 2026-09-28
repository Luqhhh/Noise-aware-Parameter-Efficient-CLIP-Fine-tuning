#!/usr/bin/env python3
"""CPU-only P75 preflight, bounded training-budget audit, and fixed configs."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
FOCUS = Path('/home/lux1/noise/worktrees/rematch750_f05_focus')
ASSETS = Path('/home/lux1/noise/artifacts/stages/repechage/20260921')
OOF = Path('/home/lux1/noise-worktrees/l05_oof_hard_filter/outputs/codex/l05_oof_hard_filter/cpu_prepare/confidence_scores.csv')
CONTROL = FOCUS / 'outputs/f05_focus_l05/RM_V5_L05_CUDA_LOCAL/seed42'
PARENT = FOCUS / 'outputs/f05_focus/local_parent/RM_LP/seed42/checkpoints/best.pt'
OUT = ROOT / 'outputs/codex/p75_mechanisms_20260927'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def preflight() -> dict:
    cfg = json.loads((ROOT/'configs/l05_stochastic_depth/fixed.json').read_text())
    assert sha(PARENT) == cfg['parent_sha256']
    assert sha(CONTROL/'checkpoints/best.pt') == cfg['control_checkpoint_sha256']
    manifest = json.loads((ASSETS/'dataset_manifest.json').read_text())
    assert manifest['stage'] == 'repechage' and manifest['data_version'] == '20260921'
    assert not manifest['validation_overlap_with_training']
    for name, value in manifest['files'].items():
        assert sha(ASSETS/name) == value, name
    training, validation = rows(ASSETS/'train_dev.csv'), rows(ASSETS/'val_dev.csv')
    assert len(training) == 133815 and len(validation) == 14880
    assert not ({r['content_group'] for r in training} &
                {r['content_group'] for r in validation})
    assert len({int(r['label']) for r in training}) == 750
    return {'parent_sha256': cfg['parent_sha256'],
            'control_sha256': cfg['control_checkpoint_sha256'],
            'stage_manifest_sha256': sha(ASSETS/'dataset_manifest.json'),
            'train_csv_sha256': sha(ASSETS/'train_dev.csv'),
            'val_csv_sha256': sha(ASSETS/'val_dev.csv'),
            'oof_sha256': sha(OOF),
            'focus_framework_commit': cfg['framework_commit']}


def audit(identity: dict) -> dict:
    sys.path.insert(0, str(FOCUS/'reproducibility/aegis_f1'))
    from aegis_clip.npu_checkpoint_compat import ensure_npu_checkpoint_stubs
    ensure_npu_checkpoint_stubs()
    training = rows(ASSETS/'train_dev.csv')
    evidence = rows(OOF)
    assert len(training) == len(evidence)
    for original, scored in zip(training, evidence):
        assert (original['image_path'], original['label'], original['content_group']) == (
            scored['image_path'], scored['label'], scored['content_group'])
        assert int(scored['fold']) in range(5)
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    weights = parent['model_state_dict']['classifier.weight'].float()
    bias = parent['model_state_dict']['classifier.bias'].float()
    features = torch.load(ASSETS/'features/features.pt', map_location='cpu', weights_only=False)
    paths = json.loads((ASSETS/'features/image_paths.json').read_text())
    assert features.shape == (len(paths), weights.shape[1])
    path_index = {path: index for index, path in enumerate(paths)}
    assert len(path_index) == len(paths)
    selected = torch.tensor([path_index[r['image_path'].removeprefix('train/')]
                             for r in training], dtype=torch.long)
    labels = torch.tensor([int(r['label']) for r in training], dtype=torch.long)
    torch.set_num_threads(4)
    py = torch.empty(len(training))
    pmax = torch.empty(len(training))
    for offset in range(0, len(selected), 1024):
        end = min(offset+1024, len(selected))
        logits = features[selected[offset:end]].float() @ weights.T + bias
        probabilities = logits.softmax(dim=1)
        py[offset:end] = probabilities.gather(1, labels[offset:end,None]).squeeze(1)
        pmax[offset:end] = probabilities.max(dim=1).values
    del features, parent
    supported = torch.tensor([
        int(row['centroid_top1']) == int(row['label']) and
        int(row['ridge_top1']) == int(row['label']) for row in evidence], dtype=torch.bool)
    weak_bins = [(0., .01), (.01, .05), (.05, .1), (.1, .3), (.3, 1.000001)]
    table = []
    by_class = defaultdict(lambda: Counter())
    for low, high in weak_bins:
        mask = (py >= low) & (py < high)
        both = mask & supported
        table.append({'lp_label_probability': [low, high], 'samples': int(mask.sum()),
                      'gce_scale_mean': float(py[mask].sqrt().mean()) if mask.any() else None,
                      'below_l05_gate_proxy': int((mask & (pmax < .7)).sum()),
                      'oof_dual_centroid_label_support': int(both.sum())})
    candidate = supported & (py < .3) & (pmax < .7)
    for index in candidate.nonzero().flatten().tolist():
        by_class[int(labels[index])]['candidate'] += 1
    logs = []
    for epoch in range(5, 17):
        path = CONTROL/'logs'/f'train_epoch_{epoch}.json'
        if not path.exists():
            path = CONTROL/'logs'/f'evaluation_epoch_{epoch}.json'
        log = json.loads(path.read_text())
        logs.append({'epoch': epoch, 'actual_local_activation_rate':
                     log['train_attention_local_activation_rate'],
                     'actual_active_examples': log['train_attention_local_active_examples'],
                     'actual_fallback_examples': log['train_attention_local_fallback_examples']})
    return {'identity': identity, 'evidence_scope': 'current-stage train only',
            'lp_proxy_not_l05_training_logits': table,
            'oof_proxy_hard_supported_count': int(candidate.sum()),
            'oof_proxy_hard_supported_classes': len(by_class),
            'l05_actual_aggregate_local_gate': logs,
            'gate_semantics': 'max global softmax below 0.7 substitutes global classification loss for local classification loss; no separate crop quality test',
            'missing_for_per_sample_coupling': ['L05 training-side per-sample logits across epochs',
                'content-group-excluded nearest-neighbor support', 'weak-view agreement for the same training rows'],
            'hard_support_decision': 'do_not_launch_without_missing_evidence',
            'test_used_for_scoring': False}


def configs(identity: dict, *, refresh: bool = False) -> list[str]:
    sys.path.insert(0, str(FOCUS/'reproducibility/aegis_f1'))
    from aegis_clip.npu_checkpoint_compat import ensure_npu_checkpoint_stubs
    ensure_npu_checkpoint_stubs()
    control = torch.load(CONTROL/'checkpoints/best.pt', map_location='cpu', weights_only=False)
    original = control['config']
    assert original['train']['init_checkpoint'] == str(PARENT)
    assert original['project']['experiment_id'] == 'RM_V5_L05_CUDA_LOCAL'
    output = []
    for name in ('P75_PATCH_READOUT', 'P75_FULL_SAM'):
        config = copy.deepcopy(original)
        config.pop('_config_path', None)
        config['project'].update(experiment_id=name, trial_id=name,
                                  mechanism=name.lower(), preparation_identity=identity)
        sources = (['scripts/p75_runtime/p75_patch_readout.py',
                    'scripts/p75_runtime/sitecustomize.py']
                   if name == 'P75_PATCH_READOUT' else
                   ['scripts/p75_runtime/p75_sam.py', 'scripts/p75_sam_overlay.py'])
        config['project']['p75_implementation_files'] = {
            source: sha(ROOT/source) for source in sources}
        config['output']['root'] = str(OUT/'runs')
        if name == 'P75_PATCH_READOUT':
            config['model']['patch_readout'] = {'enabled': True, 'version': 1}
        else:
            # The private SAM overlay replays local loss over one effective
            # batch. FP32 is required by the pinned SAM path and is recorded
            # as a numerical protocol difference from the AMP control.
            config['train']['sam'] = {'enabled': True, 'rho': .05,
                                      'mode': 'standard_global_l2'}
            config['train']['amp'] = False
        path = OUT/'configs'/f'{name}.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_dump(config, sort_keys=False)
        if path.exists() and path.read_text() != content and not refresh:
            raise FileExistsError(f'Config drift: {path}')
        path.write_text(content)
        output.append(str(path))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('audit', 'prepare'), required=True)
    parser.add_argument('--refresh', action='store_true',
                        help='Refresh only generated P75 configs after an intentional code edit')
    args = parser.parse_args()
    identity = preflight()
    if args.phase == 'audit':
        result = audit(identity)
        path = OUT/'training_budget.json'
    else:
        result = {'identity': identity, 'configs': configs(identity, refresh=args.refresh),
                  'patch_status': 'cpu_smoke_required_before_gpu',
                  'sam_status': 'private_overlay_cpu_contract_pending; FP32 differs from AMP control',
                  'hard_status': 'blocked_pending_per_sample_evidence'}
        path = OUT/'preparation.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(path)


if __name__ == '__main__':
    main()
