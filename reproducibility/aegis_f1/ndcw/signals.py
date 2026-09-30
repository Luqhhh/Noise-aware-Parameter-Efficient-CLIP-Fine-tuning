"""Read existing same-stage evidence, with coverage and unknowns preserved."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import torch

from v2.plan import require, sha
from .core import boolean
from .io import read_csv, safe_name


def load(config, rows, inputs):
    sources, result = {}, {}
    lookup = {r['image_path']: r for r in rows}
    p0 = Path(config['evidence']['p0_table'])
    summary_path = Path(config['evidence']['p0_summary'])
    summary = json.loads(summary_path.read_text())
    identity = summary['identity']
    base = Path(config['stage_artifacts'])
    require(identity['stage'] == 'repechage' and identity['data_version'] == '20260921'
            and identity['feature_tensor_sha256'] == inputs[str(base / 'features/features.pt')]
            and identity['stage_manifest_sha256'] == inputs[str(base / 'dataset_manifest.json')], 'Foreign P0 evidence')
    for name, digest in identity['split_hashes'].items():
        require(inputs[str(base / name)] == digest, 'P0 split mismatch')
    with gzip.open(p0, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    require(digest == summary['files']['sample_evidence.csv'], 'Changed P0 evidence')
    for r in read_csv(p0):
        name = r['image_path']
        safe_name(name, r['original_label'])
        require(name in lookup and int(r['original_label']) == lookup[name]['label']
                and r['content_group'] == lookup[name]['content_group'] and name not in result, 'P0 alignment mismatch')
        count, votes = float(r['independent_neighbor_groups']), float(r['original_support_votes'])
        require(0 <= votes <= count, 'Invalid existing kNN vote counts')
        result[name] = dict(knn_agreement=votes / count if count else None,
                            prototype_agreement=int(r['oof_centroid_prediction']) == lookup[name]['label'] if r['oof_centroid_prediction'] else None,
                            oof_label_probability=None,
                            oof_ridge_agreement=int(r['oof_ridge_prediction']) == lookup[name]['label'] if r['oof_ridge_prediction'] else None,
                            trusted_val_proxy=boolean(r['weak_consistent']) and boolean(r['snapshot_confidence_gate_passes']),
                            v1_low_trust=None, v1_weight=None)
    require(set(result) == set(lookup), 'P0 coverage incomplete')
    targets_path = Path(config['evidence']['v1_targets'])
    meta_path = targets_path.with_suffix('.sha256.json')
    meta = json.loads(meta_path.read_text())
    require(meta['sha256'] == sha(targets_path), 'V1 targets checksum mismatch')
    b = meta['binding']
    require(b['stage'] == 'repechage' and b['data_version'] == '20260921'
            and b['feature_manifest_sha256'] == inputs[str(base / 'features/manifest.json')]
            and b['dataset_manifest_sha256'] == inputs[str(base / 'dataset_manifest.json')], 'Foreign V1 trust')
    targets = torch.load(targets_path, map_location='cpu', weights_only=False)
    v1_paths = set(targets['image_paths'])
    require(targets['binding'] == b and len(v1_paths) == len(targets['image_paths'])
            and len(targets['labels']) == len(targets['kept']) == len(targets['weights']) == len(v1_paths), 'V1 trust binding mismatch')
    for i, name in enumerate(targets['image_paths']):
        safe_name(name)
        require(name in lookup and int(targets['labels'][i]) == lookup[name]['label'], 'V1 label mismatch')
        result[name].update(v1_low_trust=not bool(targets['kept'][i]), v1_weight=float(targets['weights'][i]))
    # Optional existing OOF table. It must bind the same dataset/cache and contain
    # train_dev only. Missing probabilities remain unknown; no teacher is fitted.
    oof_path = config['evidence'].get('oof_probabilities')
    if oof_path:
        path = Path(oof_path)
        meta_oof = json.loads(path.with_suffix('.binding.json').read_text())
        require(meta_oof['sha256'] == sha(path) and meta_oof['dataset_manifest_sha256'] == b['dataset_manifest_sha256']
                and meta_oof['feature_manifest_sha256'] == b['feature_manifest_sha256'], 'Foreign OOF probabilities')
        seen = set()
        for r in read_csv(path):
            name = r['image_path']
            require(name in v1_paths and name not in seen, 'OOF table must contain unique train_dev rows')
            require(int(r['label']) == lookup[name]['label'], 'OOF label mismatch')
            p = float(r['oof_label_probability'])
            require(0 <= p <= 1, 'Invalid OOF probability')
            result[name]['oof_label_probability'] = p
            seen.add(name)
        sources.update({str(path): sha(path), str(path.with_suffix('.binding.json')): sha(path.with_suffix('.binding.json'))})
    for path in (p0, summary_path, targets_path, meta_path):
        sources[str(path)] = sha(path)
    return result, sources


def cross_report(rows, weights, signals):
    groups = {'nd_raw': [r['image_path'] for r in weights if r['raw_nd_weight'] < 1],
              'nd_weighted': [r['image_path'] for r in weights if r['nd_weight'] < 1],
              'unaffected': [r['image_path'] for r in weights if r['raw_nd_weight'] == 1]}
    result = {}
    for name, paths in groups.items():
        entry = dict(images=len(paths))
        for key in ('v1_low_trust', 'v1_weight', 'oof_label_probability', 'knn_agreement', 'prototype_agreement', 'oof_ridge_agreement'):
            values = [signals[p][key] for p in paths if p in signals and signals[p][key] is not None]
            entry[key] = dict(known=len(values), unknown=len(paths) - len(values),
                              mean=sum(values) / len(values) if values else None)
        result[name] = entry
    return result
