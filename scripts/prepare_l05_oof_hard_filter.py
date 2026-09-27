#!/usr/bin/env python3
"""CPU-only, content-group cross-fitted hard filtering for the 750-class stage.

The output is a proposed training subset, not a validated model or submission.
No validation or test image enters the confidence calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.linalg
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.oof.build_folds import assign_group_stratified_folds  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def canonical(path: str) -> str:
    value = str(path).replace('\\', '/')
    return value.removeprefix('train/')


def make_folds(frame: pd.DataFrame, folds: int, seed: int, classes: int) -> np.ndarray:
    source = frame[['image_path', 'label', 'content_group']].copy()
    source['sample_id'] = source['image_path'].map(canonical)
    source['sha256'] = source['content_group']
    assigned = assign_group_stratified_folds(source, n_splits=folds, seed=seed)
    if assigned['image_path'].tolist() != frame['image_path'].tolist():
        raise ValueError('fold assignment changed sample order')
    if sorted(frame.label.unique().tolist()) != list(range(classes)):
        raise ValueError('training classes are incomplete')
    for fold in range(folds):
        seen = set(frame.loc[assigned.fold != fold, 'label'].tolist())
        if len(seen) != classes:
            raise ValueError(f'fold {fold} training partition lacks a class')
    return assigned.fold.to_numpy(dtype=np.int8)


def score_oof(features: np.ndarray, labels: np.ndarray, folds: np.ndarray,
              classes: int, ridge_lambda: float) -> pd.DataFrame:
    """Fit two class predictors on other folds, then score only held-out rows."""
    n, dim = features.shape
    if n != len(labels) or n != len(folds):
        raise ValueError('feature, label and fold counts differ')
    if not np.isfinite(features).all():
        raise ValueError('non-finite feature')
    result = {
        'centroid_top1': np.full(n, -1, dtype=np.int32),
        'centroid_margin': np.full(n, np.nan, dtype=np.float32),
        'ridge_top1': np.full(n, -1, dtype=np.int32),
        'ridge_margin': np.full(n, np.nan, dtype=np.float32),
    }
    for fold in sorted(np.unique(folds).tolist()):
        held = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        x_train = features[train]
        y_train = labels[train]
        counts = np.bincount(y_train, minlength=classes)
        if np.any(counts == 0):
            raise ValueError(f'fold {fold} lacks a training class')
        sums = np.zeros((classes, dim), dtype=np.float32)
        np.add.at(sums, y_train, x_train)
        means = sums / counts[:, None]
        prototypes = means / np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
        gram = x_train.T @ x_train
        gram.flat[::dim + 1] += np.float32(ridge_lambda)
        weights = scipy.linalg.solve(gram, means.T, assume_a='pos', check_finite=False).T
        weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
        for start in range(0, len(held), 2048):
            rows = held[start:start + 2048]
            own = labels[rows]
            batch = features[rows]
            for prefix, bank in (('centroid', prototypes), ('ridge', weights)):
                scores = batch @ bank.T
                top1 = scores.argmax(axis=1)
                margin = scores[np.arange(len(rows)), top1] - scores[np.arange(len(rows)), own]
                result[f'{prefix}_top1'][rows] = top1
                result[f'{prefix}_margin'][rows] = margin
        print(f'fold={fold} reference={len(train)} heldout={len(held)}', flush=True)
    if any(np.any(~np.isfinite(value)) for value in result.values()):
        raise RuntimeError('OOF scores incomplete')
    return pd.DataFrame(result)


def select_rejections(frame: pd.DataFrame, scores: pd.DataFrame, *,
                      max_drop_fraction: float, max_drop_per_class_fraction: float,
                      minimum_class_support_for_drop: int,
                      minimum_class_size_after_drop: int) -> pd.DataFrame:
    """Apply the preregistered one-percent cap and rare-class protection."""
    if len(frame) != len(scores):
        raise ValueError('score count differs from training rows')
    work = pd.concat([frame[['image_path', 'label', 'content_group']].reset_index(drop=True),
                      scores.reset_index(drop=True)], axis=1)
    agreement = (work.centroid_top1 == work.ridge_top1) & (work.centroid_top1 != work.label)
    work = work.loc[agreement & (work.centroid_margin > 0) & (work.ridge_margin > 0)].copy()
    work['consensus_margin'] = np.minimum(work.centroid_margin, work.ridge_margin)
    work = work.sort_values(['consensus_margin', 'image_path'], ascending=[False, True])
    support = frame.groupby('label').size().to_dict()
    limits = {label: min(math.floor(size * max_drop_per_class_fraction),
                         size - minimum_class_size_after_drop)
              if size >= minimum_class_support_for_drop else 0
              for label, size in support.items()}
    budget = math.floor(len(frame) * max_drop_fraction)
    taken = {label: 0 for label in support}
    selected = []
    for row in work.itertuples(index=False):
        if len(selected) >= budget:
            break
        if taken[row.label] < limits[row.label]:
            selected.append(row)
            taken[row.label] += 1
    return pd.DataFrame(selected, columns=work.columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if config['data_version'] != '20260921' or config['num_classes'] != 750:
        raise ValueError('unexpected stage scope')
    manifest_path = Path(config['dataset_manifest'])
    manifest = json.loads(manifest_path.read_text())
    feature_manifest_path = Path(config['feature_manifest'])
    feature_manifest = json.loads(feature_manifest_path.read_text())
    if manifest['data_version'] != config['data_version'] or manifest['num_classes'] != config['num_classes']:
        raise ValueError('dataset manifest mismatch')
    if feature_manifest['rematch_binding']['dataset_manifest_sha256'] != digest(manifest_path):
        raise ValueError('feature cache is not bound to this stage')
    data_dir = manifest_path.parent
    train_path, val_path = data_dir / 'train_dev.csv', data_dir / 'val_dev.csv'
    feature_paths_path = feature_manifest_path.parent / 'image_paths.json'
    feature_tensor_path = feature_manifest_path.parent / 'features.pt'
    for name, path in (('train_dev.csv', train_path), ('val_dev.csv', val_path)):
        if digest(path) != manifest['files'][name]:
            raise ValueError(f'{name} hash mismatch')
    for key, path in (('paths_file_sha256', feature_paths_path), ('tensor_sha256', feature_tensor_path)):
        if digest(path) != feature_manifest[key]:
            raise ValueError(f'feature {key} hash mismatch')
    frame = pd.read_csv(train_path).sort_values('image_path').reset_index(drop=True)
    val = pd.read_csv(val_path, usecols=['image_path', 'content_group'])
    if len(frame) != manifest['train_dev_samples'] or len(val) != manifest['val_dev_samples']:
        raise ValueError('split sample count mismatch')
    if not frame.image_path.is_unique or set(frame.content_group) & set(val.content_group):
        raise ValueError('train/validation leakage or duplicate paths')
    cache_paths = json.loads(feature_paths_path.read_text())
    full_path = data_dir / 'full_train.csv'
    if digest(full_path) != manifest['files']['full_train.csv']:
        raise ValueError('full_train.csv hash mismatch')
    full = pd.read_csv(full_path, usecols=['image_path', 'label'])
    if [canonical(path) for path in full.image_path] != [canonical(path) for path in cache_paths]:
        raise ValueError('feature path order differs from full_train.csv')
    cache_index = {canonical(path): index for index, path in enumerate(cache_paths)}
    if len(cache_index) != len(cache_paths):
        raise ValueError('duplicate canonical cache paths')
    indices = [cache_index[canonical(path)] for path in frame.image_path]
    if not np.array_equal(full.label.to_numpy(dtype=np.int32)[indices], frame.label.to_numpy(dtype=np.int32)):
        raise ValueError('feature-cache labels differ from train_dev.csv')
    torch.set_num_threads(4)
    tensor = torch.load(feature_tensor_path, map_location='cpu', weights_only=True)
    if tensor.shape != (manifest['train_samples'], 512):
        raise ValueError('feature tensor shape mismatch')
    features = np.ascontiguousarray(tensor.numpy()[indices], dtype=np.float32)
    labels = frame.label.to_numpy(dtype=np.int32)
    folds = make_folds(frame, config['folds'], config['seed'], config['num_classes'])
    with threadpool_limits(limits=4):
        scores = score_oof(features, labels, folds, config['num_classes'], config['ridge_lambda'])
    scores.insert(0, 'fold', folds)
    rejected = select_rejections(
        frame, scores,
        max_drop_fraction=config['max_drop_fraction'],
        max_drop_per_class_fraction=config['max_drop_per_class_fraction'],
        minimum_class_support_for_drop=config['minimum_class_support_for_drop'],
        minimum_class_size_after_drop=config['minimum_class_size_after_drop'],
    )
    if rejected.empty:
        raise RuntimeError('fixed conservative rule selected no samples')
    keep = frame.loc[~frame.image_path.isin(rejected.image_path)].copy()
    if len(keep) + len(rejected) != len(frame) or keep.label.nunique() != config['num_classes']:
        raise RuntimeError('filtered split lost rows or classes')
    out = (ROOT / config['output']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    score_path = out / 'confidence_scores.csv'
    reject_path = out / 'rejected.csv'
    train_out = out / 'train_filtered.csv'
    pd.concat([frame[['image_path', 'label', 'content_group']], scores], axis=1).to_csv(score_path, index=False)
    rejected.to_csv(reject_path, index=False)
    keep.to_csv(train_out, index=False)
    summary = {
        'experiment_id': config['experiment_id'], 'status': 'cpu_prepared_not_trained',
        'stage': 'repechage', 'data_version': config['data_version'],
        'confidence_definition': 'OOF agreement of normalized class-centroid and ridge-whitened class-centroid top1, ranked by minimum positive margin; margins are not calibrated probabilities',
        'training_rows': len(frame), 'rejected_rows': len(rejected), 'retained_rows': len(keep),
        'rejected_fraction': len(rejected) / len(frame), 'retained_classes': int(keep.label.nunique()),
        'minimum_retained_class_support': int(keep.groupby('label').size().min()),
        'fold_counts': {str(k): int(v) for k, v in pd.Series(folds).value_counts().sort_index().items()},
        'validation_rows_used_for_scoring': 0, 'test_rows_used': 0,
        'config': config, 'hashes': {
            'config': digest(config_path), 'dataset_manifest': digest(manifest_path),
            'feature_manifest': digest(feature_manifest_path), 'train_dev': digest(train_path),
            'val_dev': digest(val_path), 'feature_tensor': digest(feature_tensor_path),
            'feature_paths': digest(feature_paths_path), 'scores': digest(score_path),
            'rejected': digest(reject_path), 'train_filtered': digest(train_out),
        },
    }
    (out / 'manifest.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: summary[k] for k in ('status', 'training_rows', 'rejected_rows', 'retained_rows', 'retained_classes', 'minimum_retained_class_support')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
