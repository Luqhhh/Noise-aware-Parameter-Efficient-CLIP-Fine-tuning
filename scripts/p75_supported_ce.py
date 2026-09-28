#!/usr/bin/env python3
"""Fixed training-only evidence and private runtime for P75_SUPPORTED_CE."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import torch
import yaml

from p75_framework import ARCHIVE_PATHS
from p75_hard_overlay import apply as apply_support

ROOT = Path(__file__).resolve().parents[1]
FIXED = ROOT/'configs/p75_supported_ce_20260929/fixed.json'
OUT = ROOT/'outputs/codex/p75_supported_ce_20260929'
FRAMEWORK = OUT/'framework'
CONFIG = OUT/'configs/P75_SUPPORTED_CE.yaml'
RUN = OUT/'runs/P75_SUPPORTED_CE/seed42'
ROW_FIELDS = ['image_path', 'label', 'content_group']
IMPLEMENTATION = (
    'scripts/p75_runtime/p75_hard_support.py', 'scripts/p75_hard_overlay.py',
    'scripts/p75_supported_ce.py', 'scripts/p75_supported_ce_snapshot.py',
    'scripts/p75_supported_ce_report.py', 'scripts/run_p75_supported_ce.py',
    'configs/p75_supported_ce_20260929/fixed.json',
)


def sha(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_rows(path, fields, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def canonical(path):
    value = str(path).replace('\\', '/')
    if value.startswith('train/'):
        value = value[len('train/'):]
    if Path(value).is_absolute() or '..' in Path(value).parts or not value:
        raise ValueError('Expected a current-stage relative training path')
    return value


def validate_rows(training, validation, num_classes):
    for rows in (training, validation):
        paths = [canonical(row['image_path']) for row in rows]
        if len(set(paths)) != len(paths):
            raise ValueError('Duplicate image paths in split')
        if any(not row['content_group'] or int(row['label']) not in range(num_classes)
               for row in rows):
            raise ValueError('Invalid original label or content group')
    if ({canonical(row['image_path']) for row in training} &
            {canonical(row['image_path']) for row in validation} or
            {row['content_group'] for row in training} &
            {row['content_group'] for row in validation}):
        raise ValueError('Training and validation paths/content groups overlap')


def assert_alignment(expected, actual):
    if len(expected) != len(actual) or any(
        tuple(str(row[key]) for key in ROW_FIELDS) !=
        tuple(str(other[key]) for key in ROW_FIELDS)
        for row, other in zip(expected, actual)
    ):
        raise ValueError('Evidence path/label/content-group row order mismatch')


def asset_identity():
    """No images, OOF records, test predictions, or device initialization."""
    fixed = read_json(FIXED)
    assets = Path(fixed['assets'])
    manifest = read_json(assets/'dataset_manifest.json')
    if (manifest['stage'], manifest['data_version']) != ('repechage', '20260921'):
        raise ValueError('Wrong data stage')
    if manifest['validation_overlap_with_training'] or manifest['external_data']:
        raise ValueError('Unsupported data boundary')
    files = ('train_dev.csv', 'val_dev.csv', 'full_train.csv', 'class_to_idx.json')
    hashes = {name: sha(assets/name) for name in files}
    if any(hashes[name] != manifest['files'][name] for name in files):
        raise ValueError('Stage file hash mismatch')
    training, validation = read_rows(assets/'train_dev.csv'), read_rows(assets/'val_dev.csv')
    num_classes = len(read_json(assets/'class_to_idx.json'))
    if (len(training), len(validation), num_classes) != (
        manifest['train_dev_samples'], manifest['val_dev_samples'], manifest['num_classes']):
        raise ValueError('Stage split sizes changed')
    validate_rows(training, validation, num_classes)
    for key, expected in [('parent_checkpoint', 'parent_sha256'),
                          ('control_checkpoint', 'control_checkpoint_sha256')]:
        if sha(fixed[key]) != fixed[expected]:
            raise ValueError(f'Pinned checkpoint hash mismatch: {key}')
    features = assets/'features'
    feature_manifest = read_json(features/'manifest.json')
    feature_paths = read_json(features/'image_paths.json')
    full_rows = read_rows(assets/'full_train.csv')
    if [canonical(row['image_path']) for row in full_rows] != [canonical(p) for p in feature_paths]:
        raise ValueError('Frozen feature rows do not match current-stage full_train order')
    if (feature_manifest['backbone'], feature_manifest['pretrained']) != ('ViT-B/32', 'openai'):
        raise ValueError('Frozen features must come from official OpenAI ViT-B/32')
    if (feature_manifest['external_data'] is not False or
            feature_manifest['test_data_used'] is not False or
            feature_manifest['normalized'] is not True or
            feature_manifest['augmentation'] != 'none' or
            feature_manifest['source_root'] != manifest['train_root'] or
            feature_manifest['dataset_size'] != len(feature_paths)):
        raise ValueError('Frozen feature provenance mismatch')
    binding = feature_manifest['rematch_binding']
    for key, expected in [('stage', manifest['stage']), ('data_version', manifest['data_version']),
                          ('dataset_manifest_sha256', sha(assets/'dataset_manifest.json')),
                          ('class_mapping_sha256', hashes['class_to_idx.json']),
                          ('dataset_fingerprint', manifest['train_fingerprint'])]:
        if binding.get(key) != expected:
            raise ValueError(f'Frozen feature lineage mismatch: {key}')
    paths_hash, tensor_hash = sha(features/'image_paths.json'), sha(features/'features.pt')
    lines_hash = hashlib.sha256(('\n'.join(feature_paths)+'\n').encode()).hexdigest()
    if (paths_hash != feature_manifest['paths_file_sha256'] or
            tensor_hash != feature_manifest['tensor_sha256'] or
            lines_hash != feature_manifest['path_index_sha256']):
        raise ValueError('Frozen feature file/index hash mismatch')
    official_hash = sha('/home/lux1/.cache/clip/ViT-B-32.pt')
    if official_hash != binding['official_checkpoint_sha256']:
        raise ValueError('Official OpenAI weights changed')
    return dict(stage=manifest['stage'], data_version=manifest['data_version'],
        stage_manifest_sha256=sha(assets/'dataset_manifest.json'), split_hashes=hashes,
        parent_sha256=fixed['parent_sha256'], control_sha256=fixed['control_checkpoint_sha256'],
        feature_manifest_sha256=sha(features/'manifest.json'), feature_paths_sha256=paths_hash,
        feature_tensor_sha256=tensor_hash, official_checkpoint_sha256=official_hash,
        reference_cache_sha256=sha(fixed['reference_validation_cache']),
        framework_commit=fixed['framework_commit'], num_classes=num_classes,
        training_samples=len(training), validation_samples=len(validation))


def environment():
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(FRAMEWORK/'support_runtime'),
        str(FRAMEWORK/'reproducibility/aegis_f1')))
    return env


def materialize():
    manifest_path = FRAMEWORK/'source_manifest.json'
    commit = read_json(FIXED)['framework_commit']
    if FRAMEWORK.exists():
        manifest = read_json(manifest_path)
        if manifest['commit'] != commit:
            raise ValueError('Private framework commit changed')
        verify_files(FRAMEWORK, manifest['files'])
        return manifest
    archive = subprocess.check_output(['git', 'archive', commit, *ARCHIVE_PATHS], cwd=ROOT)
    FRAMEWORK.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        if any(Path(member.name).is_absolute() or '..' in Path(member.name).parts or
               not (member.isfile() or member.isdir()) for member in stream.getmembers()):
            raise ValueError('Unsafe private runtime archive')
        stream.extractall(FRAMEWORK)
    apply_support(FRAMEWORK/'reproducibility/aegis_f1/aegis_clip/trainer.py')
    # Include only this adapter. The shared p75_runtime directory contains a
    # sitecustomize hook for a different model candidate and must not be used.
    runtime = FRAMEWORK/'support_runtime'
    runtime.mkdir()
    (runtime/'p75_hard_support.py').write_bytes((ROOT/'scripts/p75_runtime/p75_hard_support.py').read_bytes())
    manifest = dict(commit=commit, overlay='p75_hard_overlay', files={
        str(path.relative_to(FRAMEWORK)): sha(path) for path in sorted(FRAMEWORK.rglob('*'))
        if path.is_file()})
    write_json(manifest_path, manifest)
    return manifest


def verify_files(root, hashes):
    for name, expected in hashes.items():
        if sha(Path(root)/name) != expected:
            raise ValueError(f'Frozen file hash mismatch: {name}')


def candidate_config(original, identity):
    config = copy.deepcopy(original)
    config.pop('_config_path', None)
    config['project'].update(experiment_id='P75_SUPPORTED_CE', trial_id='P75_SUPPORTED_CE',
        mechanism='supported_original_label_ce', preparation_identity=identity)
    config['output']['root'] = str(OUT/'runs')
    config['loss']['hard_support'] = dict(enabled=False, version=1, maximum_weight=.5,
        ce_restore_enabled=True, local_restore_enabled=False,
        path=str(OUT/'selection/support.csv'), sha256=None)
    return config


def validate_recipe(config, original, *, bound):
    expected = candidate_config(original, config['project']['preparation_identity'])
    actual = copy.deepcopy(config)
    actual.pop('_config_path', None)
    options = actual['loss']['hard_support']
    if bound:
        if options['enabled'] is not True or not isinstance(options.get('sha256'), str):
            raise ValueError('Supported CE config is not bound to admitted evidence')
        options['enabled'] = False
        options['sha256'] = None
    if actual != expected:
        raise ValueError('Candidate drifted from L05 beyond identity/output/support fields')
    if (config['loss']['name'] != 'gce' or config['loss']['gce_q'] != .5 or
            config['loss']['ce_warmup_epochs'] != 2 or
            config['loss']['attention_local_training']['confidence_gate'] != .7 or
            config['loss']['mixup_probability'] != 0 or config['trust']['enabled'] or
            config['train'].get('sam', {}).get('enabled', False) or
            config['train']['epochs'] != 16 or config['train']['effective_batch_size'] != 1024 or
            config['model']['input_resolution'] != 384 or
            options['local_restore_enabled'] is not False):
        raise ValueError('Unsupported P75_SUPPORTED_CE recipe')


def prepare():
    identity = asset_identity()
    materialize()
    saved_path = OUT/'preparation.json'
    implementation = {name: sha(ROOT/name) for name in IMPLEMENTATION}
    if saved_path.exists():
        saved = read_json(saved_path)
        if saved['identity'] != identity or saved['implementation'] != implementation:
            raise ValueError('Preparation identity/implementation changed')
        verify_files(OUT, saved['files'])
        return saved
    # This subprocess loads metadata/weights only on CPU; it never builds a model.
    control = read_json(FIXED)['control_checkpoint']
    command = ('import torch,yaml;'
        'from aegis_clip.npu_checkpoint_compat import ensure_npu_checkpoint_stubs;'
        'ensure_npu_checkpoint_stubs();'
        f'p=torch.load({control!r},map_location="cpu",weights_only=False);'
        'print(yaml.safe_dump(p["config"],sort_keys=False))')
    original = yaml.safe_load(subprocess.check_output(
        ['python3', '-c', command], cwd=FRAMEWORK, env=environment(), text=True))
    if original['train']['init_checkpoint'] != read_json(FIXED)['parent_checkpoint']:
        raise ValueError('L05 RM-LP parent changed')
    config = candidate_config(original, identity)
    validate_recipe(config, original, bound=False)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    template = OUT/'configs/template.yaml'
    with template.open('x') as stream:
        stream.write(yaml.safe_dump(config, sort_keys=False))
    write_json(OUT/'configs/l05_original.json', original)
    saved = dict(experiment_id='P75_SUPPORTED_CE', status='prepared_no_gpu', identity=identity,
        rules=read_json(FIXED)['rules'], implementation=implementation,
        files={'configs/template.yaml': sha(template),
               'configs/l05_original.json': sha(OUT/'configs/l05_original.json')},
        snapshot=None, support_count=None, candidate_metrics=None, platform_score=None)
    write_json(saved_path, saved)
    return saved


def preflight(*, require_support=False):
    saved = prepare()
    original = read_json(OUT/'configs/l05_original.json')
    config_path = CONFIG if require_support else OUT/'configs/template.yaml'
    if require_support:
        selected = read_json(OUT/'selection/manifest.json')
        if selected['status'] != 'admitted' or selected['support_count'] == 0:
            raise ValueError('Fixed support rule produced an empty set; candidate is closed')
        if selected['identity'] != saved['identity'] or selected['rules'] != saved['rules']:
            raise ValueError('Selection identity/rules changed')
        verify_files(OUT/'selection', selected['files'])
        if sha(CONFIG) != selected['config_sha256']:
            raise ValueError('Bound training config changed')
        snapshot = read_json(OUT/'snapshot/manifest.json')
        if sha(OUT/'snapshot/manifest.json') != selected['snapshot_manifest_sha256']:
            raise ValueError('Snapshot binding changed')
        verify_files(OUT/'snapshot', snapshot['files'])
    config = yaml.safe_load(config_path.read_text())
    validate_recipe(config, original, bound=require_support)
    script = ('from aegis_clip.config import load_config;'
        'from aegis_clip.rematch_protocol import validate_checkpoint;'
        f'c=load_config({str(config_path)!r});'
        'validate_checkpoint(c["train"]["init_checkpoint"],c,parent=True)')
    subprocess.run(['python3', '-c', script], cwd=FRAMEWORK, env=environment(), check=True)
    if require_support:
        script = ('import csv;from p75_hard_support import load_support;'
            'from aegis_clip.config import load_config;'
            f'c=load_config({str(config_path)!r});'
            'r=list(csv.DictReader(open(c["data"]["train_csv"])));'
            'load_support(c,[v["image_path"] for v in r],[int(v["label"]) for v in r])')
        subprocess.run(['python3', '-c', script], cwd=FRAMEWORK, env=environment(), check=True)
    return saved


def weak_metrics(first, second, labels):
    """Raw T=1 per-view probabilities. Full distribution differences are logged."""
    if first.shape != second.shape or first.ndim != 2 or len(labels) != len(first):
        raise ValueError('Weak-view probability shapes do not align')
    for probabilities in (first, second):
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or
                (probabilities > 1).any() or
                not np.allclose(probabilities.sum(1), 1., atol=2e-5)):
            raise ValueError('Invalid raw weak-view probabilities')
    labels = np.asarray(labels, dtype=np.int64)
    if (labels < 0).any() or (labels >= first.shape[1]).any():
        raise ValueError('Original labels outside probability columns')
    index = np.arange(len(labels))
    midpoint = .5*(first.astype(np.float64)+second.astype(np.float64))
    def kl(probabilities):
        p = probabilities.astype(np.float64)
        return (p*np.log(np.maximum(p, 1e-300)/np.maximum(midpoint, 1e-300))).sum(1)
    return dict(center_label_probability=first[index, labels],
        flip_label_probability=second[index, labels], center_max_probability=first.max(1),
        flip_max_probability=second.max(1), center_top1=first.argmax(1), flip_top1=second.argmax(1),
        distribution_l1=np.abs(first-second).sum(1), distribution_jsd=.5*(kl(first)+kl(second)))


def low_confidence_mask(metrics, rules):
    return ((metrics['center_label_probability'] < rules['label_probability_both_below']) &
        (metrics['flip_label_probability'] < rules['label_probability_both_below']) &
        (metrics['center_max_probability'] < rules['maximum_probability_both_below']) &
        (metrics['flip_max_probability'] < rules['maximum_probability_both_below']) &
        (metrics['center_top1'] == metrics['flip_top1']))


def nearest_groups(features, training, queries, *, batch_size=16):
    """Exact cosine query blocks over train_dev only; one vote per content group.

    The nearest image represents each group. A conflicting-label group cannot
    support any original label. Ties use original training row order.
    """
    if not 1 <= batch_size <= 16:
        raise ValueError('Neighbor query batch size must be between 1 and 16')
    if features.ndim != 2 or len(features) != len(training) or not np.isfinite(features).all():
        raise ValueError('Training feature rows do not align')
    norms = np.linalg.norm(features, axis=1)
    if not np.allclose(norms, 1., atol=1e-4):
        raise ValueError('Expected normalized frozen features')
    groups = [row['content_group'] for row in training]
    group_labels = {}
    for row in training:
        group_labels.setdefault(row['content_group'], set()).add(int(row['label']))
    queries = np.asarray(queries, dtype=np.int64)
    for offset in range(0, len(queries), batch_size):
        batch = queries[offset:offset+batch_size]
        similarities = features[batch] @ features.T
        for query, scores in zip(batch, similarities):
            seen = {groups[query]}
            neighbors = []
            for index in np.argsort(-scores, kind='stable'):
                group = groups[index]
                if group in seen:
                    continue
                seen.add(group)
                labels = sorted(group_labels[group])
                neighbors.append(dict(training_row=int(index), image_path=training[index]['image_path'],
                    content_group=group, original_labels=labels, similarity=float(scores[index]),
                    supports_original_label=(labels == [int(training[query]['label'])])))
                if len(neighbors) == 20:
                    break
            yield int(query), neighbors


def train_features(training):
    """The shared cache includes val rows; only train_dev rows enter this bank."""
    assets = Path(read_json(FIXED)['assets'])
    paths = read_json(assets/'features/image_paths.json')
    index = {canonical(path): i for i, path in enumerate(paths)}
    if len(index) != len(paths):
        raise ValueError('Duplicate frozen feature paths')
    full = torch.load(assets/'features/features.pt', map_location='cpu', weights_only=True)
    if full.shape != (len(paths), 512):
        raise ValueError('Frozen feature tensor shape changed')
    selected = torch.tensor([index[canonical(row['image_path'])] for row in training])
    bank = full.index_select(0, selected).float()
    del full
    # Cache claims normalized=true: validate rather than silently renormalize.
    return bank.numpy()


def select():
    saved = preflight()
    directory = OUT/'snapshot'
    snapshot = read_json(directory/'manifest.json')
    if (snapshot['identity'] != saved['identity'] or snapshot['temperature'] != 1. or
            snapshot['prior_applied'] or snapshot.get('tta_fusion_applied') is not False or
            snapshot['views'] != ['center_384', 'horizontal_flip_384']):
        raise ValueError('Wrong training snapshot protocol')
    verify_files(directory, snapshot['files'])
    training = read_rows(Path(read_json(FIXED)['assets'])/'train_dev.csv')
    assert_alignment(training, read_rows(directory/'rows.csv'))
    center = np.load(directory/'center_probabilities.npy', mmap_mode='r')
    flip = np.load(directory/'flip_probabilities.npy', mmap_mode='r')
    if center.shape != (len(training), saved['identity']['num_classes']) or flip.shape != center.shape:
        raise ValueError('Training snapshot dimensions changed')
    parts = []
    labels = [int(row['label']) for row in training]
    for offset in range(0, len(training), 1024):
        parts.append(weak_metrics(center[offset:offset+1024], flip[offset:offset+1024],
                                  labels[offset:offset+1024]))
    metrics = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    low = low_confidence_mask(metrics, saved['rules'])
    queries = np.flatnonzero(low)
    destination = OUT/'selection'
    destination.mkdir(exist_ok=False)
    counts = np.zeros(len(training), dtype=np.int64)
    group_counts = np.zeros(len(training), dtype=np.int64)
    if len(queries):
        features = train_features(training)
        with (destination/'neighbors.jsonl').open('x') as stream:
            for query, neighbors in nearest_groups(features, training, queries):
                counts[query] = sum(row['supports_original_label'] for row in neighbors)
                group_counts[query] = len(neighbors)
                stream.write(json.dumps(dict(training_row=query, image_path=training[query]['image_path'],
                    label=int(training[query]['label']), content_group=training[query]['content_group'],
                    neighbors=neighbors), allow_nan=False)+'\n')
        del features
    else:
        (destination/'neighbors.jsonl').touch(exist_ok=False)
    supported = low & (group_counts == saved['rules']['distinct_neighbor_groups']) & (
        counts >= saved['rules']['minimum_label_support'])
    support_rows = [dict(image_path=row['image_path'], label=int(row['label']),
                        support_weight=.5 if supported[index] else 0.)
                    for index, row in enumerate(training)]
    write_rows(destination/'support.csv', ['image_path', 'label', 'support_weight'], support_rows)
    diagnostics = [dict(**{key: row[key] for key in ROW_FIELDS},
        **{key: value[index].item() for key, value in metrics.items()},
        weak_eligible=bool(low[index]), queried_neighbor_groups=int(group_counts[index]),
        original_label_support=int(counts[index]), support_weight=support_rows[index]['support_weight'])
        for index, row in enumerate(training)]
    write_rows(destination/'diagnostics.csv', list(diagnostics[0]), diagnostics)
    config = yaml.safe_load((OUT/'configs/template.yaml').read_text())
    if supported.any():
        config['loss']['hard_support'].update(enabled=True, sha256=sha(destination/'support.csv'))
        with CONFIG.open('x') as stream:
            stream.write(yaml.safe_dump(config, sort_keys=False))
    result = dict(status='admitted' if supported.any() else 'closed_empty_support',
        identity=saved['identity'], rules=saved['rules'], snapshot_manifest_sha256=sha(directory/'manifest.json'),
        weak_eligible_count=int(low.sum()), support_count=int(supported.sum()),
        supported_classes=len({int(training[index]['label']) for index in np.flatnonzero(supported)}),
        files={name: sha(destination/name) for name in ('support.csv', 'diagnostics.csv', 'neighbors.jsonl')},
        config_sha256=sha(CONFIG) if supported.any() else None,
        evidence_scope='final L05 snapshot; conservative candidates, not confirmed clean labels',
        rejected_samples_are_noise=False, validation_images_read=False, test_images_read=False)
    write_json(destination/'manifest.json', result)
    return result
