"""Current-stage, train-only input checks. Never opens a test asset."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path, PurePosixPath

from v2.plan import dump, require, sha

OFFICIAL_SHA = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


def read_csv(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(name, label=None):
    p = PurePosixPath(name)
    require(not p.is_absolute() and len(p.parts) == 3 and p.parts[0] == 'train'
            and '..' not in p.parts and '\\' not in name, 'Only official train paths are allowed')
    if label is not None:
        require(p.parts[1] == f'{int(label):04d}', 'Label/path mismatch')
    return p


def image_file(root, row):
    name = safe_name(row['image_path'], row['label'])
    root = Path(root).resolve()
    path = root.joinpath(*name.parts[1:]).resolve()
    require(path.is_relative_to(root), 'Image symlink escapes train root')
    return path


def fresh(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def binding(config):
    """Audit train metadata and frozen cache without validate_dataset's test-file loop."""
    base = Path(config['stage_artifacts']).resolve()
    dataset = json.loads((base / 'dataset_manifest.json').read_text())
    require(dataset['stage'] == 'repechage' and dataset['data_version'] == '20260921'
            and dataset['num_classes'] == 750 and dataset['train_samples'] == 148695
            and dataset['external_data'] is False, 'Wrong current-stage dataset')
    require(Path(config['train_root']).resolve() == Path(dataset['train_root']).resolve(), 'Wrong train root')
    inputs = {}
    for name in ('full_train.csv', 'train_dev.csv', 'val_dev.csv', 'class_to_idx.json', 'decode_report.json'):
        path = base / name
        digest = sha(path)
        require(dataset['files'][name] == digest, f'Changed official train asset: {name}')
        inputs[str(path)] = digest
    inputs[str(base / 'dataset_manifest.json')] = sha(base / 'dataset_manifest.json')
    decode = json.loads((base / 'decode_report.json').read_text())
    require(decode['status'] == 'passed' and not decode['failures'], 'Unclean decode audit')
    mapping = json.loads((base / 'class_to_idx.json').read_text())
    require(mapping == {f'{i:04d}': i for i in range(750)}, 'Wrong class mapping')
    rows = read_csv(base / 'full_train.csv')
    lookup = {}
    for i, row in enumerate(rows):
        safe_name(row['image_path'], row['label'])
        require(not row['error'] and row['content_group'] and row['file_sha256'], 'Invalid train row')
        require(row['image_path'] not in lookup, 'Repeated train image')
        row['label'] = int(row['label'])
        lookup[row['image_path']] = i
    require(len(rows) == 148695 and {r['label'] for r in rows} == set(range(750)), 'Incomplete full train')
    split = {}
    for key in ('train', 'val'):
        subset = read_csv(base / f'{key}_dev.csv')
        require(all(r == {k: str(v) for k, v in rows[lookup[r['image_path']]].items()} for r in subset),
                'Split records differ from official full train')
        split[key] = [lookup[r['image_path']] for r in subset]
    from v2.plan import check_split
    check_split(rows, split, True)
    require((len(split['train']), len(split['val'])) == (133815, 14880), 'Wrong split counts')
    feature_base = base / 'features'
    fm = json.loads((feature_base / 'manifest.json').read_text())
    expected = dict(stage='repechage', data_version='20260921',
                    dataset_manifest_sha256=inputs[str(base / 'dataset_manifest.json')],
                    dataset_fingerprint=dataset['train_fingerprint'],
                    class_mapping_sha256=inputs[str(base / 'class_to_idx.json')],
                    official_checkpoint_sha256=OFFICIAL_SHA,
                    preprocessing='OpenAI CLIP 224 bicubic resize/center crop/RGB/CLIP normalization',
                    encoder_precision='float32', feature_precision='float32', autocast=False)
    require(fm['rematch_binding'] == expected and fm['pretrained'] == 'openai'
            and fm['backbone'] == 'ViT-B/32' and fm['test_data_used'] is False, 'Foreign/nonfrozen CLIP cache')
    for name, key in [('features.pt', 'tensor_sha256'), ('image_paths.json', 'paths_file_sha256')]:
        digest = sha(feature_base / name)
        require(digest == fm[key], f'Changed cache: {name}')
        inputs[str(feature_base / name)] = digest
    paths = json.loads((feature_base / 'image_paths.json').read_text())
    require(paths == [r['image_path'].removeprefix('train/') for r in rows], 'Cache row alignment mismatch')
    inputs[str(feature_base / 'manifest.json')] = sha(feature_base / 'manifest.json')
    return rows, split, inputs


def verify_files(inputs):
    for name, digest in inputs.items():
        require(sha(name) == digest, f'Frozen input changed: {name}')


def load_sidecar(path, records, *, expected_sha=None):
    """Strict full coverage, path/label/group alignment; independent of sampler weights."""
    import torch
    if expected_sha is not None:
        require(sha(path) == expected_sha, 'ND-CW sidecar checksum mismatch')
    rows = read_csv(path)
    by_name = {}
    for row in rows:
        safe_name(row['image_path'], row['label'])
        require(row['image_path'] not in by_name, 'Repeated ND-CW sample')
        w = float(row['nd_weight'])
        require(0 < w <= 1, 'ND-CW weights must be finite in (0,1]')
        by_name[row['image_path']] = row
    require(set(by_name) == {'train/' + r['relative_path'] for r in records}, 'ND-CW must cover all training rows')
    aligned = []
    for rec in records:
        row = by_name['train/' + rec['relative_path']]
        require(int(row['label']) == rec['label'] and row['content_group'] == rec['content_group'],
                'ND-CW label/content group mismatch')
        aligned.append(float(row['nd_weight']))
    return torch.tensor(aligned, dtype=torch.float32)
