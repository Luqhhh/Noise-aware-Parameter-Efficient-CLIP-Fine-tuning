"""Current-stage decode audit and support-constrained, content-isolated split."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from aegis_clip.runtime import atomic_json_dump, sha256_file, sha256_lines


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_image(task):
    path, relative, label, training = task
    record = dict(image_path=relative, label=label)
    try:
        raw = path.read_bytes()
        record.update(bytes=len(raw), file_sha256=hashlib.sha256(raw).hexdigest())
        with Image.open(io.BytesIO(raw)) as image:
            image.load()  # strict decode, with LOAD_TRUNCATED_IMAGES=False
            rgb = image.convert('RGB')
            record.update(width=rgb.width, height=rgb.height)
            if training:
                digest = hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode())
                digest.update(rgb.tobytes())
                record['content_group'] = digest.hexdigest()
        record['error'] = ''
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    return record


def split_groups(records, num_classes, seed=42, ratio=.1, minimum_train_groups=4):
    """Never split a pixel-content group, even when its labels conflict.

    Greedily meet per-class group targets, protecting four training groups.
    Classes below five groups cannot donate a validation group. Multi-label
    groups are admitted only if all their classes retain training support.
    """
    grouped = defaultdict(list)
    for row in records:
        grouped[row['content_group']].append(row)
    members = {g: set(int(r['label']) for r in rows) for g, rows in grouped.items()}
    by_class = [set() for _ in range(num_classes)]
    for g, labels in members.items():
        for c in labels:
            by_class[c].add(g)
    total = np.array([len(gs) for gs in by_class])
    if (total == 0).any():
        raise ValueError('A training class has no independent content group')
    targets = np.array([min(max(1, round(n*ratio)), n-minimum_train_groups)
                        if n > minimum_train_groups else 0 for n in total])
    remaining = total.copy()
    validation = np.zeros(num_classes, dtype=int)
    rng = np.random.default_rng(seed)
    ordered = sorted(grouped)
    rng.shuffle(ordered)
    rank = {g: i for i, g in enumerate(ordered)}
    selected = set()
    for c in sorted(range(num_classes), key=lambda c: (total[c], c)):
        for g in sorted(by_class[c], key=rank.get):
            if validation[c] >= targets[c]:
                break
            if g in selected:
                continue
            labels = members[g]
            if any(remaining[k] <= minimum_train_groups or targets[k] == 0 for k in labels):
                continue
            # Do not overfill a conflicting class merely to fill another class.
            if any(validation[k] >= targets[k] for k in labels):
                continue
            selected.add(g)
            for k in labels:
                remaining[k] -= 1
                validation[k] += 1
    train = [r for r in records if r['content_group'] not in selected]
    val = [r for r in records if r['content_group'] in selected]
    tr_counts = Counter(int(r['label']) for r in train)
    va_counts = Counter(int(r['label']) for r in val)
    support = []
    for c in range(num_classes):
        support.append(dict(label=c, class_name=f'{c:04d}', total_groups=int(total[c]),
            train_groups=int(remaining[c]), val_groups=int(validation[c]),
            train_samples=tr_counts[c], val_samples=va_counts[c],
            validation_status='covered_noisy_labels' if va_counts[c] else 'no_reliable_holdout',
            segment='tail' if tr_counts[c]<20 else 'middle' if tr_counts[c]<100 else 'head'))
    assert not ({r['content_group'] for r in train} & {r['content_group'] for r in val})
    assert all(r['train_groups'] >= min(minimum_train_groups, r['total_groups']) for r in support)
    return train, val, support


def prepare(config, workers=8):
    d = config['data']
    out = Path(d['dataset_manifest']).parent
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Prepared assets already exist: {out}')
    out.mkdir(parents=True)
    root, test_root = Path(d['train_root']), Path(d['test_root'])
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if classes != [f'{c:04d}' for c in range(config['model']['num_classes'])]:
        raise ValueError('Unexpected official class directories')
    mapping = {c: i for i,c in enumerate(classes)}
    tasks = [(p, f'train/{c}/{p.name}', mapping[c], True)
             for c in classes for p in sorted((root/c).iterdir()) if p.is_file()]
    test_tasks = [(p, f'test/{p.name}', -1, False) for p in sorted(test_root.iterdir()) if p.is_file()]
    if len(tasks)!=d['expected_official_train_samples'] or len(test_tasks)!=d['expected_test_samples']:
        raise ValueError('Official image count mismatch')
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    audits = []
    for kind, source in [('train',tasks),('test',test_tasks)]:
        rows=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i,r in enumerate(pool.map(audit_image, source)):
                rows.append(r)
                if (i+1)%10000==0:
                    print(f'{kind}: strict decoded {i+1}/{len(source)}',flush=True)
        audits.append(rows)
    records, test_records = audits
    failures=[r for rows in audits for r in rows if r['error']]
    atomic_json_dump(dict(status='failed' if failures else 'passed',
        train_checked=len(records), test_checked=len(test_records), failures=failures,
        truncated_images_allowed=False, placeholder_images_allowed=False,
        test_usage='integrity_only', grouping='decoded RGB pixels and dimensions; train only'), out/'decode_report.json')
    if failures:
        raise ValueError(f'{len(failures)} image decode failures; see {out}/decode_report.json')
    train, val, support = split_groups(records,len(classes),config['project']['seed'])
    for name, rows in [('full_train',records),('train_dev',train),('val_dev',val)]:
        write_csv(out/f'{name}.csv',rows)
    write_csv(out/'test_manifest.csv',test_records)
    write_csv(out/'class_support.csv',support)
    write_csv(out/'content_groups.csv',[dict(image_path=r['image_path'],label=r['label'],content_group=r['content_group']) for r in records])
    atomic_json_dump(mapping,out/'class_to_idx.json')
    atomic_json_dump({str(v):k for k,v in mapping.items()},out/'idx_to_class.json')
    files={p.name:sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}
    grouped=defaultdict(set)
    for r in records: grouped[r['content_group']].add(r['label'])
    manifest=dict(format_version=1,stage='repechage',data_version=config['project']['data_version'],
        seed=config['project']['seed'],num_classes=len(classes),train_samples=len(records),test_samples=len(test_records),
        train_dev_samples=len(train),val_dev_samples=len(val),validation_covered_classes=sum(r['val_samples']>0 for r in support),
        independent_groups=len(grouped),conflicting_content_groups=sum(len(s)>1 for s in grouped.values()),
        external_data=False,validation_overlap_with_training=False,test_usage='inference_only',
        train_root=str(root.resolve()),test_root=str(test_root.resolve()),
        train_fingerprint=sha256_lines(f"{r['image_path']}:{r['label']}:{r['file_sha256']}" for r in records),
        test_fingerprint=sha256_lines(f"{r['image_path']}:{r['file_sha256']}" for r in test_records),
        files=files,split_policy=dict(val_ratio=.1,minimum_train_groups=4,content_grouping='decoded_rgb_sha256',label_changes=0,removed_samples=0))
    atomic_json_dump(manifest,out/'dataset_manifest.json')
    print(json.dumps(manifest,indent=2),flush=True)
    return manifest
