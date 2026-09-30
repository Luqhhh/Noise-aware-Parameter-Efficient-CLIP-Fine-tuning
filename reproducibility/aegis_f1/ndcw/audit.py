"""A0 stages: retrieve, verify, sweep, then freeze thresholds exactly once."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml

from v2.plan import dump, require, sha
from . import core, io, perceptual, retrieval, signals


def config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    require(cfg['k'] == 10 and cfg['minimum_cosine'] == .94 and cfg['class_cap'] == .10,
            'First-round audit policy changed')
    require(cfg['device'] == 'cpu', 'A0 is CPU-only; do not compete with active training')
    require(cfg['thresholds'] == {'medium': None, 'strong': None}, 'Thresholds must come from completed A0 curve')
    return cfg


def source_binding(config_path, cfg):
    rows, split, inputs = io.binding(cfg)
    inputs[str(Path(config_path).resolve())] = sha(config_path)
    for path in Path(__file__).parent.glob('*.py'):
        inputs[str(path)] = sha(path)
    return rows, split, inputs


def build(config_path, output):
    cfg = config(config_path)
    rows, split, inputs = source_binding(config_path, cfg)
    features = torch.load(Path(cfg['stage_artifacts']) / 'features/features.pt', map_location='cpu', weights_only=True)
    require(tuple(features.shape) == (len(rows), 512), 'Wrong feature tensor dimensions')
    output = io.fresh(output)
    torch.set_num_threads(cfg['cpu_threads'])
    started = time.monotonic()
    def progress(done, total, seconds):
        if done % (cfg['query_chunk'] * 50) == 0 or done == total:
            print(json.dumps(dict(stage='knn', done=done, total=total, seconds=seconds)), flush=True)
    try:
        idx, sim = retrieval.neighbors(features, query_chunk=cfg['query_chunk'], gallery_chunk=cfg['gallery_chunk'],
                                       max_seconds=cfg['retrieval_max_seconds'], progress=progress)
        np.savez_compressed(output / 'knn_top10.npz', indices=idx, similarities=sim)
        pairs = list(retrieval.candidate_pairs(rows, idx, sim))
        fields = ['i', 'j', 'image_i', 'image_j', 'label_i', 'label_j', 'cosine_similarity',
                  'content_group_i', 'content_group_j', 'mutual_nn', 'exact']
        io.write_csv(output / 'pair_candidates.csv.gz', pairs, fields)
        dump(output / 'candidates.json', dict(status='retrieved', inputs=inputs, config=str(Path(config_path).resolve()),
              total_images=len(rows), pairs=len(pairs), elapsed_seconds=time.monotonic() - started,
              files={n: sha(output / n) for n in ('knn_top10.npz', 'pair_candidates.csv.gz')},
              retrieval='bounded exact top10; self excluded; exact groups retained; CSV.gz lossless',
              complexity='O(N^2) similarity compute in bounded tiles; O(Nk) neighbor storage; no all-pairs table'))
    except Exception as exc:
        dump(output / 'failed.json', dict(status='failed_no_training', error=str(exc)))
        raise
    return output


def candidate_context(directory):
    directory = Path(directory).resolve()
    meta = json.loads((directory / 'candidates.json').read_text())
    io.verify_files(meta['inputs'])
    for name, digest in meta['files'].items():
        require(sha(directory / name) == digest, 'Changed candidate artifact')
    cfg = config(meta['config'])
    rows, split, _ = io.binding(cfg)
    return directory, meta, cfg, rows, split


def verify(directory):
    directory, meta, cfg, rows, split = candidate_context(directory)
    require(not (directory / 'verified_pairs.csv.gz').exists(), 'Refusing to overwrite perceptual verification')
    pairs = io.read_csv(directory / 'pair_candidates.csv.gz')
    for pair in pairs:
        pair['exact'] = core.boolean(pair['exact'])
    pairs, report = perceptual.verify(rows, pairs, cfg['train_root'], cfg['perceptual'],
        lambda done, total: print(json.dumps(dict(stage='perceptual', done=done, total=total)), flush=True),
        max_seconds=cfg['perceptual_max_seconds'])
    fields = list(pairs[0]) if pairs else ['i', 'j', 'image_i', 'image_j', 'label_i', 'label_j', 'cosine_similarity',
        'content_group_i', 'content_group_j', 'mutual_nn', 'exact', 'phash_hamming', 'thumbnail_rmse',
        'log_aspect_ratio', 'perceptual_pass', 'perceptual_error']
    io.write_csv(directory / 'verified_pairs.csv.gz', pairs, fields)
    dump(directory / 'perceptual.json', dict(report, policy=cfg['perceptual'],
         candidate_sha256=sha(directory / 'candidates.json'), verified_sha256=sha(directory / 'verified_pairs.csv.gz')))
    return report


def verified_context(directory):
    directory, meta, cfg, rows, split = candidate_context(directory)
    p = json.loads((directory / 'perceptual.json').read_text())
    require(p['candidate_sha256'] == sha(directory / 'candidates.json') and
            p['verified_sha256'] == sha(directory / 'verified_pairs.csv.gz'), 'Perceptual binding changed')
    return directory, meta, cfg, rows, split, io.read_csv(directory / 'verified_pairs.csv.gz')


def audit(directory):
    directory, meta, cfg, rows, split, pairs = verified_context(directory)
    require(not (directory / 'report.json').exists(), 'Refusing to overwrite A0 report')
    groups = defaultdict(list)
    for row in rows:
        groups[row['content_group']].append(row)
    exact = [rs for rs in groups.values() if len(rs) > 1]
    exact_conflicts = [rs for rs in exact if len({r['label'] for r in rs}) > 1]
    curve = core.sweep(rows, pairs)
    io.write_csv(directory / 'similarity_sweep.csv', curve, list(curve[0]))
    signals_table, sources = signals.load(cfg, rows, meta['inputs'])
    io.write_csv(directory / 'existing_signals.csv.gz',
                 [dict(image_path=k, **v) for k, v in signals_table.items()], ['image_path'] + list(next(iter(signals_table.values()))))
    report = dict(status='audit_complete_thresholds_pending', training_started=False, test_read=False,
                  total_images=len(rows), exact_duplicate_images=sum(map(len, exact)), exact_duplicate_groups=len(exact),
                  exact_cross_label_groups=len(exact_conflicts), exact_cross_label_images=sum(map(len, exact_conflicts)),
                  exact_cross_label_pairs=sum((len(rs) * (len(rs) - 1) - sum(n * (n - 1) for n in Counter(r['label'] for r in rs).values())) // 2 for rs in exact_conflicts),
                  near_duplicate_candidate_images=len({int(p[k]) for p in pairs if not core.boolean(p['exact']) for k in ('i', 'j')}),
                  strong_conflict_images=None, medium_conflict_images=None, affected_classes=None,
                  note='Strong/medium and class/cluster statistics require one-time threshold freeze; no labels adjudicated',
                  perceptual_policy=cfg['perceptual'], similarity_curve=curve,
                  oof_probability_known=sum(s['oof_label_probability'] is not None for s in signals_table.values()),
                  trust_definition='V1 low trust = original label not retained by CL; not a ground-truth label judgment',
                  trusted_val_definition='pre-ND P0 weak-consistent and snapshot-confidence-passes proxy; not clean truth',
                  files={n: sha(directory / n) for n in ('similarity_sweep.csv', 'existing_signals.csv.gz', 'verified_pairs.csv.gz', 'perceptual.json', 'candidates.json')},
                  inputs={**meta['inputs'], **sources})
    dump(directory / 'report.json', report)
    return report


def freeze(directory, output, medium, strong, rationale):
    directory, meta, cfg, rows, split, pairs = verified_context(directory)
    report = json.loads((directory / 'report.json').read_text())
    io.verify_files(report['inputs'])
    for name, digest in report['files'].items():
        require(sha(directory / name) == digest, 'Changed A0 report asset')
    thresholds = {round(.94 + i * .005, 3) for i in range(12)}
    require(medium in thresholds and strong in thresholds and medium <= strong and rationale.strip(),
            'Choose ordered sweep-grid thresholds and record curve rationale')
    # Global lock belongs to this A0 audit, independent of requested output path.
    lock = directory / 'frozen_thresholds.json'
    require(not lock.exists(), 'A0 thresholds already frozen; no post-result retuning')
    output = io.fresh(output)
    frozen = dict(medium=medium, strong=strong, rationale=rationale, audit=str(directory),
                  audit_report_sha256=sha(directory / 'report.json'), output=str(output),
                  weights=core.LEVEL_WEIGHT, cap=.10, policy='edge_minimum; strongest_samples_first; no_chaining')
    with lock.open('x') as f:
        json.dump(frozen, f, ensure_ascii=False, indent=2)
    signal_rows = io.read_csv(directory / 'existing_signals.csv.gz')
    signal_table = {}
    for r in signal_rows:
        signal_table[r['image_path']] = {k: (None if not v else core.boolean(v) if k in ('v1_low_trust', 'prototype_agreement', 'oof_ridge_agreement', 'trusted_val_proxy') else float(v))
                                        for k, v in r.items() if k != 'image_path'}
    full = core.manifest(rows, pairs, medium, strong)
    dev = core.manifest(rows, pairs, medium, strong, eligible=split['train'])
    control = [dict(r, nd_level='weak', nd_weight=1.0, reason='paired_control_all_one') for r in full]
    files = {}
    for name, table in (('ndcw_weights.csv', full), ('ndcw_dev_weights.csv', dev), ('control_weights.csv', control)):
        io.write_csv(output / name, table, core.FIELDS)
        files[name] = sha(output / name)
    stats = {}
    for scope, table, eligible in (('full', full, None), ('dev', dev, split['train'])):
        impacts = core.impact(rows, table, eligible)
        name = f'{scope}_class_impact.csv'
        io.write_csv(output / name, impacts, list(impacts[0]))
        files[name] = sha(output / name)
        stats[scope] = dict(gates=core.gates(rows, table, signal_table, eligible),
                           strong_conflict_images=sum(r['raw_nd_level'] == 'strong' for r in table),
                           medium_conflict_images=sum(r['raw_nd_level'] == 'medium' for r in table),
                           class_cap_restored=sum(r['reason'] == 'class_safety_cap_restored' for r in table),
                           max_per_class_affected_rate=max(r['affected_rate'] for r in impacts),
                           max_per_class_supervision_loss_rate=max(r['supervision_loss_rate'] for r in impacts),
                           cross_signals=signals.cross_report(rows, table, signal_table))
        for stratum in ('head', 'middle', 'tail'):
            subset = [r for r in impacts if r['stratum'] == stratum]
            count = sum(r['samples'] for r in subset)
            stats[scope][stratum + '_affected_rate'] = sum(r['affected'] for r in subset) / count if count else None
    conflict = [dict(p, nd_level=core.classify(p, rows, medium, strong)) for p in pairs
                if core.classify(p, rows, medium, strong) != 'weak']
    conflict_fields = list(pairs[0]) + ['nd_level'] if pairs else ['i', 'j', 'nd_level']
    io.write_csv(output / 'conflict_pairs.csv', conflict, conflict_fields)
    files['conflict_pairs.csv'] = sha(output / 'conflict_pairs.csv')
    dev_impact = core.impact(rows, dev, split['train'])
    groups = dict(tail_classes=[r['label'] for r in dev_impact if r['stratum'] == 'tail'],
                  affected_classes=[r['label'] for r in dev_impact if r['affected']],
                  trusted_val_indices=[i for i in split['val'] if signal_table[rows[i]['image_path']]['trusted_val_proxy']],
                  val_paths=[rows[i]['image_path'] for i in split['val']],
                  trusted_val_paths=[rows[i]['image_path'] for i in split['val'] if signal_table[rows[i]['image_path']]['trusted_val_proxy']],
                  val_indices=split['val'])
    dump(output / 'evaluation_groups.json', groups)
    files['evaluation_groups.json'] = sha(output / 'evaluation_groups.json')
    training_allowed = stats['full']['gates']['training_allowed'] and stats['dev']['gates']['training_allowed']
    train_set, val_set = set(split['train']), set(split['val'])
    crossing = [p for p in conflict if (int(p['i']) in train_set and int(p['j']) in val_set)
                or (int(p['j']) in train_set and int(p['i']) in val_set)]
    final = dict(status='frozen_eligible_for_probe' if training_allowed else 'closed_or_missing_evidence',
                 training_allowed=training_allowed, training_started=False, data_version='20260921',
                 total_images=len(rows), thresholds=frozen, stats=stats, clusters=core.clusters(rows, pairs, medium, strong),
                 cross_split_conflict_edges=len(crossing),
                 val_images_linked_to_train=len({int(p[k]) for p in crossing for k in ('i','j') if int(p[k]) in val_set}),
                 files=files, inputs=report['inputs'], audit_report=str(directory / 'report.json'),
                 audit_report_sha256=sha(directory / 'report.json'), freeze_file=str(lock), freeze_sha256=sha(lock))
    dump(output / 'manifest.json', final)
    return final
