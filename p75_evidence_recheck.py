#!/usr/bin/env python3
"""Read-only P75 cached evidence recheck; no images, models, labels or training configs."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path


MASKS = [''.join(x) for x in itertools.product('01', repeat=3)]
RELATIONS = ['unique_original_and_model', 'unique_original_not_model',
             'unique_model_not_original', 'unique_neither', 'tied', 'no_votes']
SIGNALS = ['neighbor', 'centroid', 'ridge', 'all_three',
           'without_neighbor', 'without_centroid', 'without_ridge']


def digest(path, decompressed=False):
    h = hashlib.sha256()
    opener = gzip.open if decompressed and path.suffix == '.gz' else open
    with opener(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def boolean(value):
    if value not in ('True', 'False'):
        raise ValueError(f'Invalid boolean: {value!r}')
    return value == 'True'


def analyze(path, classes):
    counts = Counter()
    combinations = {m: Counter(rows=0, final_snapshot_hard=0, structural_eligible=0,
                                structural_eligible_hard=0) for m in MASKS}
    combination_classes = {m: set() for m in MASKS}
    combination_groups = {m: set() for m in MASKS}
    signal_counts = {s: Counter(rows=0, final_snapshot_hard=0, structural_eligible=0,
                                structural_eligible_hard=0) for s in SIGNALS}
    signal_classes = {s: set() for s in SIGNALS}
    eligible_classes = {s: set() for s in SIGNALS}
    per_class = [Counter() for _ in range(classes)]
    raw_groups = [set() for _ in range(classes)]
    selected_groups = [set() for _ in range(classes)]
    partitions = {k: Counter(rows=0, errors=0, prior_unknown_rows=0,
                             prior_unknown_errors=0) for k in RELATIONS}
    vote_hist = {k: Counter(rows=0, errors=0) for k in range(21)}
    directed = Counter()
    identities = set()
    required = {'split', 'image_path', 'content_group', 'original_label', 'channel',
                'original_supported', 'independent_neighbor_groups', 'original_support_votes',
                'neighbor_votes', 'oof_centroid_prediction', 'oof_ridge_prediction',
                'own_content_group_label_conflict', 'l05_prediction',
                'l05_fixed_decode_prediction', 'center_label_probability', 'unknown'}
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing columns: {sorted(required - set(reader.fieldnames or []))}')
        for line, row in enumerate(reader, 2):
            try:
                y = int(row['original_label'])
                pred = int(row['l05_prediction'])
                split = row['split']
                if split not in ('train', 'val') or not 0 <= y < classes or not 0 <= pred < classes:
                    raise ValueError('Invalid split/label/prediction')
                identity = (split, row['image_path'])
                if identity in identities or not row['image_path'] or not row['content_group']:
                    raise ValueError('Duplicate/empty sample identity')
                identities.add(identity)
                votes = {int(k): v for k, v in json.loads(row['neighbor_votes']).items()}
                neighbors = int(row['independent_neighbor_groups'])
                if not 0 <= neighbors <= 20 or any(not 0 <= k < classes or type(v) is not int or v < 1 for k, v in votes.items()) or sum(votes.values()) > neighbors:
                    raise ValueError('Invalid independent neighbor votes')
                support = int(row['original_support_votes'])
                if support != votes.get(y, 0):
                    raise ValueError('Original vote count differs from histogram')
                conflict = boolean(row['own_content_group_label_conflict'])
                structural = not conflict and neighbors == 20
                py = float(row['center_label_probability'])
                if not math.isfinite(py) or not 0 <= py <= 1:
                    raise ValueError('Invalid label probability')
                c = per_class[y]
                counts[split] += 1
                c[split + '_rows'] += 1
                if split == 'train':
                    oofs = [int(row[k]) for k in ('oof_centroid_prediction', 'oof_ridge_prediction')]
                    if any(not 0 <= v < classes for v in oofs):
                        raise ValueError('Invalid training OOF prediction')
                    a, b, d = support >= 12, oofs[0] == y, oofs[1] == y
                    mask = ''.join(str(int(v)) for v in (a, b, d))
                    hard = py < .3 or pred != y
                    counts['train_final_snapshot_hard'] += hard
                    counts['train_low_original_probability'] += py < .3
                    counts['train_center_errors'] += pred != y
                    counts['train_structural_excluded'] += not structural
                    counts['train_own_group_conflict'] += conflict
                    counts['train_neighbor_count_not_20'] += neighbors != 20
                    combo = combinations[mask]
                    combo['rows'] += 1
                    combo['final_snapshot_hard'] += hard
                    combo['structural_eligible'] += structural
                    combo['structural_eligible_hard'] += structural and hard
                    combination_classes[mask].add(y)
                    combination_groups[mask].add((y, row['content_group']))
                    flags = (a, b, d, a and b and d, b and d, a and d, a and b)
                    for name, flag in zip(SIGNALS, flags):
                        if flag:
                            signal_counts[name]['rows'] += 1
                            signal_counts[name]['final_snapshot_hard'] += hard
                            signal_counts[name]['structural_eligible'] += structural
                            signal_counts[name]['structural_eligible_hard'] += structural and hard
                            signal_classes[name].add(y)
                            if structural:
                                eligible_classes[name].add(y)
                            c[name + '_rows'] += 1
                    channel = row['channel']
                    accepted = structural and a and b and d
                    if channel not in ('original', 'soft', 'uncertain') or accepted != (channel == 'original') or accepted != boolean(row['original_supported']):
                        raise ValueError('Reconstructed original-channel rule mismatch')
                    counts['channel_' + channel] += 1
                    c['channel_' + channel] += 1
                    raw_groups[y].add(row['content_group'])
                    if accepted:
                        selected_groups[y].add(row['content_group'])
                else:
                    if row['oof_centroid_prediction'] or row['oof_ridge_prediction']:
                        raise ValueError('Validation must not have OOF evidence')
                    pred = int(row['l05_fixed_decode_prediction'])
                    if not 0 <= pred < classes:
                        raise ValueError('Invalid fixed-decode prediction')
                    wrong = pred != y
                    unknown = boolean(row['unknown'])
                    counts['val_errors'] += wrong
                    counts['val_prior_unknown_errors'] += wrong and unknown
                    c['val_errors'] += wrong
                    if wrong:
                        directed[y, pred] += 1
                    leaders = [k for k, v in votes.items() if v == max(votes.values())] if votes else []
                    leader = leaders[0] if len(leaders) == 1 else None
                    if not leaders:
                        category = 'no_votes'
                    elif len(leaders) > 1:
                        category = 'tied'
                    elif leader == y:
                        category = 'unique_original_not_model' if wrong else 'unique_original_and_model'
                    elif leader == pred:
                        category = 'unique_model_not_original'
                    else:
                        category = 'unique_neither'
                    partitions[category]['rows'] += 1
                    partitions[category]['errors'] += wrong
                    partitions[category]['prior_unknown_rows'] += unknown
                    partitions[category]['prior_unknown_errors'] += wrong and unknown
                    vote_hist[support]['rows'] += 1
                    vote_hist[support]['errors'] += wrong
                    if wrong and leader == y and support < 12:
                        counts['val_errors_original_is_plurality_but_below_12_votes_including_conflicts'] += 1
                        if not conflict:
                            counts['val_errors_original_is_plurality_but_below_12_votes'] += 1
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(f'CSV line {line}: {exc}') from exc
    ranked = sorted(directed.items(), key=lambda item: (-item[1], item[0]))
    retained = {(a, b) for a, b in directed if a < b and
                min(directed[a, b], directed[b, a]) >= 2 and directed[a, b] + directed[b, a] >= 6}
    errors = counts['val_errors']
    selected_errors = sum(n for (a, b), n in directed.items() if tuple(sorted((a, b))) in retained)
    pairs, cumulative = [], 0
    for rank, ((a, b), n) in enumerate(ranked, 1):
        cumulative += n
        pairs.append(dict(rank=rank, original_label=a, predicted_label=b, errors=n,
                          reverse_errors=directed[b, a], original_P0_pair_retained=tuple(sorted((a, b))) in retained,
                          cumulative_errors=cumulative, cumulative_fraction=cumulative/errors if errors else 0.0))
    class_rows = []
    for y, c in enumerate(per_class):
        fields = ['train_rows', 'val_rows', 'val_errors', 'channel_original', 'channel_soft', 'channel_uncertain']
        fields += [s + '_rows' for s in SIGNALS]
        class_rows.append(dict(class_id=y, **{name: c[name] for name in fields},
                               raw_content_groups=len(raw_groups[y]), original_channel_groups=len(selected_groups[y])))
    for s in SIGNALS:
        signal_counts[s]['classes'] = len(signal_classes[s])
        signal_counts[s]['structural_eligible_classes'] = len(eligible_classes[s])
    return dict(counts=dict(counts),
        signal_masks=[dict(mask=m, **combinations[m], classes=len(combination_classes[m]),
                           class_content_groups=len(combination_groups[m])) for m in MASKS],
        final_snapshot_hard_by_signal={s: dict(v) for s, v in signal_counts.items()},
        signal_zero_classes={s: sorted(set(range(classes))-signal_classes[s]) for s in SIGNALS},
        validation_neighbor_partitions={k: dict(v) for k, v in partitions.items()},
        validation_original_vote_histogram={str(k): dict(v) for k, v in vote_hist.items()},
        confusion=dict(all_directed_pairs=len(ranked), P0_retained_undirected_pairs=len(retained),
            P0_retained_errors=selected_errors, excluded_by_P0_pair_rule=errors-selected_errors,
            errors_on_pairs_with_no_reverse=sum(n for (a,b), n in directed.items() if directed[b,a] == 0),
            top_directed_pair_cumulative_errors={str(k): sum(n for _, n in ranked[:k]) for k in (10,50,100,200)}),
        classes=class_rows, confusion_pairs=pairs,
        limitations=['Original labels are noisy; agreement is not clean-label ground truth.',
            'OOF exists only for training. Validation uses fixed-decode L05 and existing neighbor votes.',
            'Final-snapshot hard means center p_y < 0.3 OR center top1 != original; it is not historical suppression.',
            'A vote mask tests >=12 votes; structural eligibility separately requires 20 neighbor groups and no own-group conflict.',
            'Signal-only/leave-one-out counts diagnose exclusions; they do not relax training admission.',
            'Class/error overlap and confusion are not causes or attainable improvement bounds.',
            'Missing support is not opposing evidence and cannot justify revoking class supervision.',
            'No new supervision table, candidate configuration, or training permission is produced.'])


def save_csv(path, rows, empty_fields):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else empty_fields)
        writer.writeheader()
        writer.writerows(rows)


def run(source, summary, out, classes=None):
    if out.exists():
        raise ValueError('Output path already exists; inputs are never overwritten')
    reference = json.loads(summary.read_text(encoding='utf-8'))
    identity = reference['identity']
    classes = identity['num_classes'] if classes is None else classes
    if classes != identity['num_classes'] or classes <= 0:
        raise ValueError('Class count differs from P0 summary')
    source_sha = digest(source)
    raw_sha = digest(source, decompressed=True)
    expected = reference['archived_artifacts'].get(source.name) if source.suffix == '.gz' else reference['files']['sample_evidence.csv']
    if expected is None or source_sha != expected or raw_sha != reference['files']['sample_evidence.csv']:
        raise ValueError('Input SHA256 differs from P0 summary or binding is missing')
    result = analyze(source, classes)
    required = {'train': identity['training_samples'], 'val': identity['validation_samples'],
                'val_errors': reference['baseline_errors']}
    if 'slices' in reference:
        required['val_prior_unknown_errors'] = reference['slices']['unknown']['errors']
    for k, n in reference.get('training_channels', {}).items():
        required['channel_' + k] = n
    for key, value in required.items():
        if result['counts'].get(key, 0) != value:
            raise ValueError(f'P0 count mismatch: {key}')
    if 'raw_confusion_edge_errors' in reference and result['confusion']['P0_retained_errors'] != reference['raw_confusion_edge_errors']:
        raise ValueError('P0 filtered confusion count mismatch')
    result.update(input_sha256=source_sha, decompressed_sha256=raw_sha, summary_sha256=digest(summary),
                  script_sha256=digest(Path(__file__)), source=str(source), checked_counts=required)
    class_rows, pairs = result.pop('classes'), result.pop('confusion_pairs')
    out.mkdir(parents=True, exist_ok=False)
    save_csv(out/'class_coverage.csv', class_rows, ['class_id'])
    save_csv(out/'directed_confusions.csv', pairs, ['rank','original_label','predicted_label','errors','reverse_errors','original_P0_pair_retained','cumulative_errors','cumulative_fraction'])
    (out/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# P75 existing-cache recheck', '',
        'CPU cached-table statistics only. Original-label agreement is not label truth.', '',
        '## Signal intersections', 'Mask order: neighbor >=12/20, centroid OOF, ridge OOF.',
        'Hard: final center p_y <0.3 OR center prediction differs from original label.',
        '| Mask | Rows | Hard | Classes | Structurally eligible | Eligible hard |', '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['mask']} | {r['rows']} | {r['final_snapshot_hard']} | {r['classes']} | {r['structural_eligible']} | {r['structural_eligible_hard']} |" for r in result['signal_masks']]
    lines += ['', '## Signal-only / leave-one-out (diagnostic, not training admission)',
        '| Signal | Rows | Hard | Classes | Eligible classes |', '|---|---:|---:|---:|---:|']
    lines += [f"| {k} | {v['rows']} | {v['final_snapshot_hard']} | {v['classes']} | {v['structural_eligible_classes']} |" for k,v in result['final_snapshot_hard_by_signal'].items()]
    lines += ['', '## Validation relationships (mutually exclusive)',
        '| Neighbor relationship | Rows | Errors | Prior unknown errors |','|---|---:|---:|---:|']
    lines += [f"| {k} | {v['rows']} | {v['errors']} | {v['prior_unknown_errors']} |" for k,v in result['validation_neighbor_partitions'].items()]
    lines += ['', 'Unique-original plurality below 12 votes, wrong L05, no own-group conflict: ' + str(result['counts'].get('val_errors_original_is_plurality_but_below_12_votes', 0)),
        '', '## Confusion coverage', '```json', json.dumps(result['confusion'],indent=2), '```', '', '## Limits']
    lines += ['- '+s for s in result['limitations']]
    lines += ['', 'Input SHA256: `'+source_sha+'`', 'All summary count/hash checks passed.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--classes', type=int, default=None)
    args = parser.parse_args()
    result = run(args.input, args.summary, args.out, args.classes)
    print(json.dumps({'out':str(args.out), 'counts':result['counts'], 'confusion':result['confusion']}, indent=2))


if __name__ == '__main__':
    main()
