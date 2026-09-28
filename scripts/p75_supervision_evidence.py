#!/usr/bin/env python3
"""P0: immutable, group-excluded intervention evidence; never clean-label truth."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import time
from pathlib import Path
import numpy as np
import torch
from p75_supported_ce import (asset_identity, assert_alignment, canonical, read_json,
    read_rows, sha, verify_files, weak_metrics, write_json, write_rows, FIXED as SOURCE_FIXED)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/codex/p75_supervision_rebuild_20260929'
FIXED = ROOT/'configs/p75_supervision_rebuild_20260929/fixed.json'


def context():
    rule, source = read_json(FIXED), read_json(SOURCE_FIXED)
    identity = asset_identity()
    assets = Path(source['assets'])
    train, val = read_rows(assets/'train_dev.csv'), read_rows(assets/'val_dev.csv')
    if sha(rule['oof_path']) != rule['oof_sha256']:
        raise ValueError('Current-stage OOF evidence changed')
    oof = read_rows(rule['oof_path'])
    assert_alignment(train, oof)
    folds = defaultdict(set)
    for row in oof:
        if int(row['fold']) not in range(5):
            raise ValueError('Unexpected OOF fold')
        folds[row['content_group']].add(row['fold'])
    if any(len(values) != 1 for values in folds.values()):
        raise ValueError('OOF content group crosses folds')
    identity.update(oof_sha256=rule['oof_sha256'], rules_sha256=sha(FIXED))
    return rule, source, identity, train, val, oof


def feature_rows(rows, assets):
    assets = Path(assets)
    paths = read_json(assets/'features/image_paths.json')
    index = {canonical(p): i for i, p in enumerate(paths)}
    full = torch.load(assets/'features/features.pt', map_location='cpu', weights_only=True)
    x = full[[index[canonical(r['image_path'])] for r in rows]].float().contiguous()
    if not torch.isfinite(x).all() or not torch.allclose(x.norm(dim=1), torch.ones(len(x)), atol=1e-4):
        raise ValueError('Invalid normalized official features')
    return x


def group_neighbors(bank, training, queries, query_rows, *, k=20, batch=128, deadline=None):
    """Exact row cosine then distinct groups. Stable score/row-order ties.

    Top-k row pool doubles until k independent groups exist; include all boundary
    ties before sorting. Only train_dev is in the bank, even for val queries.
    """
    if bank.ndim != 2 or queries.ndim != 2 or bank.shape[1] != queries.shape[1]:
        raise ValueError('Feature shapes do not align')
    if len(bank) != len(training) or len(queries) != len(query_rows):
        raise ValueError('Feature rows do not align')
    group_labels = defaultdict(set)
    for row in training:
        group_labels[row['content_group']].add(int(row['label']))
    group_names = sorted(group_labels)
    group_ids = {name: i for i, name in enumerate(group_names)}
    gids = np.array([group_ids[r['content_group']] for r in training])
    for offset in range(0, len(queries), batch):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError('Fixed P0 neighbor budget exhausted; no completed manifest')
        scores = (queries[offset:offset+batch] @ bank.T).numpy()
        for j, row_scores in enumerate(scores):
            query = query_rows[offset+j]
            excluded = group_ids.get(query['content_group'], -1)
            pool = min(len(training), max(64, 2*k))
            while True:
                candidates = np.argpartition(-row_scores, pool-1)[:pool]
                boundary = row_scores[candidates].min()
                candidates = np.flatnonzero(row_scores >= boundary)
                candidates = candidates[np.lexsort((candidates, -row_scores[candidates]))]
                seen, chosen = {excluded}, []
                for idx in candidates:
                    group = int(gids[idx])
                    if group in seen:
                        continue
                    seen.add(group)
                    chosen.append(int(idx))
                    if len(chosen) == k:
                        break
                if len(chosen) == k or pool == len(training):
                    break
                pool = min(len(training), 2*pool)
            votes = Counter()
            neighbors = []
            for idx in chosen:
                group = training[idx]['content_group']
                labels = sorted(group_labels[group])
                if len(labels) == 1:
                    votes[labels[0]] += 1
                neighbors.append(dict(training_row=idx, content_group=group,
                    labels=labels, cosine=float(row_scores[idx])))
            yield offset+j, votes, neighbors


def neighbors():
    rule, source, identity, train, val, _ = context()
    destination = OUT/'neighbors'
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.monotonic()
    all_rows = train+val
    x = feature_rows(all_rows, source['assets'])
    with (destination/'groups.jsonl').open('x') as stream:
        for i, votes, group_rows in group_neighbors(x[:len(train)], train, x, all_rows,
                k=rule['neighbors'], deadline=start+rule['neighbor_wall_seconds']):
            stream.write(json.dumps(dict(image_path=all_rows[i]['image_path'],
                label=int(all_rows[i]['label']), content_group=all_rows[i]['content_group'],
                votes=dict(votes), neighbors=group_rows), separators=(',', ':'))+'\n')
            if (i+1) % 4096 == 0:
                print(f'P0 exact neighbors {i+1}/{len(x)} elapsed={time.monotonic()-start:.1f}s', flush=True)
    write_json(destination/'manifest.json', dict(identity=identity, rows=len(all_rows),
        seconds=time.monotonic()-start, bank='train_dev only; unique content-group votes',
        files={'groups.jsonl': sha(destination/'groups.jsonl')}, test_used=False))


def freeze_confusions(labels, predictions, num_classes, rule):
    counts = Counter((int(a), int(b)) for a, b in zip(labels, predictions) if a != b)
    edges = sorted((a, b) for a, b in counts if a < b and
        min(counts[a,b], counts[b,a]) >= rule['confusion_min_each_direction'] and
        counts[a,b]+counts[b,a] >= rule['confusion_min_total'])
    parent = list(range(num_classes))
    def find(a):
        while a != parent[a]:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a,b in edges:
        a,b = find(a),find(b)
        parent[max(a,b)] = min(a,b)
    components = defaultdict(list)
    for a in range(num_classes):
        components[find(a)].append(a)
    groups = [v for _,v in sorted(components.items()) if len(v)>1]
    memberships = {c:i for i,g in enumerate(groups) for c in g}
    return edges, groups, memberships


def channel_for(label, votes, oof_pair, weak_pair, rule):
    """Fixed external reference required; self-confidence never grants a label."""
    original_votes = votes.get(label, 0)
    ranked = sorted(votes, key=lambda c: (-votes[c], c))
    alternative = ranked[0] if ranked else None
    original = original_votes >= rule['original_support_votes'] and oof_pair == (label,label)
    if original:
        return 'original', None
    alternate = (alternative is not None and alternative != label and
        votes[alternative] >= rule['alternative_support_votes'] and
        oof_pair == (alternative,alternative) and weak_pair == (alternative,alternative))
    return ('soft', alternative) if alternate else ('uncertain', None)


def lp_proxy(training, source):
    """Reproduce the 2,139-row RM-LP proxy; never call it L05 evidence."""
    from aegis_clip.npu_checkpoint_compat import ensure_npu_checkpoint_stubs
    ensure_npu_checkpoint_stubs()
    checkpoint = torch.load(source['parent_checkpoint'], map_location='cpu', weights_only=False)
    weights = checkpoint['model_state_dict']['classifier.weight'].float()
    bias = checkpoint['model_state_dict']['classifier.bias'].float()
    del checkpoint
    features = feature_rows(training, source['assets'])
    labels = torch.tensor([int(r['label']) for r in training])
    py, maximum = [], []
    for start in range(0, len(training), 1024):
        p = (features[start:start+1024] @ weights.T + bias).softmax(1)
        py.extend(p[torch.arange(len(p)), labels[start:start+len(p)]].tolist())
        maximum.extend(p.max(1).values.tolist())
    return py, maximum


def paired_counts(labels, baseline, candidate, mask=None):
    labels, baseline, candidate = map(np.asarray, (labels,baseline,candidate))
    if mask is None:
        mask = np.ones(len(labels), bool)
    a,b = baseline == labels,candidate == labels
    corrected,regressed = int((mask & ~a & b).sum()),int((mask & a & ~b).sum())
    classes = np.unique(labels[mask])
    macro = lambda ok: float(np.mean([ok[mask & (labels==c)].mean() for c in classes])) if len(classes) else None
    return dict(samples=int(mask.sum()), corrections=corrected, regressions=regressed,
        net=corrected-regressed, baseline_micro=float(a[mask].mean()) if mask.any() else None,
        candidate_micro=float(b[mask].mean()) if mask.any() else None,
        baseline_macro=macro(a), candidate_macro=macro(b))


def diagnose(snapshot=None):
    rule, source, identity, train, val, oof = context()
    destination = OUT/('diagnosis' if snapshot else 'diagnosis_without_snapshot')
    destination.mkdir(parents=True, exist_ok=False)
    neighbor_dir = OUT/'neighbors'
    manifest = read_json(neighbor_dir/'manifest.json')
    if manifest['identity'] != identity:
        raise ValueError('Neighbor evidence source changed')
    verify_files(neighbor_dir, manifest['files'])
    torch.set_num_threads(4)
    # Import the exact fixed decoder; this fits val-only prior, never test data.
    from p75_supported_ce_report import predictions
    payload = torch.load(source['reference_validation_cache'], map_location='cpu', weights_only=False)
    if (payload['paths'] != [canonical(r['image_path']) for r in val] or
        not np.array_equal(np.asarray(payload['labels']), [int(r['label']) for r in val]) or
        payload['validation_csv_sha256'] != identity['split_hashes']['val_dev.csv'] or
        payload['checkpoint_sha256'] != identity['control_sha256']):
        raise ValueError('L05 validation cache identity mismatch')
    val_pred = predictions(payload)
    val_metrics = weak_metrics(payload['original_logits'].float().softmax(1).numpy(),
        payload['flip_logits'].float().softmax(1).numpy(), [int(r['label']) for r in val])
    train_metrics = None
    if snapshot:
        snapshot = Path(snapshot)
        snap = read_json(snapshot/'manifest.json')
        if (snap['identity']['control_sha256'] != identity['control_sha256'] or
            snap['identity']['split_hashes'] != identity['split_hashes'] or
            snap['temperature'] != 1 or snap['prior_applied'] or snap['tta_fusion_applied'] or
            snap['views'] != ['center_384','horizontal_flip_384']):
            raise ValueError('Snapshot identity/protocol mismatch')
        verify_files(snapshot, snap['files'])
        assert_alignment(train, read_rows(snapshot/'rows.csv'))
        first = np.load(snapshot/'center_probabilities.npy', mmap_mode='r')
        second = np.load(snapshot/'flip_probabilities.npy', mmap_mode='r')
        if first.shape != (len(train),identity['num_classes']) or second.shape != first.shape:
            raise ValueError('Snapshot shape mismatch')
        parts = [weak_metrics(first[i:i+512],second[i:i+512],
            [int(r['label']) for r in train[i:i+512]]) for i in range(0,len(train),512)]
        train_metrics = {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    classes = identity['num_classes']
    lp_py, lp_max = lp_proxy(train, source)
    own_group_labels = defaultdict(set)
    for row in train+val:
        own_group_labels[row['content_group']].add(int(row['label']))
    raw = Counter(int(r['label']) for r in train)
    independent = defaultdict(set)
    trusted_groups = defaultdict(set)
    hard_groups = defaultdict(set)
    for row in train:
        independent[int(row['label'])].add(row['content_group'])
    labels = np.array([int(r['label']) for r in val])
    edges, groups, memberships = freeze_confusions(labels, val_pred, classes, rule)
    edge_set = set(edges)
    records, channels = [], Counter()
    with (neighbor_dir/'groups.jsonl').open() as stream:
        for i,row in enumerate(train+val):
            neighbor = json.loads(next(stream))
            assert_alignment([row],[neighbor])
            votes = {int(k):v for k,v in neighbor['votes'].items()}
            is_train = i < len(train)
            j = i if is_train else i-len(train)
            metrics = train_metrics if is_train else val_metrics
            label = int(row['label'])
            scored = oof[i] if is_train else None
            pair = (int(scored['centroid_top1']),int(scored['ridge_top1'])) if scored else None
            weak_pair = (int(metrics['center_top1'][j]),int(metrics['flip_top1'][j])) if metrics else None
            channel,alt = channel_for(label,votes,pair,weak_pair,rule) if is_train else ('diagnostic_only',None)
            group_conflict = len(own_group_labels[row['content_group']]) > 1
            if is_train and (group_conflict or len(neighbor['neighbors']) != rule['neighbors']):
                channel,alt = 'uncertain',None
            original_supported = (channel == 'original' if is_train else
                votes.get(label,0)>=rule['original_support_votes'] and not group_conflict)
            p = sorted(votes,key=lambda c:(-votes[c],c))
            alternate = p[0] if p and p[0] != label and votes[p[0]] >= rule['alternative_support_votes'] else None
            record = dict(split='train' if is_train else 'val', **{k:row[k] for k in ('image_path','content_group')},
                original_label=label, channel=channel, soft_alternative=alt,
                original_supported=original_supported, external_alternative=alternate,
                independent_neighbor_groups=len(neighbor['neighbors']),
                original_support_votes=votes.get(label,0),
                neighbor_votes=json.dumps(votes, sort_keys=True, separators=(',',':')),
                oof_centroid_prediction=pair[0] if pair else None,
                oof_ridge_prediction=pair[1] if pair else None,
                rm_lp_label_probability=lp_py[i] if is_train else None,
                rm_lp_max_probability=lp_max[i] if is_train else None,
                previous_2139_proxy=bool(is_train and pair == (label,label) and lp_py[i] < .3 and lp_max[i] < .7),
                own_content_group_label_conflict=group_conflict,
                weak_consistent=weak_pair[0]==weak_pair[1] if weak_pair else None,
                l05_prediction=weak_pair[0] if weak_pair else None,
                l05_fixed_decode_prediction=int(val_pred[j]) if not is_train else None,
                class_raw_samples=raw[label], class_independent_content_groups=len(independent[label]),
                confusion_group=memberships.get(label),
                training_epoch_label_probability=None, training_epoch_max_probability=None,
                training_epoch_local_enabled=None,
                history_status='missing_per_sample_history',
                weak_snapshot_status='final_checkpoint_inference' if metrics else 'missing',
                appearance_coverage='unknown; no independent source/appearance annotations')
            for key in ('center_label_probability','flip_label_probability','center_max_probability',
                        'flip_max_probability','distribution_l1','distribution_jsd'):
                record[key] = float(metrics[key][j]) if metrics else None
            record['snapshot_low_original_probability'] = record['center_label_probability'] < rule['weak_label_probability'] if metrics else None
            record['snapshot_low_max_probability'] = record['center_max_probability'] < rule['local_confidence_gate'] if metrics else None
            record['snapshot_confidence_gate_passes'] = not record['snapshot_low_max_probability'] if metrics else None
            record['snapshot_gce_gradient_scale'] = float(np.sqrt(record['center_label_probability'])) if metrics else None
            record['supported_hard'] = bool(original_supported and metrics and (
                record['snapshot_low_original_probability'] or weak_pair[0] != label))
            if is_train:
                channels[channel]+=1
                if original_supported:
                    trusted_groups[label].add(row['content_group'])
                if record['supported_hard']:
                    hard_groups[label].add(row['content_group'])
            records.append(record)
        if next(stream,None) is not None:
            raise ValueError('Unexpected extra neighbor rows')
    for row in records:
        label = row['original_label']
        trusted, hard = len(trusted_groups[label]),len(hard_groups[label])
        row.update(class_trusted_independent_groups=trusted, class_hard_supported_groups=hard)
        # Sparse accepted evidence is a failure of this construction, not proof
        # that the underlying class has no usable supervision.
        row['trusted_reference_coverage_deficit'] = trusted < rule['minimum_trusted_groups']
        row['insufficient_effective_supervision'] = len(independent[label]) < rule['minimum_trusted_groups']
        pred = row['l05_fixed_decode_prediction'] if row['split']=='val' else row['l05_prediction']
        row['stable_trusted_confusion'] = bool(pred is not None and pred != label and row['weak_consistent'] and
            row['l05_prediction'] == pred and
            row['original_supported'] and tuple(sorted((label,pred))) in edge_set and
            min(trusted,len(trusted_groups[pred])) >= rule['minimum_trusted_groups'])
        # Whole-class association is disclosed, never an estimated recoverable count.
        row['hard_supervision_associated'] = bool(row['original_supported'] and hard>=rule['minimum_hard_groups'])
        row['label_dispute'] = row['external_alternative'] is not None
        row['unknown'] = not any(row[k] for k in ('hard_supervision_associated',
            'stable_trusted_confusion','insufficient_effective_supervision','label_dispute'))
    val_records = records[len(train):]
    wrong = val_pred != labels
    flags = ('hard_supervision_associated','stable_trusted_confusion',
             'insufficient_effective_supervision','label_dispute','unknown')
    slices = {flag:[i for i,r in enumerate(val_records) if r[flag]] for flag in flags}
    counts = {k:dict(samples=len(v), errors=int(wrong[v].sum())) for k,v in slices.items()}
    overlap = {a+' & '+b:int(sum(wrong[i] and val_records[i][a] and val_records[i][b]
        for i in range(len(val)))) for j,a in enumerate(flags[:-1]) for b in flags[j+1:-1]}
    undercovered = [c for c in range(classes) if len(trusted_groups[c]) < rule['minimum_groups_for_channel_coverage']]
    coverage_ok = len(undercovered)/classes <= rule['max_undercovered_class_fraction']
    r1 = counts['hard_supervision_associated']['errors']
    r2 = counts['stable_trusted_confusion']['errors']
    # No fabricated effect size: association sizes only admit a bounded pilot.
    if train_metrics is None:
        decision = 'incomplete_P0_missing_L05_snapshot'
    elif not coverage_ok:
        decision = 'supervision_construction_failed_coverage; no_training'
    elif r1 >= rule['primary_error_scale']:
        decision = 'R1_bounded_pilot_first'
    elif r2 >= rule['primary_error_scale']:
        decision = 'R2_bounded_pilot_first'
    else:
        decision = 'R1_single_exploratory_pilot_only; no_full_training'
    write_rows(destination/'sample_evidence.csv',list(records[0]),records)
    trusted_edges=[(a,b) for a,b in edges if min(len(trusted_groups[a]),len(trusted_groups[b]))>=rule['minimum_trusted_groups']]
    write_json(destination/'frozen_slices.json',dict(identity=identity, groups=groups, edges=edges,
        trusted_edges=trusted_edges,
        validation_paths=[r['image_path'] for r in val], baseline_predictions=val_pred.tolist(),
        labels=labels.tolist(), slices=slices, evidence_sha256=sha(destination/'sample_evidence.csv')))
    summary = dict(identity=identity,decision=decision, training_channels=dict(channels),
        training_hard_supported=sum(r['supported_hard'] for r in records[:len(train)]),
        original_supported_classes=sum(bool(v) for v in trusted_groups.values()), undercovered_classes=undercovered,
        zero_original_support_classes=[c for c in range(classes) if not trusted_groups[c]],
        coverage_ok=coverage_ok,baseline=paired_counts(labels,val_pred,val_pred),
        baseline_errors=int(wrong.sum()),slices=counts, overlaps=overlap,
        confusion_groups=len(groups),confusion_edges=len(edges),
        reference_coverage_deficit=dict(classes=sum(len(trusted_groups[c])<rule['minimum_trusted_groups'] for c in range(classes)),
            validation_errors=int(sum(wrong[i] and r['trusted_reference_coverage_deficit'] for i,r in enumerate(val_records))),
            interpretation='insufficient accepted evidence under this fixed rule; not a cause attribution'),
        snapshot_manifest_sha256=sha(snapshot/'manifest.json') if snapshot else None,
        historical_per_sample_gate_available=False,recoverable_fraction=None,
        predicted_net_gain=None,platform_gain=None,full_training_allowed=False,
        files={p.name:sha(p) for p in destination.iterdir() if p.is_file()})
    previous = [r for r in records[:len(train)] if r['previous_2139_proxy']]
    summary['previous_proxy_alignment'] = dict(total=len(previous),
        original_supported=sum(r['original_supported'] for r in previous),
        l05_supported_hard=sum(r['supported_hard'] for r in previous),
        l05_low_original_probability=sum(r['snapshot_low_original_probability'] is True for r in previous) if train_metrics else None,
        l05_low_max_probability=sum(r['snapshot_low_max_probability'] is True for r in previous) if train_metrics else None,
        weak_consistent=sum(r['weak_consistent'] is True for r in previous) if train_metrics else None)
    write_json(destination/'summary.json',summary)
    lines = [f'# P0 主线选择：{decision}', '',
        f'当前 750 类；train={len(train):,}，val={len(val):,}。L05 固定协议错误 {wrong.sum():,} 张。',
        f'训练通道：{dict(channels)}；可信难样本 {summary["training_hard_supported"]:,} 张。',
        '', '|预冻结切片（可重叠）|验证样本|L05 错误|','|---|---:|---:|']
    lines += [f'|{k}|{v["samples"]}|{v["errors"]}|' for k,v in counts.items()]
    lines += ['',f'原标签通道不足 5 个独立组的类别：{len(undercovered)}/{classes}；覆盖门通过={coverage_ok}。',
        'R1 优先，R2 仅在可信混淆规模充分时替代；本轮没有恢复率或净收益实测，不准投入完整训练。',
        'hard_supervision_associated 是“验证原标签获组外支持 + 同类训练有 ≥5 个可信难组”的关联，不能解释为因果或可恢复上限。',
        '标签争议组只记录预测变化；原始验证标签指标不是干净准确率。独立来源和外观覆盖仍未知。',
        '最终中心/Flip 快照不等于训练轨迹；p_y、p_max、历史局部准入分列。缺失轨迹保持空值。',
        '本轮不训练教师、不新增 OOF、不读测试图。已关闭的微调路线保持关闭。',
        '',f'可复现产物：{destination}/sample_evidence.csv；summary.json；frozen_slices.json。']
    (destination/'decision.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ('decision','training_channels','baseline_errors','slices')},indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['preflight','neighbors','diagnose'],required=True)
    parser.add_argument('--snapshot',type=Path)
    args=parser.parse_args()
    if args.phase=='preflight':
        print(json.dumps(context()[2],indent=2))
    elif args.phase=='neighbors':
        neighbors()
    else:
        diagnose(args.snapshot)

if __name__=='__main__':
    main()
