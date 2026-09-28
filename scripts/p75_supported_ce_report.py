#!/usr/bin/env python3
"""All-val fixed decode, paired outcomes, and groups by training CE support count."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from p75_supported_ce import (CONFIG, FIXED, OUT, RUN, canonical, preflight, read_json,
                              read_rows, sha, write_json, write_rows)


def grouped_report(labels, baseline, candidate, support_labels, num_classes):
    labels, baseline, candidate = [np.asarray(value, dtype=np.int64)
                                   for value in (labels, baseline, candidate)]
    if labels.shape != baseline.shape or labels.shape != candidate.shape or labels.ndim != 1:
        raise ValueError('Validation predictions do not align')
    if any(((value < 0) | (value >= num_classes)).any()
           for value in (labels, baseline, candidate)):
        raise ValueError('Validation labels/predictions outside mapping')
    total = np.bincount(labels, minlength=num_classes)
    if not np.all(total > 0):
        raise ValueError('Full validation must cover every class')
    base_ok, new_ok = baseline == labels, candidate == labels
    before = np.bincount(labels[base_ok], minlength=num_classes)
    after = np.bincount(labels[new_ok], minlength=num_classes)
    corrected, regressed = ~base_ok & new_ok, base_ok & ~new_ok
    corrections = np.bincount(labels[corrected], minlength=num_classes)
    regressions = np.bincount(labels[regressed], minlength=num_classes)
    support_labels = np.asarray(support_labels, dtype=np.int64)
    if ((support_labels < 0) | (support_labels >= num_classes)).any():
        raise ValueError('Training support labels outside mapping')
    support = np.bincount(support_labels, minlength=num_classes)
    by_class = [dict(class_id=index, restored_training_samples=int(support[index]),
        validation_samples=int(total[index]), baseline_recall=float(before[index]/total[index]),
        candidate_recall=float(after[index]/total[index]),
        delta_recall=float((after[index]-before[index])/total[index]),
        corrections=int(corrections[index]), regressions=int(regressions[index]))
        for index in range(num_classes)]
    groups = []
    # Fixed report buckets; no class filtering by model confidence or outcomes.
    for low, high in read_json(FIXED)['recall_groups']:
        mask = (support >= low) & (support <= high if high is not None else True)
        classes = np.flatnonzero(mask)
        count = int(total[mask].sum())
        groups.append(dict(restored_training_count_range=[low, high], classes=classes.tolist(),
            class_count=len(classes), restored_training_samples=int(support[mask].sum()),
            validation_samples=count,
            baseline_macro_recall=float((before[mask]/total[mask]).mean()) if len(classes) else None,
            candidate_macro_recall=float((after[mask]/total[mask]).mean()) if len(classes) else None,
            delta_macro_recall=float(((after[mask]-before[mask])/total[mask]).mean()) if len(classes) else None,
            baseline_micro_recall=float(before[mask].sum()/count) if count else None,
            candidate_micro_recall=float(after[mask].sum()/count) if count else None,
            corrections=int(corrections[mask].sum()), regressions=int(regressions[mask].sum())))
    baseline_metrics = dict(macro=float((before/total).mean()), micro=float(base_ok.mean()))
    candidate_metrics = dict(macro=float((after/total).mean()), micro=float(new_ok.mean()))
    gain = candidate_metrics['macro']-baseline_metrics['macro']
    micro_ok = candidate_metrics['micro'] >= baseline_metrics['micro']
    gates = read_json(FIXED)['gates']
    return dict(baseline=baseline_metrics, candidate=candidate_metrics,
        delta_macro=gain, delta_micro=candidate_metrics['micro']-baseline_metrics['micro'],
        corrections=int(corrected.sum()), regressions=int(regressed.sum()),
        net_correct=int(corrected.sum()-regressed.sum()), by_class=by_class, groups=groups,
        package_gate_passed=bool(gain >= gates['package_macro_gain']-1e-12 and micro_ok),
        priority_gate_passed=bool(gain >= gates['priority_macro_gain']-1e-12 and micro_ok),
        main_metric_scope='all val_dev; labels from noisy training pool, not clean test accuracy',
        groups_defined_by='restored train_dev sample count; not validation confidence',
        test_used_for_scoring=False)


def predictions(payload):
    from aegis_clip.prior_alignment import fit_prior_bias, apply_prior_bias
    from aegis_clip.tta import fuse_paired_logits
    recipe = read_json(FIXED)['decode']
    first, second = payload['original_logits'], payload['flip_logits']
    if (first.ndim != 2 or first.shape != second.shape or
            not torch.isfinite(first).all() or not torch.isfinite(second).all()):
        raise ValueError('Invalid validation branch logits')
    logits = fuse_paired_logits(payload['original_logits'].float(), payload['flip_logits'].float(),
        mode=recipe['fusion'], temperature=recipe['temperature'])
    bias, _ = fit_prior_bias(logits)
    return apply_prior_bias(logits, bias, strength=recipe['prior_strength']).argmax(1).numpy()


def report(candidate_cache, evaluation, output):
    saved = preflight(require_support=True)
    fixed = read_json(FIXED)
    reference_path = Path(fixed['reference_validation_cache'])
    if sha(reference_path) != saved['identity']['reference_cache_sha256']:
        raise ValueError('Pinned baseline validation cache changed')
    baseline = torch.load(reference_path, map_location='cpu', weights_only=False)
    candidate = torch.load(candidate_cache, map_location='cpu', weights_only=False)
    validation = read_rows(Path(fixed['assets'])/'val_dev.csv')
    paths = [canonical(row['image_path']) for row in validation]
    labels = np.asarray([int(row['label']) for row in validation])
    for payload in (baseline, candidate):
        if (payload['paths'] != paths or
                not np.array_equal(torch.as_tensor(payload['labels']).numpy(), labels) or
                payload['validation_csv_sha256'] != saved['identity']['split_hashes']['val_dev.csv']):
            raise ValueError('Frozen validation cache path/label/split mismatch')
    if baseline['checkpoint_sha256'] != saved['identity']['control_sha256']:
        raise ValueError('Baseline is not pinned L05')
    checkpoint = RUN/'checkpoints/best.pt'
    if candidate['checkpoint_sha256'] != sha(checkpoint):
        raise ValueError('Candidate validation cache checkpoint mismatch')
    support = read_rows(OUT/'selection/support.csv')
    admitted = [int(row['label']) for row in support if float(row['support_weight']) == .5]
    base_prediction, new_prediction = predictions(baseline), predictions(candidate)
    result = grouped_report(labels, base_prediction, new_prediction, admitted,
                            saved['identity']['num_classes'])
    measured = read_json(evaluation)
    if measured['checkpoint_sha256'] != sha(checkpoint) or measured['training_config_sha256'] != sha(CONFIG):
        raise ValueError('Candidate evaluation checkpoint/config mismatch')
    if any(abs(result['candidate'][key]-measured['decode'][key]) > 1e-6 for key in ('macro', 'micro')):
        raise ValueError('Independent fixed decode disagrees with candidate evaluation')
    fit = []
    for path in sorted((RUN/'logs').glob('supported_ce_epoch_*.json'),
                       key=lambda p: int(p.stem.rsplit('_', 1)[1])):
        stats = read_json(path)
        count = stats['examples']
        fit.append(dict(epoch=stats['epoch'], loss_phase=stats['loss_phase'], examples=count,
            accuracy=stats['correct']/count, mean_label_probability=stats['label_probability_sum']/count,
            mean_base_loss=stats['base_loss_sum']/count, mean_ce_loss=stats['ce_loss_sum']/count,
            mean_blended_loss=stats['blended_loss_sum']/count))
    result.update(training_fit=fit, training_fit_is_not_generalization=True,
        identity=saved['identity'], support_count=len(admitted), decode=fixed['decode'],
        binding=dict(candidate_cache_sha256=sha(candidate_cache), evaluation_sha256=sha(evaluation),
            reference_cache_sha256=sha(reference_path), checkpoint_sha256=sha(checkpoint),
            config_sha256=sha(CONFIG), selection_manifest_sha256=sha(OUT/'selection/manifest.json')))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    class_path = output.with_name(output.stem+'_classes.csv')
    row_path = output.with_name(output.stem+'_rows.csv')
    write_rows(class_path, list(result['by_class'][0]), result['by_class'])
    row_records = [dict(image_path=row['image_path'], label=int(labels[index]),
        baseline_prediction=int(base_prediction[index]), candidate_prediction=int(new_prediction[index]),
        corrected=bool(base_prediction[index] != labels[index] == new_prediction[index]),
        regressed=bool(base_prediction[index] == labels[index] != new_prediction[index]))
        for index, row in enumerate(validation)]
    write_rows(row_path, list(row_records[0]), row_records)
    result['report_files'] = {str(path): sha(path) for path in (class_path, row_path)}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-cache', required=True)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = report(args.candidate_cache, args.evaluation, args.output)
    print({key: result[key] for key in ('baseline', 'candidate', 'corrections', 'regressions',
                                       'package_gate_passed', 'priority_gate_passed')})


if __name__ == '__main__':
    main()
