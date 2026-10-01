"""Independently recount predictions/groups, draw pairing, and both package checks."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from aegis_clip.runtime import atomic_json_dump, sha256_file
from strong_aug_pair.source import ROOT
from strong_aug_pair.runtime import verify, check_draws
from strong_aug_pair.core import ARMS, require


def verify_report(plan_path):
    plan, source = verify(plan_path)
    output = Path(plan_path).resolve().parent
    report = json.loads((output / 'report.json').read_text())
    for name, expected in report['artifact_sha256'].items():
        require(sha256_file(output / name) == expected, f'Artifact checksum: {name}')
    frozen = json.loads((output / 'frozen_groups.json').read_text())
    with (output / 'frozen_groups.csv').open() as handle:
        all_groups = list(csv.DictReader(handle))
    training_maxima = np.array([float(row['max_cosine']) for row in all_groups if row['partition']=='train_dev'])
    require(len(training_maxima) == 133815 and float(np.quantile(training_maxima, .1, method='linear')) == frozen['threshold'],
            'Training-only group-excluded percentile threshold')
    require(all(int(row['target']) == int(float(row['max_cosine']) < frozen['threshold']) for row in all_groups),
            'Frozen strict target threshold')
    require(sha256_file(output / 'frozen_groups.csv') == frozen['group_file_sha256'], 'Frozen groups changed')
    with (output / 'frozen_groups.csv').open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['partition'] == 'val_dev']
    require([r['image_path'] for r in rows] == [r['image_path'] for r in source.val], 'Group alignment')
    labels = np.array([int(r['label']) for r in rows])
    masks = dict(all=np.ones(len(rows), dtype=bool), target=np.array([r['target']=='1' for r in rows]),
                 complement=np.array([r['target']=='0' for r in rows]), tail75=np.array([r['tail75']=='1' for r in rows]))
    predictions = {}
    for name, path in [('parent', output / 'baseline_center.npz')] + [
            (arm, output / 'arms' / arm / 'val_primary.npz') for arm in ARMS]:
        data = np.load(path, allow_pickle=False)
        require(np.array_equal(data['labels'], labels) and np.array_equal(data['image_paths'], [r['image_path'] for r in rows]),
                'Prediction alignment')
        require(np.array_equal(data['predictions'], data['logits'].argmax(1)), 'Logit replay')
        predictions[name] = data['predictions']
    require(np.array_equal(predictions['parent'], source.parent_predictions), 'Original parent replay')
    comparisons = [('candidate_vs_control', ARMS[0], ARMS[1]),
                   ('candidate_vs_parent', 'parent', ARMS[1]), ('control_vs_parent', 'parent', ARMS[0])]
    for key, a, b in comparisons:
        for group, mask in masks.items():
            correct_a = (predictions[a] == labels) & mask
            correct_b = (predictions[b] == labels) & mask
            correction, regression = int((~correct_a & correct_b & mask).sum()), int((correct_a & ~correct_b & mask).sum())
            result = report[key][group]
            require(result['corrections'] == correction and result['regressions'] == regression
                    and result['net'] == correction-regression, 'Independent correction/regression recount')
            for field, predicted in [('baseline', predictions[a]), ('candidate', predictions[b])]:
                included = labels[mask]
                classes = np.unique(included)
                macro = float(np.mean([np.mean(predicted[mask][included==c] == c) for c in classes])) if len(classes) else None
                actual = result[field]
                require(actual['correct'] == int(((predicted == labels) & mask).sum()) and actual['rows'] == int(mask.sum()),
                        'Independent population recount')
                require(macro == actual['macro'] or (macro is not None and abs(macro-actual['macro']) < 1e-12),
                        'Independent macro recount')
    require(check_draws(output) == report['draws'], 'Independent draw audit')
    require(report['draws']['paired_updates'] == 4*plan['logical_batches_per_epoch'], 'Four epochs of paired updates')
    for arm in ARMS:
        status = json.loads((output / 'arms' / arm / 'status.json').read_text())
        require(status['status'] == 'four_epochs_complete' and status['epochs'] == 4
                and status['updates'] == 4*plan['logical_batches_per_epoch'], 'Complete four-epoch arm')
    for arm in ARMS:
        package = output / 'arms' / arm / 'submission'
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/check_submission.py'),
            '--test_dir', str(source.test_root), '--class-mapping', str(source.stage / 'class_to_idx.json'),
            '--csv', str(package / 'pred_results.csv'), '--zip', str(package / 'submission.zip')], capture_output=True, text=True)
        (package / 'independent_submission_check.log').write_text(result.stdout+result.stderr, encoding='utf-8')
        result.check_returncode()
        require(sha256_file(package / 'submission.zip') == report['exports'][arm]['zip_sha256'], 'Package checksum')
    validation = dict(status='completed_verified', epochs_per_arm=4, parent_replay_rows=14880,
                      packages=2, submission_checks_per_package=9, paired_draws_exact=True,
                      independent_corrections_regressions_macro_recount=True, platform_score=None)
    atomic_json_dump(validation, output / 'validation.json')
    atomic_json_dump(dict(status='completed_verified', epochs_per_arm=4, training_started=True), output / 'status.json')
    print(json.dumps(validation))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    verify_report(parser.parse_args().plan)
