"""Read-only check of machine-B delivery evidence and both saved decoders."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_delivery(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'artifact_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory) and sha256(path) == expected, 'Copied evidence changed: ' + name)
    verified = json.loads((directory / 'delivery_verification.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    run = Path(verified['source_output'])
    require(sha256(run / 'report.json') == verified['source_report_sha256']
            and sha256(run / 'validation.json') == verified['source_validation_sha256'], 'Source completion evidence changed')
    require(json.loads((run / 'status.json').read_text())['status'] == 'completed_verified', 'Run is incomplete')
    require(verified['status'] == 'completed_verified' and verified['epochs_per_arm'] == 4
            and verified['updates_per_arm'] == 14888, 'Wrong completion protocol')
    require(report['decision'] == verified['decision'] and report['decision']['status'] == 'closed_fixed_recipe',
            'Frozen review decision changed')
    with (run / 'frozen_groups.csv').open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['partition'] == 'val_dev']
    require(sha256(run / 'frozen_groups.csv') == report['frozen_groups']['group_file_sha256'], 'Frozen groups changed')
    labels = np.array([int(row['label']) for row in rows])
    masks = dict(all=np.ones(len(labels), dtype=bool), target=np.array([r['target'] == '1' for r in rows]),
                 complement=np.array([r['target'] == '0' for r in rows]), tail75=np.array([r['tail75'] == '1' for r in rows]))
    predictions = {}
    arms = ('original_augmentation', 'strong_augmentation')
    for arm in arms:
        history = json.loads((directory / arm / 'history.json').read_text())
        require([r['epoch'] for r in history] == [1, 2, 3, 4]
                and [r['optimizer_updates'] for r in history] == [3722, 7444, 11166, 14888], 'Four-epoch history changed')
        package = verified['packages'][arm]
        for key in ('checkpoint', 'csv', 'zip'):
            require(sha256(package[key]) == package[key + '_sha256'], 'Source package changed: ' + arm + '/' + key)
        sidecar = json.loads((directory / arm / 'ema_swa_2_4.sha256.json').read_text())
        require(sidecar['sha256'] == package['checkpoint_sha256'], 'Checkpoint sidecar changed')
        csv_bytes = Path(package['csv']).read_bytes()
        require(len(csv_bytes.decode('utf-8').splitlines()) == 37444, 'Submission population changed')
        with zipfile.ZipFile(package['zip']) as archive:
            require(archive.namelist() == ['pred_results.csv']
                    and archive.read('pred_results.csv') == csv_bytes, 'ZIP/CSV byte mismatch')
        predictions[arm] = {}
        for decoder, name, metric_name in (
                ('center', 'val_primary.npz', 'validation_center_metrics'),
                ('six_views', 'val_six_views.npz', 'validation_six_view_metrics')):
            with np.load(run / 'arms' / arm / name, allow_pickle=False) as data:
                require(np.array_equal(data['labels'], labels)
                        and np.array_equal(data['image_paths'], [r['image_path'] for r in rows]), 'Prediction alignment changed')
                predicted = data['logits'].argmax(1)
                require(np.array_equal(predicted, data['predictions']), 'Saved logit/prediction mismatch')
                metric = report['exports'][arm][metric_name]
                macro = float(np.mean([np.mean(predicted[labels == c] == c) for c in np.unique(labels)]))
                require(abs(macro - metric['macro']) < 1e-12
                        and int((predicted == labels).sum()) == metric['correct']
                        and float(np.mean(predicted == labels)) == metric['micro'], 'Independent metric recount differs')
                predictions[arm][decoder] = predicted.copy()
    for decoder in ('center', 'six_views'):
        a, b = [predictions[arm][decoder] for arm in arms]
        for name, mask in masks.items():
            corrections = int(((a != labels) & (b == labels) & mask).sum())
            regressions = int(((a == labels) & (b != labels) & mask).sum())
            actual = dict(rows=int(mask.sum()), corrections=corrections, regressions=regressions, net=corrections - regressions)
            require(actual == verified['paired_candidate_vs_control'][decoder][name], 'Independent paired recount differs')
            if decoder == 'center':
                require(all(actual[key] == report['candidate_vs_control'][name][key]
                            for key in ('corrections', 'regressions', 'net')), 'Frozen center comparison differs')
    return dict(status='completed_verified', copied_evidence_files=len(manifest['files']),
                independently_recounted_logit_rows=59520, packages=2, zip_csv_bytes_exact=True,
                frozen_decision=report['decision']['status'], platform_score=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', required=True)
    print(json.dumps(verify_delivery(parser.parse_args().delivery)))
