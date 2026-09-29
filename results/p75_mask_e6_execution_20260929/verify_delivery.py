#!/usr/bin/env python3
"""Read-only verification of the bounded E6 archive and its local inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args()
    archive = args.repo / 'results/p75_mask_e6_execution_20260929'
    checks = json.loads((archive / 'archive_sha256.json').read_text())
    for relative, expected in checks.items():
        assert digest(archive / relative) == expected, relative
    manifest = json.loads((args.runtime / 'manifest.json').read_text())
    for path, expected in manifest['files'].items():
        assert digest(path) == expected, path
    rows = json.loads((args.runtime / 'groups.json').read_text())
    with (archive / 'A/predictions.csv').open() as stream:
        predictions = list(csv.DictReader(stream))
    assert len(predictions) == len(rows) == 14880
    assert len({row['image_path'] for row in predictions}) == 14880
    assert [row['image_path'] for row in predictions] == [row['image_path'] for row in rows]
    report = json.loads((archive / 'A/report.json').read_text())
    sys.path.insert(0, str(args.repo / 'scripts'))
    from p75_pipeline_report import summarize
    reconstructed = summarize(
        rows,
        [int(row['label']) for row in predictions],
        [int(row['control']) for row in predictions],
        [int(row['masked']) for row in predictions],
    )
    assert reconstructed == report['groups'], 'Group or per-class metrics differ'
    expected_recipe = dict(tta='horizontal_flip', fusion='mean_probabilities',
                           temperature=1.4, prior_strength=0.6)
    for arm in ('control', 'masked'):
        binding = report['bindings'][arm]
        assert binding['prior']['recipe'] == expected_recipe
        assert binding['prior']['test_data_used'] is False
        assert digest(binding['prior']['checkpoint']) == binding['checkpoint_sha256']
        assert digest(binding['prior']['cache']) == binding['cache_sha256']
        assert digest(binding['continuation']['checkpoint']) == binding['continuation']['sha256']
    for epoch, updates in ((5, 655), (6, 786)):
        orders = []
        for arm in ('CONTROL', 'MASKED'):
            logs = archive / f'training_logs/P75_MASK_E6_{arm}/seed42/logs'
            orders.append(json.loads((logs / f'pair_order_{epoch}.json').read_text()))
            training = json.loads((logs / f'train_epoch_{epoch}.json').read_text())
            assert training['attempted_optimizer_updates'] == updates
            assert training['successful_optimizer_updates'] == updates
        assert orders[0] == orders[1], f'E{epoch} order/update/LR audit differs'
    state = json.loads((archive / 'A/status.json').read_text())
    assert state['status'] == 'local_result' and state['current'] is None
    assert len(state['history']) == 6
    assert all(stage['returncode'] == 0 for stage in state['history'])
    assert state['used_seconds'] < state['cost_gate']['budget_seconds'] == 21600
    assert report['platform_measured'] is False
    blocked = json.loads((archive / 'B/startup_status.json').read_text())
    assert blocked['authorized'] is True and blocked['training_started'] is False
    classes = reconstructed['all']['per_class']
    result = dict(
        archive_sha256_verified=len(checks),
        frozen_manifest_files_verified=len(manifest['files']),
        prediction_rows=len(predictions),
        classes=len(classes),
        all_groups_and_per_class_metrics_recomputed=True,
        checkpoint_cache_and_continuation_bindings_verified=True,
        paired_training_order_updates_and_lr_verified=True,
        successful_final_updates_per_arm=786,
        paired_extension_updates=524,
        used_seconds=state['used_seconds'],
        budget_seconds=21600,
        corrected=reconstructed['all']['corrected'],
        regressed=reconstructed['all']['regressed'],
        net=reconstructed['all']['net'],
        remaining_proxy_net=reconstructed['remaining_proxy']['net'],
        bio_dominant_net=reconstructed['bio_dominant']['net'],
        classes_with_positive_net=sum(item['net'] > 0 for item in classes.values()),
        classes_with_negative_net=sum(item['net'] < 0 for item in classes.values()),
        classes_with_zero_net=sum(item['net'] == 0 for item in classes.values()),
        platform_uploaded=False,
        new_full_candidate=False,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
