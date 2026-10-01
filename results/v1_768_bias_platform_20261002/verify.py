"""Recheck reported aggregate score, delivered bytes and fixed bias provenance."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RESULT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify():
    record = json.loads((RESULT / 'record.json').read_text())
    score = Decimal(record['platform_score_reported_text'])
    old = json.loads((ROOT / record['previous_best_record']).read_text())
    previous = Decimal(old['platform_score_reported_text'])
    comparison = record['platform_comparison']
    n = record['submission_rows']
    correct = round(score * n / 100)
    previous_correct = round(previous * n / 100)
    raw_score = Decimal(record['raw_platform_score_reported_text'])
    raw_correct = round(raw_score * n / 100)
    assert Decimal(str(record['platform_score_percent'])) == score
    assert Decimal(comparison['delta_pp_exact']) == score - previous
    assert comparison['candidate_correct_implied'] == correct == 27864
    assert comparison['previous_correct_implied'] == previous_correct == 26580
    assert comparison['delta_correct_implied'] == correct - previous_correct == 1284
    assert raw_correct == comparison['raw_correct_implied'] == 26787
    assert Decimal(comparison['bias_minus_raw_pp_exact']) == score - raw_score
    assert Decimal(comparison['raw_minus_previous_pp_exact']) == raw_score - previous
    assert comparison['bias_minus_raw_correct_implied'] == correct - raw_correct == 1077
    assert comparison['raw_minus_previous_correct_implied'] == raw_correct - previous_correct == 207
    assert Decimal(record['target_75_percent']['remaining_pp_exact']) == 75 - score
    assert record['target_75_percent']['remaining_correct_implied'] == 28083 - correct == 219
    for flag in ('platform_package_binding_verified', 'independent_platform_receipt_verified',
                 'implementation_changed', 'new_training_started', 'new_inference_started',
                 'platform_uploaded_by_agent'):
        assert record[flag] is False
    delivery = json.loads((ROOT / record['delivery_result']).read_text())
    manifests = []
    csvs = []
    for part, artifacts in (('submission', record['artifacts']), ('submission_raw', record['raw_artifacts'])):
        package = delivery['packages'][part]
        for name, artifact in artifacts.items():
            expected = package['csv_sha256'] if name == 'csv' else package['zip_sha256']
            assert sha(artifact['path']) == artifact['sha256'] == expected
        csv = Path(artifacts['csv']['path']).read_bytes()
        for name in ('zip', 'desktop_zip'):
            with zipfile.ZipFile(artifacts[name]['path']) as archive:
                assert archive.namelist() == ['pred_results.csv']
                assert archive.read('pred_results.csv') == csv
        assert len(csv.splitlines()) == n == package['samples'] == 37444
        manifest = json.loads(Path(artifacts['csv']['path']).with_name('manifest.json').read_text())
        assert manifest['checkpoint_sha256'] == record['checkpoint_sha256']
        assert manifest['submission_zip_sha256'] == package['zip_sha256']
        assert manifest['prediction_csv_sha256'] == package['csv_sha256']
        assert manifest['test_statistical_fitting'] is (part == 'submission')
        manifests.append(manifest)
        csvs.append(csv)
        checked = subprocess.run([sys.executable, str(ROOT / 'scripts/check_submission.py'),
            '--test_dir', record['test_dir'], '--class-mapping', record['class_mapping'],
            '--csv', artifacts['csv']['path'], '--zip', artifacts['desktop_zip']['path']],
            capture_output=True, text=True)
        log = checked.stdout + checked.stderr
        (RESULT / (part + '_check.log')).write_text(log)
        checked.check_returncode()
        assert 'All checks passed' in log
    for key in ('checkpoint_sha256', 'raw_test_logits_sha256', 'binding', 'decoder'):
        assert manifests[0][key] == manifests[1][key]
    assert sum(left != right for left, right in zip(csvs[0].splitlines(), csvs[1].splitlines())) == 6775
    assert sha(record['checkpoint']) == record['checkpoint_sha256'] == delivery['checkpoint_sha256']
    bias = json.loads(Path(record['bias_report']).read_text())
    assert bias['checkpoint_sha256'] == record['checkpoint_sha256']
    assert bias['iterations'] == 200 and bias['strength'] == 1 and bias['views'] == 6
    assert bias['classes'] == 750 and bias['samples'] == n
    assert bias['test_statistical_fitting'] is True
    assert bias['test_labels_used'] is False and bias['model_parameter_updates'] is False
    assert bias['permission_source'] == 'user-relayed official confirmation 2026-10-01'
    assert bias['calibrated']['argmax_min'] == 43 and bias['calibrated']['argmax_max'] == 55
    assert bias['predictions_changed'] == 6775
    assert sha(record['calibration']) == bias['calibration_sha256']
    assert manifests[0]['calibration_sha256'] == bias['calibration_sha256']
    assert manifests[1]['calibration_sha256'] is None
    assert manifests[0]['raw_test_logits_sha256'] == bias['test_logits_sha256']
    assert sha(old['artifacts']['zip']['path']) == old['artifacts']['zip']['sha256']
    docs = ('current_execution_plan.md', 'v1.md', 'p75_error_budget_policy_20260929.md',
            'v1_768_full_test_bias_20261001.md', 'v1_768_bias_platform_20261002.md')
    for name in docs:
        path = ROOT / 'docs' / name
        text = path.read_text()
        assert str(score) in text
        assert str(raw_score) in text
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' not in target:
                assert (path.parent / target.split('#', 1)[0]).exists(), (name, target)
    return dict(status='verified', record_id=record['record_id'], score_percent=str(score),
        raw_score_percent=str(raw_score), delta_pp=str(score - previous),
        bias_minus_raw_pp=str(score - raw_score), bias_minus_raw_correct_implied=1077,
        raw_minus_previous_pp=str(raw_score - previous), raw_minus_previous_correct_implied=207,
        implied_correct=correct, raw_implied_correct=raw_correct, implied_net_correct=1284,
        remaining_correct_to_75_implied=219, submission_checks_passed_per_package=9,
        packages_verified=2, same_checkpoint_logits_and_decoder_verified=True,
        source_desktop_zip_identical=True, zip_csv_byte_identical=True,
        checkpoint_hash_verified=True, calibration_hash_verified=True,
        fixed_bias_provenance_verified=True, previous_best_zip_preserved=True,
        docs_and_local_links_verified=True, independent_platform_receipt_verified=False,
        full_package_official_compliance_certified=False,
        new_training_started=False, new_inference_started=False)


if __name__ == '__main__':
    result = verify()
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    (RESULT / 'validation.json').write_text(serialized)
    print(serialized, end='')
