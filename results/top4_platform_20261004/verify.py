"""Recheck the two reported packages and score arithmetic without model execution."""
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RESULT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    record = json.loads((RESULT / 'record.json').read_text())
    assert sha(record['numbering_source']) == record['numbering_source_sha256']
    delivery = json.loads(Path(record['numbering_source']).read_text())
    checks = []
    for item, original in zip(record['reported_packages'], delivery['packages'][:2]):
        score = Decimal(item['score_reported_text'])
        assert Decimal(str(item['score_percent'])) == score
        implied = score * 37444 / 100
        assert abs(implied - item['implied_correct_count']) < Decimal('0.0000001')
        assert item['number'] == original['priority'] and item['name'] == original['name']
        assert sha(item['desktop_path']) == sha(item['source_zip']) == item['zip_sha256'] == original['zip_sha256']
        with zipfile.ZipFile(item['desktop_path']) as archive:
            assert archive.namelist() == ['pred_results.csv']
            csv = archive.read('pred_results.csv')
        assert hashlib.sha256(csv).hexdigest() == item['csv_sha256'] == original['csv_sha256']
        assert len(csv.splitlines()) == item['rows'] == 37444
        with tempfile.TemporaryDirectory(prefix='top4_platform_check_') as temporary:
            csv_path = Path(temporary) / 'pred_results.csv'
            csv_path.write_bytes(csv)
            run = subprocess.run([sys.executable, str(ROOT / 'scripts/check_submission.py'),
                '--test_dir', '/home/lux1/noise/test', '--num-classes', '750', '--csv', str(csv_path),
                '--zip', item['desktop_path']], capture_output=True, text=True, check=True)
            assert 'All checks passed' in run.stdout + run.stderr
        assert 'All checks passed' in (ROOT / item['check_log']).read_text()
        checks.append(dict(number=item['number'],zip_sha256=item['zip_sha256'],rows=37444,
                           source_and_desktop_identical=True,nine_checks_passed=True))
    a, b = [Decimal(p['score_reported_text']) for p in record['reported_packages']]
    old = json.loads((ROOT / record['previous_incumbent']['record']).read_text())['reported_packages'][0]
    assert old['zip_sha256'] == record['previous_incumbent']['zip_sha256']
    previous = Decimal(old['platform_score_reported_text'])
    v1 = Decimal('74.41512658903963')
    differences = {'new_v2_minus_previous_v2': a-previous, 'new_v2_minus_v1_512_bias': a-b,
                   'new_v2_minus_v1_768_bias': a-v1, 'v1_512_bias_minus_v1_768_bias': b-v1}
    for name, difference in differences.items():
        expected = record['comparisons'][name]
        assert Decimal(expected['delta_pp']) == difference
        assert expected['net_correct_implied'] == round(difference*37444/100)
    assert record['comparisons']['new_v2_minus_previous_v2']['net_correct_implied'] == 84
    assert a > previous and record['reported_packages'][0]['decision'] == 'promote_to_incumbent'
    for target, data in record['targets'].items():
        required = int((Decimal(target)*37444/100).to_integral_value(rounding=ROUND_CEILING))
        assert data['minimum_correct'] == required
        assert data['remaining_correct'] == required-28043
        assert Decimal(data['remaining_pp']) == Decimal(target)-a
    assert [p['number'] for p in record['next_pending']] == [3, 4]
    assert all(p['score_percent'] is None for p in record['next_pending'])
    for flag in ('independent_platform_receipt_verified', 'uploaded_bytes_independently_verified',
                 'implementation_changed', 'new_training_started', 'new_inference_started', 'platform_uploaded_by_agent'):
        assert record[flag] is False
    doc = ROOT / 'docs/top4_platform_20261004.md'
    for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
        assert (doc.parent / target).exists(), target
    assert str(a) in (ROOT/'docs/current_execution_plan.md').read_text()
    return dict(status='verified',record_id=record['record_id'],packages=checks,
                comparisons_recomputed=True,new_incumbent_percent=str(a),net_gain_correct_implied=84,
                remaining_to_75_correct_implied=40,remaining_to_80_correct_implied=1913,
                independent_platform_receipt_verified=False,new_training_started=False,new_inference_started=False)


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
