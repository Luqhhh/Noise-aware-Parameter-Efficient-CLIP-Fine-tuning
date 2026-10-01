"""Recount reported aggregate scores and verify existing delivered artifacts on CPU."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RESULT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    record = json.loads((RESULT/'record.json').read_text())
    score = Decimal(record['platform_score_reported_text'])
    reference = json.loads((ROOT/record['decision']['incumbent_validation']).read_text())
    best = Decimal(reference['platform_score_reported_text'])
    comparison = record['platform_comparison']
    assert Decimal(str(record['platform_score_percent'])) == score
    assert Decimal(comparison['delta_pp_exact']) == score-best
    assert comparison['candidate_correct_implied'] == round(score*37444/100) == 26374
    assert comparison['reference_correct_implied'] == round(best*37444/100) == 26580
    assert comparison['delta_correct_implied'] == -206
    assert record['target_75_percent']['remaining_correct_implied'] == 28083-26374
    assert record['platform_package_binding_verified'] is False
    assert record['independent_platform_receipt_verified'] is False
    delivery = json.loads((ROOT/record['delivery_result']).read_text())
    for key, item in record['artifacts'].items():
        assert sha(item['path']) == item['sha256']
        expected = delivery['csv_sha256'] if key == 'csv' else delivery['zip_sha256']
        assert item['sha256'] == expected
    assert sha(record['checkpoint']) == record['checkpoint_sha256'] == delivery['checkpoint_sha256']
    csv = Path(record['artifacts']['csv']['path']).read_bytes()
    with zipfile.ZipFile(record['artifacts']['desktop_zip']['path']) as archive:
        assert archive.namelist() == ['pred_results.csv']
        assert archive.read('pred_results.csv') == csv
    assert len(csv.splitlines()) == record['submission_rows'] == 37444
    assert delivery['submission_checks_passed'] == record['submission_checks_passed'] == 9
    assert 'All checks passed' in Path(record['submission_check_source']).read_text()
    assert sha(record['decision']['incumbent_zip']) == reference['artifacts']['zip']['sha256']
    for flag in ('implementation_changed', 'new_training_started', 'new_inference_started', 'platform_uploaded_by_agent'):
        assert record[flag] is False
    for name in ('current_execution_plan.md', 'wft448_full_20261001.md', 'wft448_full_platform_20261001.md'):
        path = ROOT/'docs'/name
        text = path.read_text()
        assert str(score) in text
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' not in target:
                assert (path.parent/target.split('#', 1)[0]).exists(), (name, target)
    return dict(status='verified',record_id=record['record_id'],score_percent=str(score),
                delta_pp=str(score-best),correct_count_difference_implied=-206,
                source_desktop_zip_identical=True,checkpoint_hash_verified=True,
                incumbent_zip_verified=True,existing_submission_checks_passed=9,
                docs_and_local_links_verified=True,independent_platform_receipt_verified=False,
                implementation_changed=False,new_training_started=False,new_inference_started=False)


if __name__ == '__main__':
    print(json.dumps(verify(),ensure_ascii=False,indent=2))
