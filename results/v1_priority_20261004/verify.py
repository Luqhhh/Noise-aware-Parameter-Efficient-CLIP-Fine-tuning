"""Recheck the pivot's source evidence and incumbent, without model execution."""
from decimal import Decimal, ROUND_CEILING
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def verify():
    source = ROOT / 'results/top4_platform_20261004/verify.py'
    spec = importlib.util.spec_from_file_location('top4_verification', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package_checks = module.verify()
    paths = [
        'results/top4_platform_20261004/record.json',
        'results/v1_768_bias_platform_20261002/record.json',
        'results/v1_crt768_dev_20261001/delivery_report.json',
        'results/v1_768_full_test_bias_20261001/target_report.json',
        'results/v1_feature_calibration_20261001/feature_report.json',
    ]
    top4, v1, crt, targets, features = [json.loads((ROOT / p).read_text()) for p in paths]
    incumbent, control = top4['reported_packages']
    assert incumbent['decision'] == 'promote_to_incumbent'
    assert package_checks['status'] == 'verified'
    csv = Path(incumbent['source_zip']).with_name('pred_results.csv')
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == incumbent['csv_sha256']
    scores = dict(v2=Decimal(incumbent['score_reported_text']),
                  v1_768=Decimal(v1['platform_score_reported_text']),
                  v1_512=Decimal(control['score_reported_text']))
    counts = {}
    for name, score in scores.items():
        count = score * incumbent['rows'] / 100
        assert abs(count - round(count)) < Decimal('0.0000001')
        counts[name] = round(count)
    assert counts == dict(v2=28043, v1_768=27864, v1_512=27705)
    assert crt['teacher_targets_unchanged'] is True
    assert crt['validation_independent'] is True
    assert crt['visual_parameter_updates'] == 0
    assert targets['original_samples'] == 148695 and targets['used'] == 136631
    assert features['head_paired']['net_correct'] == 247
    assert features['knn_paired']['net_correct'] == 116
    documents = ['docs/current_execution_plan.md', 'docs/v1.md',
                 'docs/p75_error_budget_policy_20260929.md',
                 'docs/v2_preprojection768_full_20261003.md', 'docs/v1_priority_20261004.md']
    for name in documents:
        path = ROOT / name
        text = path.read_text()
        assert str(scores['v2']) in text
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' not in target:
                assert (path.parent / target.split('#')[0]).exists(), (name, target)
    current = (ROOT / documents[0]).read_text()
    table = current.split('## 当前阶段与平台结果', 1)[1].split('## 当前方案状态与下一步', 1)[0]
    assert '**74.89317380621728%** | 当前提交基准' in table
    assert '当前优化方向：V1_PRIORITY_20261004' in current
    budget = {}
    for target in (75, 80):
        required = int((Decimal(target) * incumbent['rows'] / 100).to_integral_value(rounding=ROUND_CEILING))
        budget[str(target)] = dict(required_correct=required,
                                  from_v2=required-counts['v2'],
                                  from_v1_768=required-counts['v1_768'])
    return dict(status='verified', experiment_id='V1_PRIORITY_20261004',
                optimization_line='v1', preferred_v1_route='768',
                incumbent_unchanged=True, promotion_rule='strictly_greater_platform_score',
                scores_percent={k: str(v) for k, v in scores.items()}, implied_correct=counts,
                v1_768_net_needed_to_strictly_exceed=counts['v2']-counts['v1_768']+1,
                targets=budget, incumbent_package_checks=package_checks,
                external_csv_matches=True, source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                document_links_checked=documents, new_training_started=False,
                new_inference_started=False, new_candidate_completed=False,
                platform_score_independently_verified=False)


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
