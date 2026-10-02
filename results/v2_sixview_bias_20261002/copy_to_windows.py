from pathlib import Path
import hashlib
import json
import shutil
import zipfile

source = Path('/home/lux1/noise/worktrees/v2_sixview_bias_20261002/outputs/codex/v2_sixview_bias_20261002/candidate')
target = Path('/mnt/c/Users/lqh22/Downloads/v2_continuation_20261001/submission_bias_sixview')
report = json.loads((source / 'report.json').read_text())
verification = json.loads((source / 'independent_verification.json').read_text())
assert report['status'] == 'completed_verified_delivery'
assert verification['status'] == 'passed' and verification['full_float64_decision_agreement']
assert report['rows'] == verification['rows'] == 37444
assert report['raw_replay_passed'] is None and not report['raw_comparison_is_same_decoder'] and not report['model_parameter_updates']
assert report['binding']['source_hashes']['runs/full_576/last.pt'] == '3b9fcce3a4a2a113f2eb2ad3a2028c7f224fd1b11f1bdaadd6641ee1ccc85138'
meta = json.loads((source / 'submission_bias/manifest.json').read_text())
assert meta['test_statistical_fitting'] and meta['iterations'] == 200 and meta['strength'] == 1.0
assert meta['decoder_profile'] == 'multiscale_flip_six' and meta['input_sizes'] == [448, 512, 576]
assert meta['views'] == [f'scale{size}{suffix}' for size in (448,512,576) for suffix in ('','_flip')]
assert 'All checks passed' in (source / 'submission_bias/submission_check.log').read_text()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

expected = verification['packages']['submission_bias']
assert sha(source / 'submission_bias/pred_results.csv') == expected['csv_sha256']
assert sha(source / 'submission_bias/submission.zip') == expected['zip_sha256']
with zipfile.ZipFile(source / 'submission_bias/submission.zip') as archive:
    assert archive.namelist() == ['pred_results.csv']
    assert archive.read('pred_results.csv') == (source / 'submission_bias/pred_results.csv').read_bytes()

target.mkdir(parents=True, exist_ok=True)
files = {}
names = ['submission_bias/submission.zip', 'submission_bias/pred_results.csv',
         'submission_bias/submission_check.log', 'submission_bias/manifest.json',
         'report.json', 'independent_verification.json', 'binding.json',
         'raw_replay.json', 'source_delivery_receipt.json']
for name in names:
    src = source / name
    dst = target / src.name
    expected_sha = sha(src)
    if dst.exists():
        assert sha(dst) == expected_sha, f'Refusing to overwrite different file: {dst}'
    else:
        pending = dst.with_name(dst.name + '.copying')
        shutil.copyfile(src, pending)
        assert sha(pending) == expected_sha
        pending.replace(dst)
    assert sha(dst) == expected_sha
    files[dst.name] = dict(bytes=dst.stat().st_size, sha256=expected_sha)
receipt = dict(status='verified_bias_package_copied', source=str(source), target=str(target),
               rows=37444, files=files, csv_zip_verified=True,
               official_submission_check_passed=True, independent_verification_passed=True)
receipt_path = target / 'local_copy_receipt.json'
receipt_text = json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'
if receipt_path.exists():
    assert receipt_path.read_text() == receipt_text
else:
    receipt_path.write_text(receipt_text)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
