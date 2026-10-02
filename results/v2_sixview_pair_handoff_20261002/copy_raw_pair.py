"""Copy the existing six-view raw control; --verify-only never writes artifacts."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/lux1/noise/worktrees/v2_sixview_bias_20261002/outputs/codex/v2_sixview_bias_20261002/candidate')
DELIVERY = Path('/mnt/c/Users/lqh22/Downloads/v2_continuation_20261001')
TARGET = DELIVERY / 'submission_raw_sixview'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_or_write(path, content, verify_only):
    if path.exists():
        assert path.read_bytes() == content, f'Refusing to overwrite different file: {path}'
    else:
        assert not verify_only, f'Missing delivered file: {path}'
        with path.open('xb') as stream:
            stream.write(content)


def run(verify_only=False):
    reference = ROOT / 'results/v2_sixview_bias_20261002'
    for name in ['report.json', 'independent_verification.json']:
        assert (SOURCE / name).read_bytes() == (reference / name).read_bytes()
    report = json.loads((SOURCE / 'report.json').read_text())
    verification = json.loads((SOURCE / 'independent_verification.json').read_text())
    assert report['status'] == 'completed_verified_delivery'
    assert verification['status'] == 'passed' and verification['full_float64_decision_agreement']
    assert not report['training'] and not report['model_parameter_updates']
    meta = {k: json.loads((SOURCE / f'submission_{k}/manifest.json').read_text()) for k in ('raw', 'bias')}
    for key in ['checkpoint_sha256', 'weights', 'views', 'input_sizes', 'reduction', 'binding', 'decoder_profile']:
        assert meta['raw'][key] == meta['bias'][key], f'Unpaired decoder field: {key}'
    assert meta['raw']['test_statistical_fitting'] is False
    assert meta['bias']['test_statistical_fitting'] is True
    assert meta['bias']['iterations'] == 200 and meta['bias']['strength'] == 1.0
    assert meta['raw']['views'] == [f'scale{s}{flip}' for s in (448, 512, 576) for flip in ('', '_flip')]
    assert meta['raw']['checkpoint_sha256'] == '3b9fcce3a4a2a113f2eb2ad3a2028c7f224fd1b11f1bdaadd6641ee1ccc85138'
    names = ['submission_raw/submission.zip', 'submission_raw/pred_results.csv',
             'submission_raw/submission_check.log', 'submission_raw/manifest.json',
             'report.json', 'independent_verification.json', 'binding.json', 'source_delivery_receipt.json']
    if not verify_only:
        TARGET.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in names:
        src, dst = SOURCE / name, TARGET / Path(name).name
        if dst.exists():
            assert sha(dst) == sha(src), f'Refusing to overwrite different file: {dst}'
        else:
            assert not verify_only, f'Missing delivered file: {dst}'
            pending = dst.with_name(dst.name + '.copying')
            with pending.open('xb') as out, src.open('rb') as incoming:
                shutil.copyfileobj(incoming, out)
            assert sha(pending) == sha(src)
            pending.rename(dst)
        files[dst.name] = dict(bytes=dst.stat().st_size, sha256=sha(dst))

    packages, predictions = {}, {}
    for kind in ('raw', 'bias'):
        directory = DELIVERY / f'submission_{kind}_sixview'
        csv_path, zip_path = directory / 'pred_results.csv', directory / 'submission.zip'
        expected = verification['packages'][f'submission_{kind}']
        assert sha(csv_path) == expected['csv_sha256'] and sha(zip_path) == expected['zip_sha256']
        with zipfile.ZipFile(zip_path) as archive:
            assert archive.namelist() == ['pred_results.csv']
            assert archive.read('pred_results.csv') == csv_path.read_bytes()
        rows = list(csv.reader(io.StringIO(csv_path.read_text()), skipinitialspace=True))
        predictions[kind] = dict(rows)
        assert len(rows) == len(predictions[kind]) == report['rows']
        assert 'All checks passed' in (directory / 'submission_check.log').read_text()
        packages[kind] = dict(path=str(zip_path), **expected)
    assert predictions['raw'].keys() == predictions['bias'].keys()
    changes = sum(label != predictions['bias'][name] for name, label in predictions['raw'].items())
    assert changes == report['bias_changed_predictions'] == verification['bias_changed_predictions'] == 3077
    receipt = dict(experiment_id='V2_SIXVIEW_PAIR_HANDOFF_20261002', status='verified_existing_pair_delivered',
        source=str(SOURCE), raw_copy_target=str(TARGET), copied_raw_files=files,
        packages=packages, checkpoint_sha256=meta['raw']['checkpoint_sha256'], weights=meta['raw']['weights'],
        views=meta['raw']['views'], reduction=meta['raw']['reduction'],
        predictions_sha256=report['predictions_sha256'], paired_difference='fixed_uniform_bias_only',
        changed_predictions=changes, rows=report['rows'], new_training=False, new_inference=False,
        platform_upload=False, platform_scores={'raw': None, 'bias': None},
        previous_four_view_raw=str(DELIVERY / 'submission/submission.zip'),
        previous_four_view_raw_is_paired_control=False,
        original_report_sha256=sha(reference / 'report.json'),
        original_verification_sha256=sha(reference / 'independent_verification.json'))
    payload = (json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode()
    same_or_write(DELIVERY / 'sixview_pair_receipt.json', payload, verify_only)
    same_or_write(Path(__file__).parent / 'pair_receipt.json', payload, verify_only)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    receipt = run(args.verify_only)
    print(json.dumps(dict(status=receipt['status'], rows=receipt['rows'], changes=receipt['changed_predictions'],
                         target=receipt['raw_copy_target'], verify_only=args.verify_only)))
