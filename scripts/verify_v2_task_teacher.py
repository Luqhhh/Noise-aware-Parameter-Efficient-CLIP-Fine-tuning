"""Independently verify the failed native replay; do not evaluate transfer gates."""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import torch

from diagnose_v1_candidate_errors import read_json, read_rows, require, sha
from verify_v1_frozen_teacher_recovery import counts

ROOT = Path(__file__).resolve().parents[1]


def verify(directory):
    directory = Path(directory).resolve()
    failure = read_json(directory / 'failure.json')
    require(failure['status'] == 'failed_closed' and failure['error'] == 'Too many native top1 replay changes' and
            failure['new_training'] is False and failure['new_candidate'] is False, 'Wrong failure')
    for path, digest in failure['sources'].items():
        require(sha(path) == digest, 'Changed source: ' + path)
    config_path = next(p for p in failure['sources'] if p.endswith('/configs/v2_task_teacher_recovery_20261002.json'))
    cfg = read_json(config_path)
    previous = read_json(ROOT / cfg['v2_error_report'])
    require(sha(ROOT / cfg['v2_error_report']) == cfg['v2_error_report_sha256'], 'Wrong prior report')
    rows = read_rows(previous['aligned_csv'])
    require(sha(previous['aligned_csv']) == previous['aligned_csv_sha256'], 'Wrong prior CSV')
    val = read_rows(Path(cfg['stage_root']) / 'val_dev.csv')
    with np.load(directory / 'scores.npz', allow_pickle=False) as z:
        cache = {k: z[k] for k in z.files}
    expected = {'logits', 'image_paths', 'labels', 'original_v2_prediction', 'teacher_prediction',
                'confidence', 'margin', 'supported'} | set(cfg['v1_candidates']) | {
                'group_' + k for k in cfg['groups'] + ['common_v1_errors', 'other_primary_errors']}
    require(set(cache) == expected, 'Unexpected score/fusion columns')
    labels = np.array([int(r['label']) for r in val])
    require(np.array_equal(cache['labels'], labels) and
        np.array_equal(cache['image_paths'], [r['image_path'] for r in val]), 'Val alignment differs')
    require(len(rows) == len(val) == previous['validation_rows'], 'Wrong population')
    for a, b in zip(rows, val):
        require(all(a[k] == b[k] for k in ('image_path', 'label', 'content_group')), 'Prior population differs')
    groups = {k: np.array([r[k] == '1' for r in rows]) for k in cfg['groups']}
    students = {k: np.array([int(r[k]) for r in rows]) for k in cfg['v1_candidates']}
    for k, p in students.items():
        require(np.array_equal(cache[k], p), 'Student source differs: ' + k)
    groups['common_v1_errors'] = np.array([all(int(r[k]) != int(r['label']) for k in cfg['v1_candidates']) for r in rows])
    groups['other_primary_errors'] = (students[cfg['primary_student']] != labels) & ~groups['common_v1_errors']
    for k, group in groups.items():
        require(np.array_equal(cache['group_' + k], group), 'Group differs: ' + k)
    original = np.array([int(r['s3_ema']) for r in rows])
    require(np.array_equal(cache['original_v2_prediction'], original), 'Archived V2 changed')
    require(cache['logits'].shape == (len(labels), previous['classes']), 'Logit population differs')
    tensor = torch.from_numpy(cache['logits']).to(torch.float64)
    require(bool(torch.isfinite(tensor).all()), 'Nonfinite logits')
    probabilities = tensor.softmax(dim=1)
    top = probabilities.topk(2, dim=1).values
    pred = tensor.argmax(1).numpy()
    confidence, margin = top[:, 0].numpy(), (top[:, 0] - top[:, 1]).numpy()
    require(np.array_equal(pred, cache['teacher_prediction']), 'Independent predictions differ')
    errors = {k: float(np.abs(v - cache[k]).max()) for k, v in [('confidence', confidence), ('margin', margin)]}
    require(max(errors.values()) < 1e-11, 'Independent probabilities differ')
    support = (confidence >= cfg['support_rule']['confidence_min']) & (margin >= cfg['support_rule']['probability_margin_min'])
    require(np.array_equal(support, cache['supported']), 'Support membership differs')
    changed = pred != original
    replay = dict(rows=len(labels), top1_changes=int(changed.sum()),
        changed_probability_margin_max=float(margin[changed].max()) if changed.any() else 0.,
        supported_top1_changes=int((changed & support).sum()), exact_top1_replay=not bool(changed.any()))
    limits = cfg['native_replay']
    replay_gates = dict(top1_count=replay['top1_changes'] <= limits['max_top1_changes'],
        changed_margin=replay['changed_probability_margin_max'] <= limits['changed_probability_margin_max'],
        no_supported_change=replay['supported_top1_changes'] <= limits['supported_changes_allowed'])
    require(not all(replay_gates.values()), 'Failure no longer reproduces')

    first = read_json(ROOT / 'results/v2_task_teacher_recovery_20261002/first_attempt.json')
    require(sha(first['scores_path']) == first['scores_sha256'] and
            sha(first['failure_path']) == first['failure_sha256'], 'Initial failure artifact changed')
    for commit, file, expected_sha in [(first['original_protocol_commit'], 'configs/v2_task_teacher_recovery_20261002.json',
            first['original_config_sha256']), (first['original_implementation_commit'], 'scripts/probe_v2_task_teacher.py',
            first['original_implementation_sha256'])]:
        content = subprocess.check_output(['git', 'show', f'{commit}:{file}'], cwd=ROOT)
        require(hashlib.sha256(content).hexdigest() == expected_sha, 'Initial commit bytes differ')
    with np.load(first['scores_path'], allow_pickle=False) as first_scores:
        identical_logits = np.array_equal(first_scores['logits'], cache['logits'])
    require(identical_logits, 'Two local batch replays no longer agree')
    archive_path = next(p for p in previous['sources'] if p.endswith('delivery_metadata.tar.gz'))
    require(sha(archive_path) == previous['sources'][archive_path], 'Archive SHA mismatch')
    archive_checks = {}
    with tarfile.open(archive_path) as archive:
        for suffix in ('v2/model.py', 'v2/runtime.py', 'v2/training_utils.py', 'aegis_clip/model.py'):
            members = [m for m in archive.getmembers() if m.name.endswith('/reproducibility/aegis_f1/' + suffix)]
            require(len(members) == 1 and members[0].isfile(), 'Ambiguous archived source')
            digest = hashlib.sha256(archive.extractfile(members[0]).read()).hexdigest()
            require(digest == sha(ROOT / 'reproducibility/aegis_f1' / suffix), 'Archived source differs')
            archive_checks[suffix] = digest
    total = first['diagnostic_seconds'] + failure['elapsed_seconds']
    require(total <= cfg['total_diagnostic_budget_seconds'], 'Cumulative budget exceeded')
    return dict(status='verified_failed_native_replay', experiment_id=cfg['experiment_id'],
        teacher_predictions_replayed=len(labels), support_memberships_replayed=len(labels),
        numerical_max_errors=errors, source_files_checked=len(failure['sources']), archive_source_sha=archive_checks,
        native_replay=replay, native_gate_components=replay_gates, native_gate_passed=False,
        initial_and_corrected_logits_identical=True,
        local_original_label_metrics=counts(labels, pred, np.ones(len(labels), bool)),
        archived_original_label_metrics=counts(labels, original, np.ones(len(labels), bool)),
        first_attempt_seconds=first['diagnostic_seconds'], corrected_attempt_seconds=failure['elapsed_seconds'],
        cumulative_diagnostic_seconds=total, score_file=str(directory / 'scores.npz'),
        score_sha256=sha(directory / 'scores.npz'), failure_sha256=sha(directory / 'failure.json'),
        verifier_sha256=sha(__file__), counter_implementation_sha256=sha(ROOT / 'scripts/verify_v1_frozen_teacher_recovery.py'),
        transfer_gate_evaluated=False, decision='close_audit_native_replay_mismatch_transfer_unknown',
        remaining_mismatch_cause='unknown; matching batch does not explain it; no remote logits available here',
        new_candidate=False, new_training=False, test_data_used=False, platform_gain_known=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = verify(args.directory)
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
