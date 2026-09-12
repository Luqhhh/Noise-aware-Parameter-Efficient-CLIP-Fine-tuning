"""Verify historical teacher audit bytes and recover its declared producer command.

Read-only with respect to assets. Does not execute teacher inference or authorize
formal reproduction. Recorded counters remain historical, never new measurements.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_stage_assets import audit_assets
from reproducibility.aegis_f1.aegis_clip.runtime import sha256_file

PARAMETERS = ('temperature', 'minimum_confidence', 'minimum_margin',
              'maximum_clean_probability', 'admission_clean_probability',
              'correction_alpha', 'maximum_class_fraction')
ASSETS = ('checkpoint', 'train_csv', 'base_trust', 'teacher_logits_cache', 'output')


def audit_teacher_producer(audit_path: Path, *, base_dir: Path) -> dict:
    record = json.loads(audit_path.read_text())
    missing = [k for k in ASSETS if not record.get(k) or not record.get(k + '_sha256')]
    missing += ['parameters.' + k for k in PARAMETERS if k not in record.get('parameters', {})]
    if missing:
        raise ValueError('Incomplete teacher audit: ' + ', '.join(missing))
    view = record.get('teacher_view_spec')
    if not isinstance(view, dict) or view.get('mode') != 'attention_multiscale':
        raise ValueError('This reader requires an explicit attention_multiscale view specification')
    for k in ('local_crop_sizes', 'local_top_k', 'local_weight', 'local_temperature'):
        if k not in view:
            raise ValueError('Incomplete teacher view: ' + k)
    assets = audit_assets([
        {'artifact_id':k, 'artifact_type':'teacher_input' if k != 'output' else 'trust_bundle',
         'path':record[k], 'sha256':record[k + '_sha256']}
        for k in ASSETS
    ], base_dir=base_dir)
    argv = ['{python}', '-m', 'aegis_clip.cli.build_teacher_trust']
    for k, flag in [('checkpoint','--checkpoint'), ('train_csv','--train-csv'),
                    ('base_trust','--base-trust')]:
        argv += [flag, record[k]]
    # New destinations are mandatory. Historical output paths are never replayed.
    argv += ['--output','{new_trust_output}', '--audit-output','{new_audit_output}',
             '--teacher-logits-cache','{verified_or_new_teacher_cache}',
             '--teacher-view',view['mode']]
    for k in ('local_crop_sizes', 'local_top_k', 'local_weight', 'local_temperature'):
        value = ','.join(map(str,view[k])) if k == 'local_crop_sizes' else str(view[k])
        argv += ['--' + k.replace('_','-'), value]
    for k in PARAMETERS:
        argv += ['--' + k.replace('_','-'), str(record['parameters'][k])]
    # These runtime conditions are not present in historical audit JSON.
    argv += ['--batch-size','{recorded_batch_size}', '--num-workers','{recorded_num_workers}']
    mismatches = [x['artifact_id'] for x in assets if x['status'] == 'hash_mismatch']
    missing_assets = [x['artifact_id'] for x in assets if not x['exists']]
    return {'schema_version':1, 'audit_source_sha256':sha256_file(audit_path),
            'status':'blocked' if mismatches or missing_assets else 'bytes_checks_passed',
            'artifacts':assets, 'hash_mismatches':mismatches,'missing_assets':missing_assets,
            'producer_argv_template':argv,
            'recorded_teacher_view_spec':view, 'recorded_parameters':record['parameters'],
            'unresolved':['producer commit', 'official initialization and upstream training provenance',
                          'source group-set binding', 'runtime conditions from original command'],
            'authorizes_execution':False, 'production_reproduced':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--base-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    result = audit_teacher_producer(Path(args.audit), base_dir=Path(args.base_dir))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=False)
    (out/'producer_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','missing_assets','hash_mismatches')},indent=2))
    return 2 if result['status']=='blocked' else 0


if __name__ == '__main__':
    raise SystemExit(main())
