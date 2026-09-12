"""Fail-closed version-2 frozen calibration application bindings."""
import csv
import hashlib
import json
from pathlib import Path
import torch
from aegis_clip.runtime import sha256_file


def protocol_sha256(descriptor):
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_frozen_prior(record, context, *, base_dir):
    if record.get('schema_version') != 2:
        raise ValueError('Legacy prior lacks source bindings; version 2 required for inference')
    for key in ('stage', 'dataset_id', 'target_checkpoint_sha256', 'class_mapping_sha256',
                'num_classes', 'inference_protocol_sha256'):
        if key not in context or record.get(key) != context[key]:
            raise ValueError(f'Frozen prior {key} mismatch')
    if record.get('fit_checkpoint_sha256') != record['target_checkpoint_sha256']:
        raise ValueError('Cross-model calibration requires a separate registered implementation')
    if record.get('test_data_used') is not False:
        raise ValueError('Test-fitted prior forbidden')
    if not record.get('calibration_design_record') or not record.get('strength_source'):
        raise ValueError('Calibration design and fixed strength source required')
    source = record.get('source_audit', {})
    path = Path(source.get('path',''))
    if not path.is_absolute():path = Path(base_dir)/path
    if not path.is_file() or sha256_file(path) != source.get('sha256'):
        raise ValueError('Missing or mismatched source audit')
    audit = json.loads(path.read_text())
    if (audit.get('status') != 'checks_passed' or
        audit.get('source_authenticity_verified') is not True or
        audit.get('authorizes_calibration') is not True):
        raise ValueError('Source audit does not establish fitting provenance')
    if audit.get('fit_scope') not in {'calibration_fit', 'training_overlap_calibration'}:
        raise ValueError('Source scope cannot fit calibration')
    for key in ('stage','dataset_id','fit_checkpoint_sha256','class_mapping_sha256',
                'inference_protocol_sha256'):
        if audit.get(key) != record.get(key):raise ValueError(f'Source audit {key} mismatch')
    # Source files remain necessary at apply time until a portable signed/verified
    # release receipt exists. Merely changing test_data_used cannot pass this.
    for key in ('fit_sample_manifest', 'fit_group_set', 'validation_logits'):
        binding = audit.get(key,{})
        asset = Path(binding.get('path',''))
        if not asset.is_absolute():asset = path.parent/asset
        if not asset.is_file() or sha256_file(asset) != binding.get('sha256'):
            raise ValueError(f'Source {key} missing or changed')
        if record.get(key+'_sha256') != binding['sha256']:
            raise ValueError(f'Prior source {key} binding mismatch')
    train_root = Path(context['train_root']).resolve()
    def source_path(key):
        p = Path(audit[key]['path'])
        return p if p.is_absolute() else path.parent/p
    rows = list(csv.DictReader(source_path('fit_sample_manifest').open()))
    if not rows or any('image_path' not in row or 'label' not in row for row in rows):
        raise ValueError('Invalid fit sample manifest')
    identities = []
    for row in rows:
        raw = Path(row['image_path'].replace('\\', '/'))
        if '..' in raw.parts:raise ValueError('Parent traversal in calibration source')
        if raw.is_absolute():
            resolved = raw.resolve()
        else:
            parts = raw.parts[1:] if raw.parts and raw.parts[0] == train_root.name else raw.parts
            resolved = train_root.joinpath(*parts).resolve()
        if not resolved.is_relative_to(train_root):
            raise ValueError('Calibration source is outside official training root')
        identities.append(resolved.relative_to(train_root).as_posix())
    if len(identities) != len(set(identities)):
        raise ValueError('Repeated calibration sample')
    actual_groups = set()
    for identity, row in zip(identities, rows):
        sample = (train_root/identity).resolve()
        if not sample.is_relative_to(train_root) or not sample.is_file():
            raise ValueError('Calibration source is not an official training-root sample')
        if not 0 <= int(row['label']) < context['num_classes']:
            raise ValueError('Calibration label outside mapping')
        actual_groups.add(sha256_file(sample))
    declared_groups = json.loads(source_path('fit_group_set').read_text())
    if not isinstance(declared_groups, list) or set(declared_groups) != actual_groups:
        raise ValueError('Fit content-group identities do not match source images')
    payload = torch.load(source_path('validation_logits'), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('fit_scope') != audit['fit_scope']:
        raise ValueError('Logits source scope mismatch')
    for key in ('fit_checkpoint_sha256', 'inference_protocol_sha256', 'class_mapping_sha256'):
        if payload.get(key) != record.get(key):raise ValueError('Logits producer binding mismatch')
    if payload.get('paths') != identities:
        raise ValueError('Logits sample order/source mismatch')
    logits = payload.get('logits')
    if not isinstance(logits, torch.Tensor) or logits.shape != (len(rows), context['num_classes']) or not torch.isfinite(logits).all():
        raise ValueError('Invalid calibration logits asset')
    bias = torch.tensor(record.get('bias',[]), dtype=torch.float32)
    strength = float(record.get('strength',float('nan')))
    if bias.shape != (context['num_classes'],) or not torch.isfinite(bias).all():
        raise ValueError('Invalid frozen bias vector')
    if not 0 <= strength <= 1:raise ValueError('Invalid frozen strength')
    return bias, strength
