"""Fail-closed provenance for the fixed rematch750 experiment family."""
from __future__ import annotations

import json
from pathlib import Path

from aegis_clip.runtime import sha256_file, sha256_lines


def is_rematch(config):
    return bool(config.get('data', {}).get('dataset_manifest'))


def require(condition, message):
    if not condition:
        raise ValueError(f'Rematch provenance: {message}')


def official_weight_hash(config):
    from clip.clip import _MODELS
    p=Path(config['model']['official_checkpoint'])
    expected=_MODELS['ViT-B/32'].split('/')[-2]
    digest=sha256_file(p)
    require(digest==expected,'official OpenAI checkpoint SHA-256 mismatch')
    return digest


def validate_dataset(config):
    require(config['project']['stage']=='repechage','wrong stage')
    require(config['model']['backbone']=='ViT-B/32' and config['model']['pretrained']=='openai','wrong pretrained source')
    require(config['model'].get('input_resolution',224)==224,'reference preprocessing requires 224px')
    require(config['data']['external_data'] is False and config['data']['test_usage']=='inference_only','invalid data scope')
    path=Path(config['data']['dataset_manifest'])
    manifest=json.loads(path.read_text())
    require(manifest['stage']=='repechage','dataset belongs to another stage')
    require(manifest['data_version']==config['project']['data_version'],'wrong data version')
    require(manifest['num_classes']==config['model']['num_classes'],'class count mismatch')
    require(manifest['train_samples']==config['data']['expected_official_train_samples'],'train size mismatch')
    require(manifest['test_samples']==config['data']['expected_test_samples'],'test size mismatch')
    for key in ('train_root','test_root'):
        require(Path(config['data'][key]).resolve()==Path(manifest[key]).resolve(),f'{key} changed')
    for name,digest in manifest['files'].items():
        require(sha256_file(path.parent/name)==digest,f'dataset asset changed: {name}')
    report=json.loads((path.parent/'decode_report.json').read_text())
    require(report['status']=='passed' and not report['failures'],'decode audit failed')
    for key,name in [('class_mapping','class_to_idx.json')]:
        require(Path(config['data'][key]).resolve()==(path.parent/name).resolve(),f'foreign {key}')
    full=config['project'].get('full_training',False)
    for key,name in [('train_csv','full_train.csv' if full else 'train_dev.csv'),('val_csv','val_dev.csv')]:
        require(Path(config['data'][key]).resolve()==(path.parent/name).resolve(),f'foreign {key}')
    require(config['data'].get('validation_overlap_with_training',False)==full,'overlap declaration mismatch')
    require(not config['trust'].get('enabled') and not config['trust'].get('bundle_path'),'trust disabled in first round')
    require(config['longtail']['loss_reweighting']=='none','unexpected loss reweighting')
    require(config['longtail']['balanced_softmax_tau']==0,'prior disabled')
    for key in ('bundle_path','groups_path'):
        require(not config['trust'].get(key),f'foreign {key}')
    require(not config.get('lineage'),'legacy lineage inputs prohibited')
    for section in ('clean_routing','prototype_contrastive','dynamic_trust','elr'):
        require(not config.get(section,{}).get('enabled'),f'{section} not in first-round recipe')
    for key,value in config['loss'].items():
        if isinstance(value,dict):
            require(not value.get('enabled'),f'extra mechanism {key} prohibited')
    return manifest


def expected_binding(config, manifest):
    return dict(stage='repechage',data_version=manifest['data_version'],
        dataset_manifest_sha256=sha256_file(config['data']['dataset_manifest']),
        dataset_fingerprint=manifest['train_fingerprint'],
        class_mapping_sha256=sha256_file(config['data']['class_mapping']),
        official_checkpoint_sha256=official_weight_hash(config),
        preprocessing='OpenAI CLIP 224 bicubic resize/center crop/RGB/CLIP normalization',
        encoder_precision='float32',feature_precision='float32',autocast=False)


def validate_cache(config, manifest=None):
    manifest=manifest or validate_dataset(config)
    base=Path(config['data']['dataset_manifest']).parent/'features'
    for key,name in [('tensor_path','features.pt'),('paths_path','image_paths.json'),('manifest_path','manifest.json')]:
        require(Path(config['features'][key]).resolve()==(base/name).resolve(),f'foreign feature {key}')
    m=json.loads((base/'manifest.json').read_text())
    require(m.get('rematch_binding')==expected_binding(config,manifest),'feature provenance mismatch')
    require(sha256_file(base/'features.pt')==m['tensor_sha256'],'feature tensor changed')
    require(sha256_file(base/'image_paths.json')==m['paths_file_sha256'],'feature paths changed')
    import csv
    with (base.parent/'full_train.csv').open() as f:
        paths=[r['image_path'].removeprefix('train/') for r in csv.DictReader(f)]
    require(json.loads((base/'image_paths.json').read_text())==paths,'feature sample order mismatch')
    require(m['path_index_sha256']==sha256_lines(paths),'feature order hash mismatch')
    return m


def checkpoint_binding(config):
    manifest=validate_dataset(config)
    return dict(stage='repechage',data_version=manifest['data_version'],
        dataset_manifest_sha256=sha256_file(config['data']['dataset_manifest']),
        class_mapping_sha256=sha256_file(config['data']['class_mapping']),
        train_csv_sha256=sha256_file(config['data']['train_csv']),
        official_checkpoint_sha256=official_weight_hash(config),
        feature_manifest_sha256=sha256_file(config['features']['manifest_path']))


def validate_checkpoint(path, config, *, parent=False):
    path=Path(path)
    meta=json.loads(path.with_suffix('.binding.json').read_text())
    require(meta['checkpoint_sha256']==sha256_file(path),'checkpoint hash mismatch')
    require(meta['binding']==checkpoint_binding(config),'checkpoint data lineage mismatch')
    if not parent:
        require(meta['training_config_sha256']==sha256_file(config['_config_path']),'checkpoint training config mismatch')
    if parent:
        require(meta['experiment_id'] in ('RM_LP','RM_FULL_LP'),'initialization must be current-stage LP')
    return meta


def validate_training(config, resume=None, init_checkpoint=None):
    manifest=validate_dataset(config)
    validate_cache(config,manifest)
    if resume:
        validate_checkpoint(resume,config)
    source=init_checkpoint or config['train'].get('init_checkpoint')
    if source:
        validate_checkpoint(source,config,parent=True)
    elif config['model']['peft_mode']!='frozen':
        raise ValueError('Rematch visual fine-tuning must start from the shared RM-LP')
