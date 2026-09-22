import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image

from aegis_clip.config import load_config
from aegis_clip.data import TestImageDataset as ImageDataset
from aegis_clip.evaluation import support_metrics
from aegis_clip.rematch_data import audit_image, split_groups


def test_split_protects_sparse_classes_and_never_splits_conflicting_group():
    rows=[dict(image_path=f'{c}/{i}',label=c,content_group=f'{c}-{i}')
          for c,n in enumerate([4,5,100]) for i in range(n)]
    rows += [dict(image_path='conflict',label=0,content_group='1-0')]
    train,val,support=split_groups(rows,3)
    assert not {r['content_group'] for r in train}&{r['content_group'] for r in val}
    assert len(train)+len(val)==len(rows)
    assert all(r['train_groups']>=4 for r in support)
    assert split_groups(rows,3)==(train,val,support)


def test_four_groups_no_holdout_and_five_groups_one_holdout():
    rows=[dict(image_path=f'{c}/{i}',label=c,content_group=f'{c}-{i}')
          for c,n in enumerate([4,5,100]) for i in range(n)]
    _,_,support=split_groups(rows,3)
    assert [r['val_groups'] for r in support]==[0,1,10]
    assert support[0]['validation_status']=='no_reliable_holdout'


def test_macro_omits_uncovered_classes_and_uses_fixed_support_bins():
    m=support_metrics(torch.tensor([0,1,1]),torch.tensor([0,1,2]),torch.tensor([5,20,100,101]))
    assert m['validation_covered_classes']==3
    assert m['support_head']['total_classes']==2
    assert m['support_head']['covered_classes']==1
    assert m['support_head']['macro']==0.
    assert m['per_class'][3]['recall'] is None
    assert m['support_tail']['macro']==1.


def test_pixel_groups_detect_same_pixels_with_different_encoding(tmp_path):
    a,b=tmp_path/'a.png',tmp_path/'b.bmp'
    image=Image.new('RGB',(5,5),'red');image.save(a);image.save(b)
    x=audit_image((a,'a',0,True)); y=audit_image((b,'b',1,True))
    assert x['file_sha256']!=y['file_sha256']
    assert x['content_group']==y['content_group']
    assert 'content_group' not in audit_image((b,'b',-1,False))


def test_strict_test_dataset_never_substitutes_placeholder(tmp_path):
    p=tmp_path/'a.jpg';p.write_bytes(b'broken')
    ds=ImageDataset(tmp_path,lambda im:torch.zeros(3,224,224),source_hashes={'a.jpg':hashlib.sha256(p.read_bytes()).hexdigest()})
    with pytest.raises(RuntimeError,match='Strict test'):
        ds[0]


def test_ft_lt_only_sampler_and_identity_differ():
    root=Path(__file__).resolve().parents[3]
    a=load_config(root/'configs/rematch750_ft.yaml')
    b=load_config(root/'configs/rematch750_lt.yaml')
    for c in (a,b):
        c.pop('_config_path');c['project'].pop('experiment_id');c['longtail'].pop('sampler_mode')
    assert a==b
    assert a['train']['epochs']==a['train']['schedule_epochs']==8
    assert a['evaluation']['selector_metric']=='raw_macro'


def test_official_stage_cannot_disable_manifest_guard():
    from aegis_clip.config import validate_config, ConfigError
    root=Path(__file__).resolve().parents[3]
    cfg=load_config(root/'configs/rematch750_lp.yaml')
    del cfg['data']['dataset_manifest']
    with pytest.raises(ConfigError,match='dataset_manifest'):
        validate_config(cfg)


def test_foreign_checkpoint_metadata_is_rejected_before_loading_weights(tmp_path, monkeypatch):
    import json
    import aegis_clip.rematch_assets as assets
    p=tmp_path/'legacy.pt';p.write_bytes(b'not torch data')
    p.with_suffix('.binding.json').write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),binding={'stage':'preliminary'},experiment_id='RM_LP')))
    monkeypatch.setattr(assets,'checkpoint_binding',lambda c: {'stage':'repechage'})
    with pytest.raises(ValueError,match='lineage'):
        assets.validate_checkpoint(p,{},parent=True)


def test_cache_rejects_same_shape_wrong_dataset_fingerprint(tmp_path, monkeypatch):
    import json
    import aegis_clip.rematch_assets as assets
    base=tmp_path/'features';base.mkdir()
    (base/'manifest.json').write_text(json.dumps({'rematch_binding':{'dataset_fingerprint':'old'}}))
    config={'data':{'dataset_manifest':str(tmp_path/'dataset_manifest.json')},'features':{
        'tensor_path':str(base/'features.pt'),'paths_path':str(base/'image_paths.json'),'manifest_path':str(base/'manifest.json')}}
    monkeypatch.setattr(assets,'expected_binding',lambda c,m: {'dataset_fingerprint':'current'})
    with pytest.raises(ValueError,match='feature provenance'):
        assets.validate_cache(config,manifest={'stage':'repechage'})


def test_pending_counts_once_and_tiny_platform_gain_updates_best():
    from aegis_clip.rematch_registry import record_receipt
    rows=[dict(candidate=n,platform_submission_id='',csv_sha256=n,status='ready',is_highest_score='false') for n in ['RM_LP','RM_FT','RM_LT']]
    def record(name, ident, status, score=None):
        return record_receipt(rows,name,submission_id=ident,submitted_at='2026-09-21T15:00:00+08:00',platform_period='actual-platform-period-1',status=status,score=score)
    record('RM_FT','A','pending')
    record('RM_FT','A','valid',41.)
    record('RM_LT','B','valid',41.0001)
    assert rows[2]['is_highest_score']=='true'
    with pytest.raises(ValueError,match='Two submissions'):
        record('RM_LP','C','pending')
    with pytest.raises(ValueError,match='duplicate upload'):
        record('RM_FT','D','pending')


def test_full_recipe_requires_platform_and_keeps_original_horizon(tmp_path):
    import json,shutil,yaml
    from aegis_clip.rematch_registry import full_configs
    root=Path(__file__).resolve().parents[3]
    (tmp_path/'configs').mkdir()
    for n in ('lp','ft'):
        shutil.copy(root/f'configs/rematch750_{n}.yaml',tmp_path/f'configs/rematch750_{n}.yaml')
    with pytest.raises(ValueError,match='waits'):
        full_configs(tmp_path,'RM_FT',[])
    for n,epoch,horizon in [('RM_LP',12,20),('RM_FT',6,8)]:
        p=tmp_path/'outputs/rematch750'/n/'seed42/checkpoints';p.mkdir(parents=True)
        (p/'selected_report.json').write_text(json.dumps(dict(selected_epoch=epoch,schedule_epochs=horizon)))
    paths=full_configs(tmp_path,'RM_FT',[dict(candidate=n,status='valid') for n in ('RM_FT','RM_LT')])
    a,b=[yaml.safe_load(p.read_text()) for p in paths]
    assert (a['train']['epochs'],a['train']['schedule_epochs'])==(12,20)
    assert (b['train']['epochs'],b['train']['schedule_epochs'])==(6,8)
    assert 'init_checkpoint' not in a['train']
    assert b['evaluation']['selection_policy']=='last_epoch'
    assert b['data']['validation_overlap_with_training'] is True


@pytest.mark.parametrize('inject_nonfinite', [False, True])
def test_interval_training_writes_only_selected_and_last(tmp_path,monkeypatch,inject_nonfinite):
    import json
    import pandas as pd
    import aegis_clip.trainer as trainer
    from aegis_clip.model import AegisCLIP
    root=Path(__file__).resolve().parents[3]
    cfg=load_config(root/'configs/rematch750_lp.yaml')
    cfg['data'].pop('dataset_manifest')
    cfg['model'].update(num_classes=3,feature_dim=4)
    cfg['model'].pop('official_checkpoint')
    cfg['train'].update(device='cpu',epochs=4,schedule_epochs=4,num_workers=0,loader_timeout=0,batch_size=4,amp=False)
    cfg['output']['root']=str(tmp_path/'out')
    paths=[f'000{c}/{j}.jpg' for c in range(3) for j in range(5)]
    labels=[c for c in range(3) for j in range(5)]
    frame=pd.DataFrame(dict(image_path=paths,label=labels))
    mask=frame.index%5==0
    for key,data in [('train_csv',frame[~mask]),('val_csv',frame[mask])]:
        p=tmp_path/(key+'.csv');data.to_csv(p,index=False);cfg['data'][key]=str(p)
    p=tmp_path/'mapping.json';p.write_text(json.dumps({f'{i:04d}':i for i in range(3)}));cfg['data']['class_mapping']=str(p)
    tensor=tmp_path/'features.pt';torch.save(torch.randn(15,4),tensor)
    index=tmp_path/'paths.json';index.write_text(json.dumps(paths))
    cfg['features']=dict(tensor_path=str(tensor),paths_path=str(index))
    monkeypatch.setattr(trainer,'build_model',lambda c,d:(AegisCLIP(visual=torch.nn.Linear(4,4),num_classes=3,feature_dim=4,peft_mode='frozen'),None))
    if inject_nonfinite:
        cfg['train']['require_finite_gradients']=True
        monkeypatch.setattr(trainer,'_gradient_norm',lambda params, **kwargs:float('inf'))
        def forbidden_step(*args,**kwargs):
            raise AssertionError('Optimizer must not run with nonfinite gradients')
        monkeypatch.setattr(torch.optim.AdamW,'step',forbidden_step)
        with pytest.raises(RuntimeError,match='before optimizer and scheduler'):
            trainer.train(cfg)
        assert not list((tmp_path/'out').rglob('*.pt'))
        return
    best=trainer.train(cfg)
    assert best.exists()
    assert (best.parent/'last.pt').exists()
    assert not list(best.parent.glob('epoch_*.pt'))
    logs=best.parent.parent/'logs'
    assert (logs/'evaluation_epoch_2.json').exists() and (logs/'evaluation_epoch_4.json').exists()
    assert not (logs/'evaluation_epoch_1.json').exists()
    assert len(json.loads((logs/'longtail_epoch_1.json').read_text())['classes'])==3


def test_submission_does_not_require_predicted_class_coverage(tmp_path):
    from aegis_clip.submission import create_submission
    import zipfile
    ckpt=tmp_path/'model.pt';ckpt.write_bytes(b'fixture')
    output=tmp_path/'submission'
    create_submission([('one.jpg','0001'),('two.jpg','0001')],['one.jpg','two.jpg'],output,ckpt,
                      inference_mode='none',tta_risk_acknowledged=False,
                      valid_labels={'0000','0001','0002'},space_after_comma=True)
    raw=(output/'pred_results.csv').read_bytes()
    assert b'one.jpg, 0001' in raw
    with zipfile.ZipFile(output/'submission.zip') as z:
        assert z.read('pred_results.csv')==raw
