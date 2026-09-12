import hashlib
import json
import pytest
import torch
from aegis_clip import stage_pipeline as pipeline
from aegis_clip.development_scope import development_rows, select_development_features


def fixture(tmp_path):
    split=tmp_path/'outputs/seed42';cache=split/'features';cache.mkdir(parents=True)
    (split/'train.csv').write_text('image_path,label\n0/a,0\n1/b,1\n')
    (split/'val.csv').write_text('image_path,label\n0/v,0\n')
    (split/'content_groups.json').write_text(json.dumps({'0/a':'a','1/b':'b','0/v':'v'}))
    allowed=tmp_path/'allowed.json';allowed.write_text(json.dumps(['a','b']))
    paths=['0/v','1/b','0/a'];torch.save(torch.tensor([[99.,99.],[0.,1.],[1.,0.]]),cache/'features.pt')
    (cache/'image_paths.json').write_text(json.dumps(paths));(cache/'labels.json').write_text(json.dumps([0,1,0]))
    manifest={'stage':'preliminary','seed':42,'train_root':str(tmp_path/'train'),'output_root':str(split.parent),'expected_classes':2,'expected_samples':3,'device':'cpu','steps':['trust'],'fit_scope':'development_fit','allowed_fit_groups':{'path':str(allowed),'sha256':hashlib.sha256(allowed.read_bytes()).hexdigest()}}
    path=tmp_path/'pipeline.json';path.write_text(json.dumps(manifest))
    return split,path,manifest


def test_trust_builder_never_receives_validation_features(tmp_path,monkeypatch):
    split,path,_=fixture(tmp_path)
    captured={}
    def builder(features,labels,paths,num_classes,**kwargs):
        captured.update(features=features,labels=labels,paths=paths,groups=kwargs['groups'])
        return {},{}
    monkeypatch.setattr(pipeline,'build_cross_fitted_trust',builder)
    result=json.loads(pipeline.run_stage_pipeline(path).read_text())
    assert captured['paths']==['0/a','1/b']
    assert captured['features'].tolist()==[[1.,0.],[0.,1.]]
    assert captured['groups']==['a','b']
    assert result['fit_scope']=='development_fit'
    assert result['source_authenticity_verified'] is False


def test_fullfit_assignments_rejected_in_development(tmp_path):
    split,path,m=fixture(tmp_path);m['steps']=['oof'];path.write_text(json.dumps(m))
    (split/'oof_assignments.csv').write_text('image_path,label,fold,sample_id\n0/a,0,0,a\n1/b,1,1,b\n0/v,0,0,v\n')
    with pytest.raises(ValueError,match='assignments do not match'):pipeline.run_stage_pipeline(path)


def test_merge_and_group_tampering_rejected(tmp_path):
    split,path,m=fixture(tmp_path);m['steps']=['final_train'];path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='cannot merge'):pipeline.run_stage_pipeline(path)
    type(path)(m['allowed_fit_groups']['path']).write_text('["a"]')
    with pytest.raises(ValueError,match='binding changed'):
        development_rows(split,'train',m['allowed_fit_groups'],base_dir=tmp_path)


def test_wrong_cache_label_cannot_silently_align():
    with pytest.raises(ValueError,match='label mismatch'):
        select_development_features(torch.ones(2,2),torch.tensor([1,0]),['a','v'],{'a':0},str)


def test_existing_trust_output_is_preserved(tmp_path):
    split,path,_=fixture(tmp_path)
    directory=split/'trust';directory.mkdir();saved=directory/'trust.pt';saved.write_bytes(b'previous')
    with pytest.raises(FileExistsError,match='output exists'):pipeline.run_stage_pipeline(path)
    assert saved.read_bytes()==b'previous'
