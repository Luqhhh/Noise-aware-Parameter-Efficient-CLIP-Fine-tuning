import json
import pandas as pd
import pytest
from aegis_clip.stage_pipeline import assign_oof_folds


def fixture(tmp_path):
    train=tmp_path/'train.csv';val=tmp_path/'val.csv';groups=tmp_path/'groups.json'
    paths=['0000/a','0000/b','0000/c','0001/a','0001/b','0001/c']
    pd.DataFrame({'image_path':paths,'label':[0,0,0,1,1,1]}).to_csv(train,index=False)
    pd.DataFrame({'image_path':['0000/v','0001/v'],'label':[0,1]}).to_csv(val,index=False)
    groups.write_text(json.dumps({p:p for p in paths+['0000/v','0001/v']}))
    return train,val,groups,set(paths)


def test_development_does_not_include_validation_in_folds(tmp_path):
    t,v,g,allowed=fixture(tmp_path);out=tmp_path/'folds.csv'
    assign_oof_folds(t,v,g,out,folds=2,seed=42,root_name='train',fit_scope='development_fit',allowed_fit_groups=allowed)
    df=pd.read_csv(out);assert len(df)==6 and not any(df.image_path.str.endswith('/v'))


def test_missing_scope_and_shared_content_block(tmp_path):
    t,v,g,allowed=fixture(tmp_path)
    with pytest.raises(ValueError,match='registered'):
        assign_oof_folds(t,v,g,tmp_path/'x.csv',folds=2,seed=42,root_name='train',fit_scope='development_fit')
    d=json.loads(g.read_text());d['0000/v']='0000/a';g.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='share content'):
        assign_oof_folds(t,v,g,tmp_path/'x.csv',folds=2,seed=42,root_name='train',fit_scope='development_fit',allowed_fit_groups=allowed)
