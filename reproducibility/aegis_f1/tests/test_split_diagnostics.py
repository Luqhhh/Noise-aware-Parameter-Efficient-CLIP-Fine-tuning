import pytest
from aegis_clip.split_diagnostics import diagnose_group_split
from aegis_clip.cli.prepare_stage import prepare_stage


def test_duplicate_images_do_not_create_independent_support():
    r=diagnose_group_split([0,0,1,1], ['same','same','b','c'],num_classes=2,val_ratio=.5)
    assert r['status']=='blocked' and r['classes'][0]['unique_content_groups']==1
    assert not r['automatic_fallback']


def test_conflict_counts_and_impossible_folds_are_visible():
    r=diagnose_group_split([0,1,0,1],['x','x','a','b'],num_classes=2,val_ratio=.5,folds=3)
    assert [x['conflict_group_samples'] for x in r['classes']]==[1,1]
    assert any(x['reason']=='too_few_independent_groups_for_folds' for x in r['errors'])


def test_small_validation_capacity_blocks():
    r=diagnose_group_split([0,0,1,1,2,2],list('abcdef'),num_classes=3,val_ratio=.1)
    assert any(x['reason']=='split_capacity_below_class_count' for x in r['errors'])


def test_failure_emits_diagnosis_without_split_or_overwriting(tmp_path):
    root=tmp_path/'train';(root/'0000').mkdir(parents=True)
    (root/'0000/a.jpg').write_bytes(b'same');(root/'0000/b.jpg').write_bytes(b'same')
    out=tmp_path/'out'
    args=dict(train_root=root,output_dir=out,stage='preliminary',seed=42,val_ratio=.5,
              expected_classes=1,expected_samples=2)
    with pytest.raises(ValueError,match='infeasible'):prepare_stage(**args)
    assert (out/'split_diagnostics.json').exists() and not (out/'train.csv').exists()
    before=(out/'split_diagnostics.json').read_bytes()
    with pytest.raises(FileExistsError):prepare_stage(**args)
    assert (out/'split_diagnostics.json').read_bytes()==before
