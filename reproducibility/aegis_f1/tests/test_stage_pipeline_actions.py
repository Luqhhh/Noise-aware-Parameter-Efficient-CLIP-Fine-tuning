import json
import pytest
from aegis_clip.stage_pipeline import run_stage_pipeline


@pytest.mark.parametrize('step',['final_train','prepare_final_train_csv'])
def test_final_train_legacy_alias_records_csv_action_not_model_training(tmp_path,step):
    split=tmp_path/'outputs/seed42';split.mkdir(parents=True)
    for name,sample in [('train','a'),('val','b')]:
        (split/f'{name}.csv').write_text(f'image_path,class_name,label\n0000/{sample}.jpg,0000,0\n')
    manifest=tmp_path/'pipeline.json'
    manifest.write_text(json.dumps({'stage':'preliminary','seed':42,'train_root':str(tmp_path/'images'),
                                   'output_root':str(split.parent),'expected_classes':1,'expected_samples':2,
                                   'device':'cpu','steps':[step]}))
    record=json.loads(run_stage_pipeline(manifest).read_text())
    assert record['step_results'][step]['operation']=='prepare_final_train_csv'
    assert record['step_results'][step]['model_training_performed'] is False
    assert (split/'final_train.csv').is_file()
    assert not list(split.rglob('*.pt'))
