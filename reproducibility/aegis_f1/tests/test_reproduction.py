import json
from pathlib import Path
import pytest
from aegis_clip.reproduction import audit_recipe, run_recipe
from aegis_clip.runtime import sha256_file

REPO=Path(__file__).resolve().parents[3]


def recipe(tmp_path):
    for name,sample in [('train','a'),('val','b')]:
        (tmp_path/f'{name}.csv').write_text(f'image_path,class_name,label\n0000/{sample}.jpg,0000,0\n')
    root=tmp_path/'run'
    node={'node_id':'merge','operation':'aegis_clip.cli.prepare_final_train','cwd':'reproducibility/aegis_f1',
          'argv':['python3','-m','aegis_clip.cli.prepare_final_train','--train-csv',str(tmp_path/'train.csv'),'--val-csv',str(tmp_path/'val.csv'),'--output-csv',str(root/'merged.csv'),'--expected-samples','2'],
          'depends_on':[], 'input_artifacts':[{'path':str(tmp_path/f'{n}.csv'),'sha256':sha256_file(tmp_path/f'{n}.csv'),'stage':'preliminary','scope':'synthetic_dryrun'} for n in ('train','val')],
          'output_artifacts':[str(root/'merged.csv')]}
    manifest={'schema_version':1,'stage':'preliminary','dataset_id':'synthetic_unit','fit_scope':'synthetic_dryrun','output_root':str(root),'protocol':{'approved':True,'decision_source':'test fixture','budget':{'max_nodes':1,'max_wall_seconds':60}},'nodes':[node]}
    path=tmp_path/'recipe.json';path.write_text(json.dumps(manifest));return path,manifest


def test_default_is_audit_only(tmp_path):
    path,m=recipe(tmp_path)
    assert run_recipe(path,REPO)['status']=='checks_passed'
    assert not Path(m['output_root']).exists()


def test_real_node_execution_resume_and_changed_output_refusal(tmp_path):
    path,m=recipe(tmp_path)
    first=run_recipe(path,REPO,execute=True)
    assert first['nodes']['merge']['status']=='checks_passed'
    assert run_recipe(path,REPO,execute=True,resume=True)['status']=='checks_passed'
    Path(m['nodes'][0]['output_artifacts'][0]).write_text('tampered')
    with pytest.raises(ValueError,match='changed'):run_recipe(path,REPO,execute=True,resume=True)


def test_missing_input_and_formal_scope_blocked(tmp_path):
    path,m=recipe(tmp_path);(tmp_path/'train.csv').unlink()
    assert any('missing input' in e for e in audit_recipe(path,REPO)['errors'])
    m['fit_scope']='final_fit';path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='formal execution blocked'):run_recipe(path,REPO,execute=True)


def test_concurrent_lock_and_existing_output_refused(tmp_path):
    path,m=recipe(tmp_path);root=Path(m['output_root']);root.mkdir();(root/'.reproduce.lock').write_text('other')
    with pytest.raises(FileExistsError):run_recipe(path,REPO,execute=True)
    assert (root/'.reproduce.lock').read_text()=='other'
    (root/'.reproduce.lock').unlink();(root/'merged.csv').write_text('existing')
    assert any('output conflict' in e for e in audit_recipe(path,REPO)['errors'])
    with pytest.raises(ValueError,match='output conflict'):run_recipe(path,REPO,execute=True)
    assert (root/'merged.csv').read_text()=='existing'


def test_interruption_never_marks_node_complete(tmp_path,monkeypatch):
    path,m=recipe(tmp_path)
    def interrupt(*args,**kwargs):raise KeyboardInterrupt()
    monkeypatch.setattr('aegis_clip.reproduction.subprocess.run',interrupt)
    with pytest.raises(KeyboardInterrupt):run_recipe(path,REPO,execute=True)
    state=json.loads((Path(m['output_root'])/'reproduction_run.json').read_text())
    assert state['nodes']['merge']['status']=='failed'
    assert not (Path(m['output_root'])/'.reproduce.lock').exists()
    with pytest.raises(ValueError,match='incomplete'):run_recipe(path,REPO,execute=True,resume=True)


def test_changed_input_cannot_reuse_completed_node(tmp_path):
    path,m=recipe(tmp_path);run_recipe(path,REPO,execute=True)
    (tmp_path/'train.csv').write_text('changed')
    with pytest.raises(ValueError,match='hash mismatch'):run_recipe(path,REPO,execute=True,resume=True)
