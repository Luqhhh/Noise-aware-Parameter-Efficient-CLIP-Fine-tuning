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


def test_shadowed_cli_arguments_blocked(tmp_path):
    path, m = recipe(tmp_path)
    m['nodes'][0]['argv'] += ['--train-csv=' + str(tmp_path/'val.csv')]
    path.write_text(json.dumps(m))
    assert any('duplicate CLI option' in e for e in audit_recipe(path, REPO)['errors'])


def test_operation_cannot_misrepresent_csv_merge_as_training(tmp_path):
    path, m = recipe(tmp_path)
    m['nodes'][0]['operation'] = 'aegis_clip.cli.train'
    path.write_text(json.dumps(m))
    assert any('operation does not match' in e for e in audit_recipe(path, REPO)['errors'])


def test_produced_input_requires_dependency_ancestry(tmp_path):
    import copy
    path, m = recipe(tmp_path)
    second = copy.deepcopy(m['nodes'][0]); second['node_id'] = 'second'
    first_output = second['output_artifacts'][0]
    second['input_artifacts'][0]['path'] = first_output
    second['argv'][second['argv'].index('--train-csv') + 1] = first_output
    second['output_artifacts'] = [str(tmp_path/'run/second.csv')]
    second['argv'][second['argv'].index('--output-csv') + 1] = second['output_artifacts'][0]
    m['nodes'].append(second); m['protocol']['budget']['max_nodes'] = 2
    path.write_text(json.dumps(m))
    assert any('dependency ancestry' in e for e in audit_recipe(path, REPO)['errors'])
    second['depends_on'] = ['merge']; path.write_text(json.dumps(m))
    assert audit_recipe(path, REPO)['status'] == 'checks_passed'


def test_inference_checkpoint_must_be_bound(tmp_path):
    path, m = recipe(tmp_path); n = m['nodes'][0]
    n['operation'] = 'aegis_clip.cli.infer'
    n['argv'] = ['python3','-m', n['operation'], '--checkpoint', str(tmp_path/'hidden.pt'),
                 '--output-dir', str(tmp_path/'run/submission')]
    n['output_artifacts'] = [str(tmp_path/'run/submission/submission.zip')]
    path.write_text(json.dumps(m))
    assert any('undeclared --checkpoint' in e for e in audit_recipe(path, REPO)['errors'])


def test_parent_lineage_manifests_are_real_training_dependencies(tmp_path, monkeypatch):
    path, m = recipe(tmp_path); n = m['nodes'][0]
    parent = tmp_path/'parent.csv'; parent.write_text('parent lineage')
    cfg = tmp_path/'config.yaml'; cfg.write_text('{}')
    config = {'project': {'stage':'preliminary','experiment_id':'unit','seed':42},
              'data': {k:str(tmp_path/'train.csv') for k in ('train_csv','val_csv','class_mapping')},
              'features': {}, 'train': {'init_checkpoint':str(tmp_path/'train.csv'),
                                       'require_lineage_for_init_checkpoint':True},
              'lineage': {'parent_train_csv':str(parent),'parent_val_csv':str(parent)},
              'output': {'root':str(tmp_path/'run')}}
    monkeypatch.setattr('aegis_clip.config.load_config', lambda _: config)
    n.update(operation='aegis_clip.cli.train', config_path=str(cfg), config_sha256=sha256_file(cfg),
             argv=['python3','-m','aegis_clip.cli.train','--config',str(cfg)],
             output_artifacts=[str(tmp_path/'run/unit/seed42/checkpoints/best.pt')])
    path.write_text(json.dumps(m))
    assert any('undeclared input assets' in e for e in audit_recipe(path, REPO)['errors'])
    n['input_artifacts'].append({'path':str(parent),'sha256':sha256_file(parent),
                                'stage':'preliminary','scope':'synthetic_dryrun'})
    path.write_text(json.dumps(m))
    assert audit_recipe(path, REPO)['status']=='checks_passed'
