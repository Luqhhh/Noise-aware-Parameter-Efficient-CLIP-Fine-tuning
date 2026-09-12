import hashlib
import json
from pathlib import Path
import pytest
from scripts.audit_teacher_producer import audit_teacher_producer, ASSETS, PARAMETERS


def fixture(tmp_path):
    record = {'parameters':{k:0.5 for k in PARAMETERS}, 'teacher_view_spec':{
        'mode':'attention_multiscale', 'local_crop_sizes':[128,144,160],
        'local_top_k':5,'local_weight':0.4,'local_temperature':1.5}}
    for key in ASSETS:
        p=tmp_path/key;p.write_bytes(key.encode())
        record[key]=key;record[key+'_sha256']=hashlib.sha256(key.encode()).hexdigest()
    p=tmp_path/'audit.json';p.write_text(json.dumps(record));return p,record


def test_actual_bytes_and_nonexecuting_template(tmp_path):
    p,record=fixture(tmp_path);r=audit_teacher_producer(p,base_dir=tmp_path)
    assert r['status']=='bytes_checks_passed'
    assert not r['production_reproduced'] and not r['authorizes_execution']
    argv=r['producer_argv_template']
    assert '--overwrite' not in argv
    assert argv[argv.index('--output')+1]=='{new_trust_output}'
    assert argv[argv.index('--local-crop-sizes')+1]=='128,144,160'


def test_missing_cache_is_not_replaced_by_declared_hash(tmp_path):
    p,_=fixture(tmp_path);(tmp_path/'teacher_logits_cache').unlink()
    r=audit_teacher_producer(p,base_dir=tmp_path)
    assert r['status']=='blocked' and r['missing_assets']==['teacher_logits_cache']
    assert next(x for x in r['artifacts'] if x['artifact_id']=='teacher_logits_cache')['sha256'] is None


def test_tampered_teacher_cannot_reuse_historical_audit(tmp_path):
    p,_=fixture(tmp_path);(tmp_path/'checkpoint').write_bytes(b'other teacher')
    r=audit_teacher_producer(p,base_dir=tmp_path)
    assert r['hash_mismatches']==['checkpoint'] and r['status']=='blocked'


def test_no_default_method_parameters(tmp_path):
    p,r=fixture(tmp_path);del r['parameters']['minimum_margin'];p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='minimum_margin'):audit_teacher_producer(p,base_dir=tmp_path)
