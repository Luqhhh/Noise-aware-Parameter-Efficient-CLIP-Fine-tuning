import json
import pytest
import torch
from aegis_clip.calibration_binding import validate_frozen_prior, protocol_sha256
from aegis_clip.prior_alignment import apply_prior_bias
from aegis_clip.runtime import sha256_file


def fixture(tmp_path):
    context=dict(stage='preliminary',dataset_id='synthetic_unit',target_checkpoint_sha256='model',
                 class_mapping_sha256='mapping',num_classes=2,
                 original_supervision_sha256='supervision',
                 inference_protocol_sha256=protocol_sha256({'views':2,'temperature':1.5}))
    prior=dict(context,schema_version=2,fit_checkpoint_sha256='model',test_data_used=False,
               score_semantics='log_final_fused_probabilities',bias=[.1,-.1],strength=.9,
               strength_source='test fixture',calibration_design_record='test fixture')
    audit={k:v for k,v in prior.items() if k in ['stage','dataset_id','fit_checkpoint_sha256',
           'class_mapping_sha256','inference_protocol_sha256','original_supervision_sha256']}
    audit.update(status='checks_passed',source_authenticity_verified=True,authorizes_calibration=True,fit_scope='calibration_fit')
    train=tmp_path/'train'; (train/'0000').mkdir(parents=True)
    (train/'0000/a.jpg').write_bytes(b'synthetic image bytes')
    context['train_root']=str(train)
    (tmp_path/'fit_sample_manifest').write_text('image_path,label\n0000/a.jpg,0\n')
    (tmp_path/'fit_group_set').write_text(json.dumps([sha256_file(train/'0000/a.jpg')]))
    (tmp_path/'original_train_csv').write_text('image_path,label\n0000/a.jpg,0\n')
    (tmp_path/'trust_bundle').write_bytes(b'synthetic trust')
    torch.save(dict(logits=torch.tensor([[1.,2.]]),paths=['0000/a.jpg'],fit_scope='calibration_fit',
                    score_semantics='log_final_fused_probabilities',
                    fit_checkpoint_sha256='model',class_mapping_sha256='mapping',
                    original_supervision_sha256='supervision',
                    inference_protocol_sha256=context['inference_protocol_sha256']),tmp_path/'validation_logits')
    for k in ('fit_sample_manifest','fit_group_set','validation_logits','original_train_csv','trust_bundle'):
        p=tmp_path/k
        audit[k]={'path':str(p),'sha256':sha256_file(p)};prior[k+'_sha256']=sha256_file(p)
    p=tmp_path/'audit.json';p.write_text(json.dumps(audit));prior['source_audit']={'path':str(p),'sha256':sha256_file(p)}
    return prior,context,audit


@pytest.mark.parametrize(
    'key',
    [
        'stage',
        'dataset_id',
        'target_checkpoint_sha256',
        'class_mapping_sha256',
        'inference_protocol_sha256',
        'original_supervision_sha256',
    ],
)
def test_same_dimensions_cannot_hide_binding_mismatch(tmp_path,key):
    p,c,_=fixture(tmp_path);c[key]='changed'
    with pytest.raises(ValueError,match='mismatch'):validate_frozen_prior(p,c,base_dir=tmp_path)


def test_test_scope_and_false_flag_do_not_prove_source(tmp_path):
    p,c,a=fixture(tmp_path);a['fit_scope']='official_test';path=tmp_path/'audit.json';path.write_text(json.dumps(a));p['source_audit']['sha256']=sha256_file(path)
    with pytest.raises(ValueError,match='scope'):validate_frozen_prior(p,c,base_dir=tmp_path)


def test_training_overlap_scope_is_explicitly_accepted(tmp_path):
    p,c,a=fixture(tmp_path);a['fit_scope']='training_overlap_calibration'
    q=tmp_path/'validation_logits';payload=torch.load(q,weights_only=True)
    payload['fit_scope']='training_overlap_calibration';torch.save(payload,q)
    a['validation_logits']['sha256']=sha256_file(q);p['validation_logits_sha256']=sha256_file(q)
    path=tmp_path/'audit.json';path.write_text(json.dumps(a));p['source_audit']['sha256']=sha256_file(path)
    validate_frozen_prior(p,c,base_dir=tmp_path)


def test_test_fitted_prior_is_rejected_even_with_valid_assets(tmp_path):
    p,c,_=fixture(tmp_path);p['test_data_used']=True
    with pytest.raises(ValueError,match='Test-fitted'):validate_frozen_prior(p,c,base_dir=tmp_path)


def test_cross_model_bias_is_rejected(tmp_path):
    p,c,_=fixture(tmp_path);p['fit_checkpoint_sha256']='different-model'
    with pytest.raises(ValueError,match='Cross-model'):validate_frozen_prior(p,c,base_dir=tmp_path)


def test_score_semantics_mismatch_is_rejected(tmp_path):
    p,c,_=fixture(tmp_path);p['score_semantics']='single_branch_logits'
    with pytest.raises(ValueError,match='semantics'):validate_frozen_prior(p,c,base_dir=tmp_path)


def test_legacy_and_source_tampering_rejected(tmp_path):
    p,c,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='Legacy'):validate_frozen_prior({'format_version':1},c,base_dir=tmp_path)
    (tmp_path/'validation_logits').write_text('changed')
    with pytest.raises(ValueError,match='changed'):validate_frozen_prior(p,c,base_dir=tmp_path)


def test_fixed_application_independent_of_batch_members(tmp_path):
    p,c,_=fixture(tmp_path);bias,strength=validate_frozen_prior(p,c,base_dir=tmp_path)
    logits=torch.tensor([[1.,2.],[3.,1.],[.4,.8]])
    together=apply_prior_bias(logits,bias,strength=strength)
    separate=torch.cat([apply_prior_bias(x[None],bias,strength=strength) for x in logits])
    torch.testing.assert_close(together,separate,rtol=0,atol=0)
    torch.testing.assert_close(together.flip(0),apply_prior_bias(logits.flip(0),bias,strength=strength),rtol=0,atol=0)


def test_rebinding_test_logits_hash_does_not_hide_scope(tmp_path):
    p,c,a=fixture(tmp_path);q=tmp_path/'validation_logits'
    payload=torch.load(q,weights_only=True);payload['fit_scope']='official_test';torch.save(payload,q)
    a['validation_logits']['sha256']=sha256_file(q);p['validation_logits_sha256']=sha256_file(q)
    audit=tmp_path/'audit.json';audit.write_text(json.dumps(a));p['source_audit']['sha256']=sha256_file(audit)
    with pytest.raises(ValueError,match='scope'):validate_frozen_prior(p,c,base_dir=tmp_path)
