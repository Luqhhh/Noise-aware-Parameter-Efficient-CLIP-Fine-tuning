import pytest
import torch
from aegis_clip.classifier_norm import align_branch_logits, mean_norm_scale


def test_equalizes_norms_preserving_mean_direction_and_bias():
    w = torch.tensor([[3., 4.], [0., 10.]])
    scale = mean_norm_scale(w)
    out = w * scale[:, None]
    torch.testing.assert_close(out.norm(dim=1), torch.tensor([7.5, 7.5]))
    torch.testing.assert_close(torch.nn.functional.normalize(out), torch.nn.functional.normalize(w))
    x, b = torch.tensor([[1., -2.], [3., 1.]]), torch.tensor([.5, -1.])
    torch.testing.assert_close(align_branch_logits(x @ w.T + b, b, scale), x @ out.T + b, rtol=1e-5, atol=1e-5)


def test_dual_additive_feature_residual_uses_same_transform():
    g = torch.Generator().manual_seed(42)
    w = torch.randn(5, 7, generator=g); b = torch.randn(5, generator=g)
    base = torch.randn(3, 7, generator=g); residual = torch.randn(3, 7, generator=g)
    scale = mean_norm_scale(w)
    logits = base @ w.T + b + residual @ w.T
    direct = base @ (w * scale[:, None]).T + b + residual @ (w * scale[:, None]).T
    torch.testing.assert_close(align_branch_logits(logits, b, scale), direct, rtol=1e-5, atol=1e-5)


def test_identity_scale_is_exact():
    logits = torch.tensor([[1., 2.]])
    assert torch.equal(align_branch_logits(logits, torch.zeros(2), torch.ones(2)), logits)


@pytest.mark.parametrize('w', [torch.zeros(2, 3), torch.tensor([[float('nan')]]), torch.tensor([[float('inf')]]), torch.ones(3), torch.empty(0, 3), torch.ones(2, 3, dtype=torch.int64)])
def test_invalid_weights_fail(w):
    with pytest.raises(ValueError): mean_norm_scale(w)


def test_wrong_mapping_dimension_fails():
    with pytest.raises(ValueError): align_branch_logits(torch.zeros(2, 3), torch.zeros(2), torch.ones(2))


def test_materialized_checkpoint_is_inference_only_and_preserves_parent(tmp_path):
    from aegis_clip.classifier_norm import aligned_inference_checkpoint
    parent={'model_state_dict':{'classifier.weight':torch.tensor([[3.,4.],[0.,10.]]),
                               'classifier.bias':torch.tensor([1.,2.]), 'visual.weight':torch.ones(3)},
            'optimizer_state_dict':{},'metrics':{'stale':True},
            'local_feature_adapter':{'state_dict':{'w':torch.ones(2)}}}
    auth={'action':'materialize_classifier_norm_candidate','decision_source':'unit fixture'}
    candidate=aligned_inference_checkpoint(parent,parent_sha256='a'*64,authorization=auth)
    path=tmp_path/'candidate.pt';torch.save(candidate,path)
    restored=torch.load(path,weights_only=False)
    assert 'optimizer_state_dict' not in restored and 'metrics' not in restored
    assert torch.equal(restored['model_state_dict']['visual.weight'],parent['model_state_dict']['visual.weight'])
    assert torch.equal(restored['model_state_dict']['classifier.bias'],parent['model_state_dict']['classifier.bias'])
    assert torch.equal(parent['model_state_dict']['classifier.weight'],torch.tensor([[3.,4.],[0.,10.]]))
    torch.testing.assert_close(restored['model_state_dict']['classifier.weight'].norm(dim=1),torch.tensor([7.5,7.5]))
    assert restored['classifier_norm_alignment']['inference_only']


def test_materialization_requires_explicit_source():
    from aegis_clip.classifier_norm import aligned_inference_checkpoint
    with pytest.raises(ValueError):aligned_inference_checkpoint({},parent_sha256='a'*64,authorization={})
