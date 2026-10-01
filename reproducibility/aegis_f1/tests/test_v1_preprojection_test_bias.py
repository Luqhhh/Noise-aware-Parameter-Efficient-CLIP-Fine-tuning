"""Readout equivalence, LoRA gradients, and explicit transductive boundaries."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from clip.model import VisionTransformer

from aegis_clip.v1_pipeline import load_recipe, seed_training
from aegis_clip.v1_strategy import V1Classifier
from aegis_clip.v1_test_bias import fit_test_uniform_bias

ROOT = Path(__file__).resolve().parents[3]


def test_preprojection_matches_native_cls_and_keeps_all_adapters_effective():
    torch.set_num_threads(2)
    seed_training(42)
    original = VisionTransformer(32, 8, 32, 2, 4, 16).float().eval()
    visual = copy.deepcopy(original)
    model = V1Classifier(visual, 3, image_size=32, rank=2, alpha=4,
                         blocks=2, feature_path="pre_projection")
    captured = []
    handle = original.ln_post.register_forward_hook(lambda m, a, z: captured.append(z.detach()))
    images = torch.randn(4, 3, 32, 32)
    original(images)
    handle.remove()
    assert visual.proj is None and model.head.weight.shape == (3, 32)
    torch.testing.assert_close(model.visual(images), captured[0], rtol=0, atol=0)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
    model.train()
    model(images).square().mean().backward()
    for block in model.visual.transformer.resblocks:
        for module, key in ((block.attn, "in_proj_weight"), (block.attn.out_proj, "weight"),
                            (block.mlp.c_fc, "weight"), (block.mlp.c_proj, "weight")):
            assert module.parametrizations[key][0].lora_B.grad.abs().sum() > 0
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(parameter, frozen[name])


def test_uniform_bias_matches_independent_numpy_and_balances_soft_mass():
    generator = torch.Generator().manual_seed(9)
    logits = torch.randn(60, 4, generator=generator) + torch.tensor([3., -1., 0., .5])
    bias = fit_test_uniform_bias(logits, iterations=200)
    matrix = logits.numpy().astype(np.float64)
    expected = np.zeros(4)
    for _ in range(200):
        z = matrix + expected
        p = np.exp(z-z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        expected -= np.log(np.maximum(p.mean(0)*4, 1e-8))
        expected -= expected.mean()
    np.testing.assert_allclose(bias.numpy(), expected, atol=2e-6, rtol=0)
    torch.testing.assert_close((logits+bias).softmax(1).mean(0), torch.full((4,), .25), atol=1e-6, rtol=0)
    assert (logits+bias).argmax(1).bincount(minlength=4).unique().numel() > 1


@pytest.mark.parametrize("bad", [torch.zeros(0, 3), torch.zeros(5, 1), torch.full((2, 3), float('nan'))])
def test_uniform_bias_rejects_invalid_matrix(bad):
    with pytest.raises(ValueError, match="Invalid"):
        fit_test_uniform_bias(bad)


def test_research_switch_and_cache_are_required_and_old_recipe_is_unchanged(tmp_path):
    original = yaml.safe_load((ROOT/'configs/v1_full_swa_20260930.yaml').read_text())
    path = tmp_path/'recipe.yaml'
    path.write_text(yaml.safe_dump(original))
    assert load_recipe(path)['decode']['bias_source'] == 'val_dev'
    research = copy.deepcopy(original)
    research['decode'].update(bias_source='test_uniform_experimental', logit_reduction='sum')
    path.write_text(yaml.safe_dump(research))
    with pytest.raises(ValueError, match="explicit"):
        load_recipe(path)
    research['decode']['experimental_test_bias'] = True
    research['model']['feature_path'] = 'pre_projection'
    path.write_text(yaml.safe_dump(research))
    with pytest.raises(ValueError, match="cache"):
        load_recipe(path)
    research['source']['preprojection_cache'] = {'tensor_path':'toy'}
    path.write_text(yaml.safe_dump(research))
    assert load_recipe(path)['model']['feature_path'] == 'pre_projection'


def test_test_uniform_inference_delivers_both_checked_packages_with_truthful_lineage(tmp_path, monkeypatch):
    import csv
    import json
    import zipfile
    from types import SimpleNamespace
    from PIL import Image
    from aegis_clip import v1_pipeline as pipeline
    from aegis_clip.v1_test_bias import infer_test_uniform
    from aegis_clip.runtime import sha256_file

    test_root = tmp_path/'test'
    test_root.mkdir()
    rows = []
    for i in range(9):
        name = f'{i}.jpg'
        Image.new('RGB', (48, 40), color=(i*20, 60, 100)).save(test_root/name)
        rows.append({'image_path':f'test/{name}', 'label':'-1'})
    with (tmp_path/'test_manifest.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image_path','label'])
        writer.writeheader()
        writer.writerows(rows)
    mapping = tmp_path/'class_to_idx.json'
    mapping.write_text(json.dumps({'0000':0,'0001':1,'0002':2}))
    checkpoint = tmp_path/'selected.pt'
    checkpoint.write_bytes(b'cpu-test-checkpoint')
    model = V1Classifier(VisionTransformer(32, 8, 32, 2, 4, 16), 3,
        image_size=32, rank=2, alpha=4, blocks=2, feature_path='pre_projection').eval()
    monkeypatch.setattr(pipeline, 'load_selected', lambda *args:(model, {'selected_policy':'swa_ema'}))
    ctx = SimpleNamespace(config={'project':{'experiment_id':'synthetic'},
        'model':{'image_size':32}, 'train':{'batch_size':3,'num_workers':0},
        'decode':{'bias_source':'test_uniform_experimental','experimental_test_bias':True,
            'logit_reduction':'sum','scales':[32,40,48],'flip':True,'bias_iterations':200,'bias_strength':1.},
        'output':{'root':str(tmp_path/'output')}}, classes=['0000','0001','0002'],
        reference={'data':{'dataset_manifest':str(tmp_path/'manifest.json'),'class_mapping':str(mapping)}},
        test_root=test_root, binding={'stage':'synthetic_cpu'}, manifest={'test_samples':9})
    before = {name:parameter.detach().clone() for name,parameter in model.named_parameters()}
    destination = infer_test_uniform(ctx, checkpoint, 'cpu')
    for directory, test_fit in [('submission',True),('submission_raw',False)]:
        package = destination.parent/directory
        assert 'All checks passed' in (package/'submission_check.log').read_text()
        manifest = json.loads((package/'manifest.json').read_text())
        assert manifest['test_statistical_fitting'] is test_fit
        assert manifest['checkpoint_sha256'] == sha256_file(checkpoint)
        with zipfile.ZipFile(package/'submission.zip') as archive:
            assert archive.namelist() == ['pred_results.csv']
            assert archive.read('pred_results.csv') == (package/'pred_results.csv').read_bytes()
    for name,parameter in model.named_parameters():
        assert torch.equal(parameter,before[name])
    artifact = torch.load(destination.parent/'test_calibration.pt',weights_only=False)
    assert artifact['test_data_used'] is True and artifact['official_permission_confirmed'] is True
    assert artifact['backbone_or_head_updates'] is False
    checkpoint.write_bytes(b'changed-checkpoint')
    with pytest.raises(ValueError,match='bound'):
        infer_test_uniform(ctx, checkpoint,'cpu')
