"""Requested input sizes, exact flip pairs and independent six-view delivery."""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_v2_fixed_prior as run
from tests.test_v2_fixed_prior import delivered, dump


def config():
    return run.load_config(ROOT / 'configs/v2_sixview_bias_20261002.json')


@pytest.mark.parametrize('field,value', [
    ('image_sizes', [448, 512, 768]), ('views', run.SIX_VIEWS[:-1]),
    ('bias_iterations', 100), ('bias_strength', .8), ('decoder_profile', 'other')])
def test_six_view_protocol_rejects_changes(tmp_path, field, value):
    cfg = config()
    cfg[field] = value
    path = tmp_path / 'bad.json'
    dump(path, cfg)
    with pytest.raises(ValueError):
        run.load_config(path)


def test_real_six_view_collection_resizes_and_flips_each_input(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    image = np.zeros((700, 900, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(900, dtype=np.uint16) % 255
    image[:, :, 1] = np.arange(700, dtype=np.uint16)[:, None] % 255
    Image.fromarray(image).save(tmp_path / 'a.png')
    cfg = config()
    calls, logits_seen = [], []

    class Model:
        head = SimpleNamespace(out_features=3)

        def __call__(self, images):
            calls.append(images.clone())
            values = [20, -3, -2] if len(calls) == 1 else [-1, 2, 1]
            logits = torch.tensor([values], dtype=torch.float32).expand(len(images), -1)
            logits_seen.append(logits)
            return logits

    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    monkeypatch.setattr(run, 'amp', nullcontext)
    real_loader = run.DataLoader
    def cpu_loader(dataset, **kwargs):
        kwargs.update(num_workers=0, pin_memory=False)
        kwargs.pop('prefetch_factor')
        return real_loader(dataset, **kwargs)
    monkeypatch.setattr(run, 'DataLoader', cpu_loader)
    probabilities = run.collect_probabilities(Model(), tmp_path, ['a.png'], cfg, tmp_path)
    assert [batch.shape[-1] for batch in calls] == [448, 448, 512, 512, 576, 576]
    for i in (0, 2, 4):
        torch.testing.assert_close(calls[i].flip(3), calls[i+1], rtol=0, atol=0)
    expected = sum(logits.softmax(1).numpy() for logits in logits_seen) / 6
    np.testing.assert_allclose(probabilities, expected, atol=1e-7, rtol=0)
    assert probabilities.argmax(1).item() != sum(logits_seen).argmax(1).item()


def test_six_view_source_binding_keeps_original_package(delivered):
    cfg, source, _ = delivered
    cfg.update(decoder_profile='multiscale_flip_six', views=run.SIX_VIEWS, image_sizes=[448, 512, 576])
    inputs = run.verified_delivery(cfg)
    assert inputs['plan']['recipe']['views'] == run.VIEWS
    assert inputs['source_hashes'][run.FINAL] == run.sha256_file(source / run.FINAL)


def test_six_view_end_to_end_allows_changed_decoder_and_replays_bias(delivered, monkeypatch):
    cfg, source, _ = delivered
    cfg.update(decoder_profile='multiscale_flip_six', views=run.SIX_VIEWS, image_sizes=[448, 512, 576])
    path = source / 'six.json'
    dump(path, cfg)
    rng = np.random.default_rng(47)
    scores = rng.normal(size=(9, 3)) + [5, -1, -2]
    p = np.exp(scores-scores.max(1, keepdims=True))
    p = (p / p.sum(1, keepdims=True)).astype(np.float32)
    monkeypatch.setattr(run, 'ensure_idle_cuda', lambda: None)
    monkeypatch.setattr(run, 'load_model', lambda *a: object())
    monkeypatch.setattr(run, 'collect_probabilities', lambda *a: p)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'synthetic_cpu')
    output = source / 'six_candidate'
    run.run(cfg, path, output)
    report = run.read_json(output / 'report.json')
    independent = run.read_json(output / 'independent_verification.json')
    assert report['status'] == 'completed_verified_delivery'
    assert report['raw_remote_disagreement'] > 0
    assert report['raw_replay_passed'] is None and not report['raw_comparison_is_same_decoder']
    assert independent['status'] == 'passed' and independent['full_float64_decision_agreement']
    assert independent['views'] == run.SIX_VIEWS
    assert report['input_sizes'] == [448, 512, 576]
    assert report['platform_score'] is None
    with pytest.raises(ValueError, match='Output exists'):
        run.run(cfg, path, output)
