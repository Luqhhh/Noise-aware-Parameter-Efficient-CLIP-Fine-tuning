"""Guard the authorized sum against averaging, drift, and invalid delivery."""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_v2_sixview_logit_sum as run
import verify_v2_sixview_logit_sum as verify
from tests.test_v2_fixed_prior import delivered, dump


def config():
    return run.load_config(ROOT / "configs/v2_sixview_logit_sum_20261003.json")


@pytest.mark.parametrize("field,value", [("reduction", "mean_view_logits"), ("bias_iterations", 201),
    ("bias_strength", .8), ("image_sizes", [448, 512, 768]), ("training", True),
    ("promotion_baseline_percent", 74.0)])
def test_fixed_protocol(tmp_path, field, value):
    cfg = config()
    cfg[field] = value
    dump(tmp_path / "bad.json", cfg)
    with pytest.raises(ValueError, match="protocol"):
        run.load_config(tmp_path / "bad.json")


def test_collect_real_image_exact_sum_and_flip_pairs(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    pixels = np.zeros((700, 900, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(900, dtype=np.uint16) % 255
    Image.fromarray(pixels).save(tmp_path / "a.png")
    images_seen, logits_seen = [], []
    class Model:
        head = SimpleNamespace(out_features=3)
        def __call__(self, images):
            images_seen.append(images.clone())
            logits = torch.tensor([[20, -3, -2] if len(images_seen) == 1 else [-1, 2, 1]], dtype=torch.float32)
            logits_seen.append(logits)
            return logits
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    monkeypatch.setattr(run, "amp", nullcontext)
    original = run.DataLoader
    def loader(dataset, **kwargs):
        kwargs.update(num_workers=0, pin_memory=False)
        kwargs.pop("prefetch_factor")
        return original(dataset, **kwargs)
    monkeypatch.setattr(run, "DataLoader", loader)
    scores, mean = run.collect_views(Model(), tmp_path, ["a.png"], config(), tmp_path)
    assert [x.shape[-1] for x in images_seen] == [448, 448, 512, 512, 576, 576]
    for i in (0, 2, 4):
        torch.testing.assert_close(images_seen[i].flip(3), images_seen[i+1], atol=0, rtol=0)
    np.testing.assert_array_equal(scores, sum(logits_seen).numpy())
    np.testing.assert_allclose(mean, sum(x.softmax(1).numpy() for x in logits_seen)/6, atol=0, rtol=0)
    assert scores.argmax(1).item() != mean.argmax(1).item()
    np.testing.assert_array_equal(np.load(tmp_path / "view_logits.npy"), torch.stack(logits_seen).numpy())


def test_native_replay_rejects_drift(tmp_path):
    cfg = config()
    previous = np.array([[.8, .2]], np.float32)
    path = tmp_path / "native.npy"
    np.save(path, previous)
    cfg.update(incumbent_probabilities=str(path), incumbent_probabilities_sha256=run.sha256_file(path))
    assert run.check_native_replay(previous, cfg)["passed"]
    assert not run.check_native_replay(previous[:, ::-1], cfg)["passed"]


def test_independent_sum_accounts_for_ieee_rounding():
    views = np.array([1, 2**-24, 2**-24, 0, 0, 0], dtype=np.float32).reshape(6, 1, 1)
    actual = np.zeros((1, 1), dtype=np.float32)
    for view in views:
        actual += view
    expected, report = verify.verify_sum(views, actual)
    np.testing.assert_array_equal(actual, expected)
    assert report["float32_rounding_entries"] == 1
    with pytest.raises(AssertionError):
        verify.verify_sum(views, actual/6)


def test_probability_control_uses_float32_reduction_bound():
    expected = np.array([[.9, .1]], dtype=np.float64)
    rounded = expected + np.array([[3e-7, -3e-7]])
    report = verify.probability_rounding_report(rounded, expected, 750, 6)
    assert report["max_absolute_error"] > 2e-7
    with pytest.raises(AssertionError):
        verify.probability_rounding_report(expected + [[.001, -.001]], expected, 750, 6)


def test_complete_delivery_and_independent_sum_rejection(delivered, monkeypatch):
    cfg, source, _ = delivered
    cfg.update(decoder_profile="multiscale_flip_six", views=run.parent.SIX_VIEWS, image_sizes=[448, 512, 576],
               reduction="sum_view_logits", promotion_baseline_percent=74.6688387992735,
               native_replay_max_probability_error=.01, native_replay_max_disagreement_fraction=.001)
    rng = np.random.default_rng(51)
    views = torch.from_numpy(rng.normal(size=(6, 9, 3)).astype(np.float32)).bfloat16().float().numpy()
    scores = views.sum(0)
    mean = sum(torch.from_numpy(x).softmax(1).numpy() for x in views)/6
    native = source / "native.npy"
    np.save(native, mean)
    cfg.update(incumbent_probabilities=str(native), incumbent_probabilities_sha256=run.sha256_file(native))
    path = source / "sum.json"
    dump(path, cfg)
    def collect(model, root, files, cfg, output):
        np.save(output / "view_logits.npy", views)
        np.save(output / "summed_logits.npy", scores)
        np.save(output / "replayed_mean_probabilities.npy", mean)
        return scores, mean
    monkeypatch.setattr(run.parent, "ensure_idle_cuda", lambda: None)
    monkeypatch.setattr(run.parent, "load_model", lambda *a: object())
    monkeypatch.setattr(run, "collect_views", collect)
    monkeypatch.setattr(run, "cold_replay", lambda *a: dict(status="passed", rows=9, views=6, max_absolute_error=0.0))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "synthetic_cpu")
    output = source / "candidate"
    run.run(cfg, path, output)
    report = run.parent.read_json(output / "report.json")
    assert report["status"] == "completed_verified_delivery"
    assert run.parent.read_json(output / "independent_verification.json")["full_float64_decision_agreement"]
    with pytest.raises(ValueError, match="Output exists"):
        run.run(cfg, path, output)
    # Update the hash too: the independent arithmetic must reject a mean, not only a changed file.
    np.save(output / "summed_logits.npy", scores/6)
    report["cache_sha256"]["summed_logits.npy"] = run.sha256_file(output / "summed_logits.npy")
    dump(output / "report.json", report)
    with pytest.raises(AssertionError):
        verify.verify(cfg, path, output)
