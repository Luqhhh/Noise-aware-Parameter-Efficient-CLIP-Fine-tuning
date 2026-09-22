"""Opt-in real-device checkpoint/AMP check: AEGIS_TEST_NPU=1 pytest this file."""
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("AEGIS_TEST_NPU") != "1", reason="requires explicit NPU allocation")


def test_cpu_loader_workers_after_npu_initialization():
    from aegis_clip.device import resolve_device
    from aegis_clip.runtime import set_seed, seed_worker
    device = resolve_device("npu:0")
    set_seed(42)
    dataset = torch.utils.data.TensorDataset(torch.arange(16).reshape(4, 4))
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=4,
                                       worker_init_fn=seed_worker, pin_memory=False)
    (batch,) = next(iter(loader))
    torch.testing.assert_close(batch.to(device).cpu(), dataset.tensors[0])


@pytest.mark.parametrize("recipe", ["ft_npu", "lora"])
def test_official_clip_amp_checkpoint_and_rng_roundtrip(tmp_path, recipe):
    from aegis_clip.device import resolve_device
    from aegis_clip.config import load_config
    from aegis_clip.model import build_model
    from aegis_clip.runtime import set_seed
    from aegis_clip.checkpoint import save_checkpoint, resume_checkpoint, build_from_checkpoint

    device = resolve_device("npu:0")
    set_seed(42)
    config = load_config(Path(__file__).resolve().parents[3] / f"configs/rematch750_{recipe}.yaml")
    config["train"]["device"] = "npu:0"
    # Synthetic engineering check; formal provenance is exercised by the full run.
    config["data"].pop("dataset_manifest")
    model, _ = build_model(config, device)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=3e-6, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("npu", init_scale=128)
    generator = torch.Generator().manual_seed(42)
    images = torch.randn(4, 3, 224, 224, device=device)
    before = model.classifier.weight.detach().clone()
    visual_before = {name: value.detach().clone() for name, value in model.visual.named_parameters()
                     if value.requires_grad}
    with torch.autocast("npu", dtype=torch.float16):
        logits = model(images=images)
        loss = torch.nn.functional.cross_entropy(logits.float(), torch.arange(4, device=device))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    assert not torch.equal(before, model.classifier.weight)
    assert any(not torch.equal(visual_before[name], value) for name, value in model.visual.named_parameters()
               if name in visual_before)
    path = tmp_path / "npu.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, epoch=1, global_step=1, best_selector=0.0,
                    config=config, metrics={}, adaptive_cap_state=None,
                    data_generator_state=generator.get_state())
    expected_rng = torch.rand(16, device=device)
    resume_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                      scaler=scaler, device=device, data_generator=generator)
    assert torch.equal(expected_rng, torch.rand(16, device=device))
    model.eval()
    with torch.no_grad():
        expected = model(images=images).cpu()
    restored, _, _ = build_from_checkpoint(path, device)
    restored.eval()
    with torch.no_grad():
        actual = restored(images=images).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert scheduler.last_epoch == 1
    assert scaler.get_scale() == 128
