"""Actual handed-over checkpoint and images; opt-in to local real-asset checks."""
import importlib
import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest
import torch

LOCATIONS = os.environ.get("LP_LORA768_LOCATIONS")
pytestmark = pytest.mark.skipif(not LOCATIONS, reason="Requires exact handover and official local images")


def implementation():
    assert importlib.util.find_spec("lp_lora768") is not None, "Independent 768 continuation entry point is missing"
    return importlib.import_module("lp_lora768.inputs")


def test_real_parent_zero_update_first128():
    api = implementation()
    context = api.Context(Path(LOCATIONS))
    model = api.initial_model(context, "lora_and_head")
    assert len(model.adapted_modules) == 48
    assert tuple(model.head.weight.shape) == (len(context.classes), 768)
    assert model.visual.proj is None
    predictions, _ = api.center_predictions(context, model, context.val[:128], batch_size=8)
    np.testing.assert_array_equal(predictions, context.frozen["unbalanced768"][:128])
    assert context.inputs_report["positive_weight_rows"] == 119074
    assert context.inputs_report["all_ordered_labels_and_paths_match"]


def test_frozen_control_export_restores_all_lora():
    api = implementation()
    context = api.Context(Path(LOCATIONS))
    control = api.initial_model(context, "head_only")
    complete = api.adapter_head_state(control)
    assert len([k for k in complete if "lora_" in k]) == 96
    assert all(not p.requires_grad for p in control.visual.parameters())
    restored = api.initial_model(context, "lora_and_head")
    api.restore_adapter_head(restored, complete)
    for key, value in api.adapter_head_state(restored).items():
        torch.testing.assert_close(value, complete[key], rtol=0, atol=0)
    assert context.parent["selected_policy"] == "frozen_visual_head_last20"


def test_real_paired_rng_reliability_and_updates():
    api = implementation()
    assert importlib.util.find_spec("lp_lora768.paired") is not None, "Independent paired RNG/update runtime is missing"
    paired = importlib.import_module("lp_lora768.paired")
    context = api.Context(Path(LOCATIONS))
    batches = []
    for perturbation in (0, 997):
        torch.rand(perturbation)
        loader = paired.training_loader(context, epoch=1)
        batches.append(next(iter(loader)))
    torch.testing.assert_close(batches[0][0], batches[1][0], rtol=0, atol=0)
    torch.testing.assert_close(batches[0][1], batches[1][1], rtol=0, atol=0)
    permutation, lam = paired.mix_plan(context, 1, 1, len(batches[0][1]))
    original = context.targets["weights"][batches[0][1]]
    mixed = lam*original+(1-lam)*original[permutation]
    torch.testing.assert_close(mixed.sum(), original.sum(), rtol=1e-6, atol=1e-6)
    states = {}
    for arm in context.protocol["arms"]:
        model = api.initial_model(context, arm)
        before = api.adapter_head_state(model)
        optimizer = paired.optimizer_for(model, context)
        model.train()
        paired.update_once(context, model, optimizer, *batches[0], epoch=1, batch=1)
        after = api.adapter_head_state(model)
        visual_changed = any(not torch.equal(before[k], after[k]) for k in before if "lora_" in k)
        assert visual_changed == (arm == "lora_and_head")
        assert not torch.equal(before["head.weight"], after["head.weight"])
        assert len(after) == len(context.parent["selected_state"])
        states[arm] = after
        del model, optimizer
        torch.cuda.empty_cache()
    assert len(states["head_only"]) == len(states["lora_and_head"])


def test_real_control_cold_checkpoint_keeps_frozen_lora(tmp_path):
    api = implementation()
    assert importlib.util.find_spec("lp_lora768.runtime") is not None, "Cold-checkpoint runtime is missing"
    runtime = importlib.import_module("lp_lora768.runtime")
    context = api.Context(Path(LOCATIONS))
    model = api.initial_model(context, "head_only")
    state = api.adapter_head_state(model)
    checkpoint = tmp_path / "probe.pt"
    runtime.export_checkpoint(context, "head_only", state, checkpoint, epoch=0, policy="probe", updates=0)
    cold, payload = runtime.cold_load(context, "head_only", checkpoint)
    assert payload["complete"] is False
    assert set(payload["selected_state"]) == set(context.parent["selected_state"])
    for key, value in api.adapter_head_state(cold).items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    predictions, _ = api.center_predictions(context, cold, context.val[:128], batch_size=8)
    np.testing.assert_array_equal(predictions, context.frozen["unbalanced768"][:128])
