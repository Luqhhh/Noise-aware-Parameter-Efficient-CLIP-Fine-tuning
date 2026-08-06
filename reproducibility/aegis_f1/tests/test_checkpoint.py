import torch

from aegis_clip.checkpoint import (
    _merge_parametrized_state_for_plain_model,
    load_initial_weights,
    resume_checkpoint,
    save_checkpoint,
)
from aegis_clip.losses import EarlyLearningRegularizer
from aegis_clip.trust_subspace import OnlineTrustGradientSubspace


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def effective_spec(self) -> dict:
        return {"backbone": "ViT-B/32", "num_classes": 1}


def test_merge_parametrized_state_matches_effective_lora_weights() -> None:
    spec = {
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "mlp_lora_rank": 2,
        "mlp_lora_alpha": 4.0,
    }
    original_attn = torch.randn(12, 4)
    q_a = torch.randn(4, 4)
    q_b = torch.randn(4, 4)
    v_a = torch.randn(4, 4)
    v_b = torch.randn(4, 4)
    original_mlp = torch.randn(6, 8)
    mlp_a = torch.randn(2, 8)
    mlp_b = torch.randn(6, 2)

    state = {
        "visual.attn.parametrizations.in_proj_weight.original": original_attn,
        "visual.attn.parametrizations.in_proj_weight.0.q_A": q_a,
        "visual.attn.parametrizations.in_proj_weight.0.q_B": q_b,
        "visual.attn.parametrizations.in_proj_weight.0.v_A": v_a,
        "visual.attn.parametrizations.in_proj_weight.0.v_B": v_b,
        "visual.mlp.c_fc.parametrizations.weight.original": original_mlp,
        "visual.mlp.c_fc.parametrizations.weight.0.lora_A": mlp_a,
        "visual.mlp.c_fc.parametrizations.weight.0.lora_B": mlp_b,
        "visual.positional_embedding": torch.randn(4, 4),
    }

    merged = _merge_parametrized_state_for_plain_model(state, spec)

    expected_attn = original_attn + 2.0 * torch.cat(
        [q_b @ q_a, torch.zeros_like(q_b @ q_a), v_b @ v_a], dim=0
    )
    expected_mlp = original_mlp + 2.0 * (mlp_b @ mlp_a)
    assert torch.allclose(merged["visual.attn.in_proj_weight"], expected_attn)
    assert torch.allclose(merged["visual.mlp.c_fc.weight"], expected_mlp)
    assert torch.equal(
        merged["visual.positional_embedding"],
        state["visual.positional_embedding"],
    )
    assert not any(".parametrizations." in name for name in merged)


def test_checkpoint_restores_cpu_generator_state(tmp_path) -> None:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    generator = torch.Generator().manual_seed(123)
    elr = EarlyLearningRegularizer(2, 2)
    elr.update_and_loss(torch.tensor([0]), torch.tensor([[3.0, -3.0]]))
    saved_elr_targets = elr.targets.clone()
    saved_generator_state = generator.get_state().clone()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=1,
        global_step=4,
        best_selector=0.5,
        config={"project": {"experiment_id": "tiny"}},
        metrics={},
        adaptive_cap_state=None,
        data_generator_state=saved_generator_state,
        elr_state_dict=elr.state_dict(),
    )

    with torch.no_grad():
        model.weight.fill_(9.0)
    torch.rand(3, generator=generator)
    elr.targets.zero_()
    state = resume_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=torch.device("cpu"),
        elr_regularizer=elr,
        data_generator=generator,
    )
    assert state["epoch"] == 1
    assert model.weight.item() == 1.0
    assert torch.equal(generator.get_state(), saved_generator_state)
    assert torch.equal(elr.targets, saved_elr_targets)


def test_checkpoint_restores_trust_subspace_auxiliary(tmp_path) -> None:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    generator = torch.Generator().manual_seed(123)
    subspace = OnlineTrustGradientSubspace(max_rank=2)
    subspace.update(torch.tensor([1.0, 0.0]))
    path = tmp_path / "subspace_checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=1,
        global_step=4,
        best_selector=0.5,
        config={"project": {"experiment_id": "tiny"}},
        metrics={},
        adaptive_cap_state=None,
        data_generator_state=generator.get_state(),
        training_aux_state=subspace.state_dict(),
    )

    restored = OnlineTrustGradientSubspace(max_rank=2)
    resume_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=torch.device("cpu"),
        data_generator=generator,
        training_auxiliary=restored,
    )
    projection, ratio = restored.project(torch.tensor([2.0, 3.0]))
    assert torch.equal(projection, torch.tensor([2.0, 0.0]))
    assert 0.0 < ratio < 1.0


class ResolutionAwareModel(torch.nn.Module):
    """Minimal model exposing a visual position embedding for resolution loading."""

    def __init__(self, pos_shape: tuple[int, int]) -> None:
        super().__init__()
        self.peft_mode = "frozen"
        self.visual = torch.nn.Module()
        self.visual.positional_embedding = torch.nn.Parameter(
            torch.randn(*pos_shape)
        )
        self.classifier = torch.nn.Linear(pos_shape[1], 10)


def test_load_initial_weights_interpolates_position_embedding(tmp_path) -> None:
    # 224px = 7x7 patches + CLS = 50 tokens; 288px = 9x9 + CLS = 82 tokens.
    source = ResolutionAwareModel((50, 512))
    path = tmp_path / "source.pt"
    torch.save({"model_state_dict": source.state_dict()}, path)
    target = ResolutionAwareModel((82, 512))
    original_positions = target.visual.positional_embedding[1:].clone()
    load_initial_weights(target, path, torch.device("cpu"))
    assert tuple(target.visual.positional_embedding.shape) == (82, 512)
    # CLS token is preserved exactly by the bicubic interpolation.
    assert torch.allclose(
        target.visual.positional_embedding[0],
        source.visual.positional_embedding[0],
        atol=1.0e-6,
    )
    # The checkpoint's 49-grid positions were interpolated into the 81-grid
    # (proves the load actually replaced the target's random init).
    assert not torch.allclose(
        target.visual.positional_embedding[1:],
        original_positions,
        atol=1.0e-4,
    )
