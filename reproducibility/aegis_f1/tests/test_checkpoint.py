import torch

from aegis_clip.checkpoint import resume_checkpoint, save_checkpoint
from aegis_clip.losses import EarlyLearningRegularizer
from aegis_clip.trust_subspace import OnlineTrustGradientSubspace


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def effective_spec(self) -> dict:
        return {"backbone": "ViT-B/32", "num_classes": 1}


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
