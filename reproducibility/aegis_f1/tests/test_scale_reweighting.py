import pytest
import torch

from aegis_clip.scale_reweighting import (
    parse_scale_weights,
    parse_shared_top_k,
    reconstruct_nested_scale_probabilities,
    weighted_scale_probabilities,
)


def test_nested_scale_reconstruction_is_exact() -> None:
    small = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    middle = torch.tensor([[0.4, 0.4, 0.2], [0.1, 0.6, 0.3]])
    large = torch.tensor([[0.2, 0.5, 0.3], [0.3, 0.3, 0.4]])
    pair = (middle + large) / 2.0
    triple = (small + middle + large) / 3.0

    recovered, report = reconstruct_nested_scale_probabilities(
        triple.log(), pair.log(), large.log()
    )

    assert torch.allclose(recovered[0], small, atol=1.0e-7)
    assert torch.allclose(recovered[1], middle, atol=1.0e-7)
    assert torch.allclose(recovered[2], large, atol=1.0e-7)
    assert report["shape"] == [2, 3]


def test_weighted_scale_probabilities_uses_declared_weights() -> None:
    values = (
        torch.tensor([[0.8, 0.2]]),
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([[0.2, 0.8]]),
    )
    fused = weighted_scale_probabilities(values, (0.25, 0.5, 0.25))
    assert torch.allclose(fused, torch.tensor([[0.5, 0.5]]))


def test_invalid_scale_weights_fail_closed() -> None:
    assert parse_scale_weights("128,144,160", "0.25,0.5,0.25") == (
        (128, 144, 160),
        (0.25, 0.5, 0.25),
    )
    with pytest.raises(ValueError, match="sum to one"):
        parse_scale_weights("128,144,160", "0.2,0.2,0.2")
    with pytest.raises(ValueError, match="increasing"):
        parse_scale_weights("144,128,160", "0.25,0.5,0.25")


def test_shared_top_k_is_extracted_and_must_match() -> None:
    modes = [
        "attention_multiscale_flip:topk=3:crops=128-144-160:local_weight=0.4",
        "attention_multiscale_flip:topk=3:crops=144-160:local_weight=0.4",
        "attention_crop_flip:topk=3:crop=160:local_weight=0.4",
    ]
    assert parse_shared_top_k(modes) == 3
    with pytest.raises(ValueError, match="different top-k"):
        parse_shared_top_k([modes[0], modes[1].replace("topk=3", "topk=5")])
    with pytest.raises(ValueError, match="exactly one"):
        parse_shared_top_k(["attention_multiscale_flip:crops=128-144-160"])


def test_materially_negative_reconstruction_fails_closed() -> None:
    triple = torch.tensor([[0.4, 0.6]])
    pair = torch.tensor([[0.1, 0.9]])
    single = torch.tensor([[0.9, 0.1]])
    with pytest.raises(ValueError, match="negative"):
        reconstruct_nested_scale_probabilities(triple.log(), pair.log(), single.log())
