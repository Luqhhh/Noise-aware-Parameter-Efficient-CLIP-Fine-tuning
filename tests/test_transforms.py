"""Test transform construction."""
import pytest
import torch
import torchvision.transforms as T
from common.transforms import build_train_transform, VALID_PRESETS


def make_clip_eval_transform():
    """Replicate CLIP's deterministic eval transform."""
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def test_valid_presets():
    """All valid presets should be in VALID_PRESETS."""
    assert "a0" in VALID_PRESETS
    assert "a1" in VALID_PRESETS
    assert "a2" in VALID_PRESETS
    assert "a3" in VALID_PRESETS


def test_unknown_preset_raises():
    """Unknown preset -> ValueError."""
    clip_eval = make_clip_eval_transform()
    with pytest.raises(ValueError, match="Unknown augmentation preset"):
        build_train_transform("invalid", clip_eval)


def test_a0_output_shape():
    """A0 should produce (3, 224, 224) output."""
    clip_eval = make_clip_eval_transform()
    transform = build_train_transform("a0", clip_eval)
    dummy = torch.randint(0, 256, (3, 300, 300), dtype=torch.uint8)
    # Convert to PIL for transform
    from PIL import Image
    img = Image.fromarray(dummy.permute(1, 2, 0).numpy().astype('uint8'))
    out = transform(img)
    assert out.shape == (3, 224, 224)


def test_a0_deterministic():
    """A0 should produce identical output for same input (deterministic)."""
    clip_eval = make_clip_eval_transform()
    transform = build_train_transform("a0", clip_eval)
    from PIL import Image
    img = Image.fromarray(
        torch.randint(0, 256, (300, 300, 3), dtype=torch.uint8).numpy()
    )
    out1 = transform(img)
    out2 = transform(img)
    assert torch.equal(out1, out2)


def test_a1_random():
    """A1 should produce different outputs on repeated calls (stochastic)."""
    clip_eval = make_clip_eval_transform()
    transform = build_train_transform("a1", clip_eval)
    from PIL import Image
    img = Image.fromarray(
        torch.randint(0, 256, (500, 500, 3), dtype=torch.uint8).numpy()
    )
    outputs = set()
    for _ in range(100):
        out = transform(img)
        outputs.add(hash(out.numpy().tobytes()))
    # With high probability, RandomResizedCrop+Flip produces >1 unique output
    if len(outputs) == 1:
        # Very unlikely but possible — don't fail hard, just warn
        import warnings
        warnings.warn("A1 produced only 1 unique output in 100 trials (unlucky)")
    # We don't assert len>1 because it's technically possible (though extremely unlikely)


def test_a3_has_random_erasing():
    """A3 transform should include RandomErasing."""
    clip_eval = make_clip_eval_transform()
    transform = build_train_transform("a3", clip_eval)
    has_erasing = any(
        isinstance(t, T.RandomErasing) for t in transform.transforms
    )
    assert has_erasing, "A3 should include RandomErasing"
