# Baseline Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 4 improvements (data augmentation, cosine classifier, feature caching, multi-split validation) across B0/E0-E5/C0-C2 experiments with dev/confirm/final-fit pipeline.

**Architecture:** New common modules (`clip_utils.py`, `class_mapping.py`, `transforms.py`, `cache.py`, `evaluation.py`) provide shared infrastructure. Existing `experiments/baseline/` is refactored to use them. New `experiments/cosine/` adds the cosine classifier. Training script supports modes (dev/confirm/final_fit), epoch freezing, and rich checkpoint metadata. Feature caching encodes full dataset once with dual-fingerprint manifest.

**Tech Stack:** PyTorch, OpenAI CLIP, torchvision, pandas, PyYAML

## Global Constraints

- `drop_last=False` everywhere
- Production code uses explicit exceptions (`ValueError`/`RuntimeError`), not `assert`
- `split_seed` and `train_seed` are separate concepts
- Canonical class mapping from `data/preliminary/metadata/`, used by ALL stages
- Feature caching ONLY for augmentation=a0 AND freeze_clip=True; B0 MUST use online encoding
- C0/C1/C2 reuse E1's tuned lr/wd/scheduler/batch_size/train_seed/max_epochs
- E2/E3/E4 use E0's tuned lr/wd/scheduler/batch_size
- E0/E1 equal-budget: 9 trials each (3 lr × 3 wd)
- Cached features batch size must NOT silently increase for E0/E1 main experiments
- Checkpoint embeds class_to_idx, idx_to_class, head_type, augmentation_preset, trained_epochs, epoch metadata
- Final submission: ONE model on FULL training set, csv.writer, no header, `img.jpg,0001`

---

## Phase 1: Core Infrastructure (Tasks 1-4)

### Task 1: CLIP Utilities Module

**Files:**
- Create: `common/clip_utils.py`
- Create: `tests/test_clip_utils.py`

**Interfaces:**
- Produces: `load_openai_clip(device, model_name="ViT-B/32", pretrained_source="openai") -> Tuple[clip_model, preprocess_fn]`
- Produces: `encode_frozen_clip_features(clip_model, images, device, use_amp=False) -> Tensor`
- Produces: `ALLOWED_MODEL_NAME = "ViT-B/32"`, `ALLOWED_PRETRAINED_SOURCE = "openai"`

- [ ] **Step 1: Create `common/clip_utils.py`**

```python
"""
CLIP model loading and frozen-feature encoding.

Centralizes CLIP loading so model_name and pretrained_source are validated
in exactly one place. Also provides the single canonical encode path used
by both online training and feature caching.
"""

import logging
from typing import Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ALLOWED_MODEL_NAME = "ViT-B/32"
ALLOWED_PRETRAINED_SOURCE = "openai"


def load_openai_clip(device, model_name=ALLOWED_MODEL_NAME,
                     pretrained_source=ALLOWED_PRETRAINED_SOURCE):
    """Load CLIP model from OpenAI, validating model name and pretrained source.

    Args:
        device: torch device.
        model_name: Must be "ViT-B/32".
        pretrained_source: Must be "openai".

    Returns:
        Tuple of (clip_model, preprocess_fn).

    Raises:
        ValueError: If model_name or pretrained_source is not the allowed value.
    """
    if model_name != ALLOWED_MODEL_NAME:
        raise ValueError(
            f"This project requires {ALLOWED_MODEL_NAME}, got {model_name}"
        )
    if pretrained_source != ALLOWED_PRETRAINED_SOURCE:
        raise ValueError(
            f"Only OpenAI weights are allowed, got {pretrained_source}"
        )

    try:
        import clip
    except ImportError:
        raise ImportError(
            "The 'clip' package is required. Install it with:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    model, preprocess = clip.load(ALLOWED_MODEL_NAME, device=device, jit=False)
    return model, preprocess


@torch.no_grad()
def encode_frozen_clip_features(clip_model, images, device, use_amp=False):
    """Encode images through a frozen CLIP visual encoder.

    Features are L2-normalized. This is the single canonical encoding path
    shared by online training (B0) and offline cache building.

    Args:
        clip_model: CLIP model (from load_openai_clip).
        images: Image batch on the correct device, shape (B, 3, H, W).
        device: torch device (used for autocast device_type).
        use_amp: Whether to use torch.autocast during encoding.

    Returns:
        L2-normalized feature tensor of shape (B, feature_dim), float32.
    """
    with torch.autocast(device_type=device.type, enabled=use_amp):
        features = clip_model.encode_image(images)
    return F.normalize(features.float(), dim=-1)
```

- [ ] **Step 2: Create `tests/test_clip_utils.py`**

```python
"""Test CLIP utilities."""
import pytest
import torch
from common.clip_utils import (
    ALLOWED_MODEL_NAME,
    ALLOWED_PRETRAINED_SOURCE,
    load_openai_clip,
    encode_frozen_clip_features,
)


def test_load_openai_clip_rejects_wrong_model():
    """Non-ViT-B/32 model name -> ValueError."""
    with pytest.raises(ValueError, match="ViT-B/32"):
        load_openai_clip(torch.device("cpu"), model_name="RN50")


def test_load_openai_clip_rejects_wrong_source():
    """Non-openai pretrained source -> ValueError."""
    with pytest.raises(ValueError, match="OpenAI"):
        load_openai_clip(torch.device("cpu"), pretrained_source="laion")


def test_load_openai_clip_accepts_defaults():
    """Default args should not raise."""
    device = torch.device("cpu")
    model, preprocess = load_openai_clip(device)
    assert model is not None
    assert preprocess is not None


def test_encode_frozen_clip_features_output_shape():
    """Output should be (batch, 512), L2-normalized."""
    device = torch.device("cpu")
    model, preprocess = load_openai_clip(device)
    model = model.float()  # ensure float32

    dummy_images = torch.randn(4, 3, 224, 224)
    features = encode_frozen_clip_features(model, dummy_images, device, use_amp=False)

    assert features.shape == (4, 512)
    norms = features.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_encode_frozen_clip_features_no_grad():
    """Encoding should not track gradients."""
    device = torch.device("cpu")
    model, preprocess = load_openai_clip(device)
    model = model.float()

    dummy_images = torch.randn(4, 3, 224, 224)
    features = encode_frozen_clip_features(model, dummy_images, device)
    assert not features.requires_grad
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_clip_utils.py -v
```

- [ ] **Step 4: Commit**

```bash
git add common/clip_utils.py tests/test_clip_utils.py
git commit -m "feat: add common/clip_utils.py with load_openai_clip and encode_frozen_clip_features

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Canonical Class Mapping Module

**Files:**
- Create: `common/class_mapping.py`
- Create: `tests/test_class_mapping.py`

**Interfaces:**
- Produces: `generate_canonical_mapping(train_dir, expected_num_classes) -> Tuple[Dict[str,int], Dict[str,int]]`
- Produces: `load_or_generate_mapping(metadata_dir, train_dir, expected_num_classes, regenerate=False) -> Tuple[Dict, Dict]`
- Produces: `validate_class_directory_names(class_names) -> None`

- [ ] **Step 1: Create `common/class_mapping.py`**

```python
"""
Canonical class mapping — generated once from the full training directory,
stored at data/{stage}/metadata/, and reused by ALL stages (dev/confirm/final-fit).

Lifecycle:
  - not exist -> generate
  - exists and matches -> reuse
  - exists but inconsistent -> ValueError (needs --regenerate-class-mapping)
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def validate_class_directory_names(class_names):
    """Validate that class directory names are 4-digit strings.

    Args:
        class_names: List of class directory names.

    Raises:
        ValueError: If any class name doesn't match the 4-digit format.
    """
    for name in class_names:
        if len(name) != 4 or not name.isdigit():
            raise ValueError(f"Invalid class directory name: {name!r}")
    logger.info(f"Validated {len(class_names)} class directory names.")


def generate_canonical_mapping(train_dir, expected_num_classes):
    """Generate class_to_idx and idx_to_class from a training directory.

    Args:
        train_dir: Path to the training data root (class subdirectories).
        expected_num_classes: Expected number of classes (500 preliminary, etc).

    Returns:
        Tuple of (class_to_idx: Dict[str,int], idx_to_class: Dict[str,str]).

    Raises:
        ValueError: If directory count != expected, or any class name is invalid.
    """
    train_dir = Path(train_dir)
    class_dirs = sorted(
        [p for p in train_dir.iterdir() if p.is_dir()],
        key=lambda x: x.name,
    )
    class_names = [d.name for d in class_dirs]

    validate_class_directory_names(class_names)

    if len(class_names) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes, found {len(class_names)}"
        )

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {str(i): name for name, i in class_to_idx.items()}

    logger.info(
        f"Generated canonical mapping: {len(class_to_idx)} classes "
        f"from {train_dir}"
    )
    return class_to_idx, idx_to_class


def _compute_class_mapping_hash(mapping):
    """Compute a deterministic SHA256 hash of the mapping for cache validation."""
    canonical = json.dumps(
        {k: mapping[k] for k in sorted(mapping.keys())},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_or_generate_mapping(
    metadata_dir, train_dir, expected_num_classes, regenerate=False
):
    """Load an existing canonical mapping or generate a new one.

    Lifecycle:
      - not exist -> generate and save
      - exists and matches train_dir -> load
      - exists but inconsistent -> ValueError (regenerate=True to force)

    Args:
        metadata_dir: Path to store/load mapping JSON files.
        train_dir: Path to training data root.
        expected_num_classes: Expected number of classes.
        regenerate: If True, overwrite existing mapping.

    Returns:
        Tuple of (class_to_idx: Dict[str,int], idx_to_class: Dict[str,str]).

    Raises:
        ValueError: If existing mapping is inconsistent with train_dir.
    """
    metadata_dir = Path(metadata_dir)
    class_to_idx_path = metadata_dir / "class_to_idx.json"
    idx_to_class_path = metadata_dir / "idx_to_class.json"

    if regenerate or not class_to_idx_path.exists():
        metadata_dir.mkdir(parents=True, exist_ok=True)
        class_to_idx, idx_to_class = generate_canonical_mapping(
            train_dir, expected_num_classes
        )
        with open(class_to_idx_path, "w") as f:
            json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
        with open(idx_to_class_path, "w") as f:
            json.dump(idx_to_class, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved canonical mapping to {metadata_dir}")
        return class_to_idx, idx_to_class

    # Load existing
    with open(class_to_idx_path, "r") as f:
        class_to_idx = json.load(f)
    with open(idx_to_class_path, "r") as f:
        idx_to_class = json.load(f)

    # Validate against current directory
    train_dir = Path(train_dir)
    current_dirs = sorted(
        [p.name for p in train_dir.iterdir() if p.is_dir()]
    )

    validate_class_directory_names(current_dirs)

    if len(current_dirs) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes, found {len(current_dirs)}"
        )

    expected_class_to_idx = {name: i for i, name in enumerate(current_dirs)}
    if class_to_idx != expected_class_to_idx:
        raise ValueError(
            f"Cached class mapping is inconsistent with current training directory. "
            f"Re-run with --regenerate-class-mapping to overwrite."
        )

    logger.info(f"Loaded existing canonical mapping from {metadata_dir}")
    return class_to_idx, idx_to_class
```

- [ ] **Step 2: Create `tests/test_class_mapping.py`**

```python
"""Test canonical class mapping."""
import json
import pytest
import tempfile
from pathlib import Path
from common.class_mapping import (
    validate_class_directory_names,
    generate_canonical_mapping,
    load_or_generate_mapping,
)


def make_dummy_train_dir(base_dir, class_names):
    """Create dummy class directories."""
    train_dir = Path(base_dir) / "train"
    train_dir.mkdir(parents=True)
    for name in class_names:
        (train_dir / name).mkdir()
    return train_dir


def test_validate_valid_names():
    """4-digit names pass validation."""
    validate_class_directory_names(["0000", "0001", "0499"])


def test_validate_invalid_length():
    """Non-4-length name -> ValueError."""
    with pytest.raises(ValueError, match="Invalid"):
        validate_class_directory_names(["000"])


def test_validate_invalid_non_digit():
    """Non-numeric name -> ValueError."""
    with pytest.raises(ValueError, match="Invalid"):
        validate_class_directory_names(["abcd"])


def test_generate_canonical_mapping():
    """Generate mapping from dummy train dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        class_names = ["0000", "0001", "0002"]
        train_dir = make_dummy_train_dir(tmpdir, class_names)
        c2i, i2c = generate_canonical_mapping(train_dir, expected_num_classes=3)
        assert c2i == {"0000": 0, "0001": 1, "0002": 2}
        assert i2c == {"0": "0000", "1": "0001", "2": "0002"}


def test_generate_wrong_count():
    """Class count mismatch -> ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = make_dummy_train_dir(tmpdir, ["0000", "0001"])
        with pytest.raises(ValueError, match="Expected 5"):
            generate_canonical_mapping(train_dir, expected_num_classes=5)


def test_load_or_generate_creates_new():
    """When no mapping exists, generate one."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = make_dummy_train_dir(tmpdir, ["0000", "0001"])
        meta_dir = Path(tmpdir) / "metadata"
        c2i, i2c = load_or_generate_mapping(
            meta_dir, train_dir, expected_num_classes=2
        )
        assert c2i == {"0000": 0, "0001": 1}
        assert (meta_dir / "class_to_idx.json").exists()


def test_load_or_generate_reuses_existing():
    """When mapping exists and matches, reuse it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = make_dummy_train_dir(tmpdir, ["0000", "0001"])
        meta_dir = Path(tmpdir) / "metadata"
        c2i_1, _ = load_or_generate_mapping(
            meta_dir, train_dir, expected_num_classes=2
        )
        c2i_2, _ = load_or_generate_mapping(
            meta_dir, train_dir, expected_num_classes=2
        )
        assert c2i_1 == c2i_2


def test_load_or_generate_detects_mismatch():
    """When train dir changes, detect inconsistency."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = make_dummy_train_dir(tmpdir, ["0000", "0001"])
        meta_dir = Path(tmpdir) / "metadata"
        load_or_generate_mapping(meta_dir, train_dir, expected_num_classes=2)

        # Change train dir
        import shutil
        shutil.rmtree(train_dir)
        train_dir2 = make_dummy_train_dir(tmpdir, ["0000", "0002"])
        with pytest.raises(ValueError, match="inconsistent"):
            load_or_generate_mapping(meta_dir, train_dir2, expected_num_classes=2)
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_class_mapping.py -v
```

- [ ] **Step 4: Commit**

```bash
git add common/class_mapping.py tests/test_class_mapping.py
git commit -m "feat: add common/class_mapping.py for canonical class mapping lifecycle

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Transform Construction Module

**Files:**
- Create: `common/transforms.py`
- Create: `tests/test_transforms.py`

**Interfaces:**
- Produces: `build_train_transform(preset, clip_eval_transform) -> Callable`
- Produces: `VALID_PRESETS = {"a0", "a1", "a2", "a3"}`

- [ ] **Step 1: Create `common/transforms.py`**

```python
"""
Training transform construction.

Provides build_train_transform() which composes CLIP's deterministic eval
transform with augmentation presets (A0-A3). The CLIP preprocess is NOT
loaded inside this module — it's passed in from the caller.
"""

import logging
from typing import Callable, Set

import torchvision.transforms as T

logger = logging.getLogger(__name__)

VALID_PRESETS: Set[str] = {"a0", "a1", "a2", "a3"}


def build_train_transform(preset: str, clip_eval_transform: Callable):
    """Build a training transform by composing augmentation presets.

    All presets start from CLIP's eval transform (Resize(224) + CenterCrop(224) +
    ToTensor + Normalize). A0 returns it unchanged.

    Args:
        preset: One of "a0", "a1", "a2", "a3".
        clip_eval_transform: CLIP's deterministic eval preprocess (torchvision
            Compose, typically Resize+CenterCrop+ToTensor+Normalize).

    Returns:
        A torchvision transform (Compose).

    Raises:
        ValueError: If preset is not in VALID_PRESETS.
    """
    if preset not in VALID_PRESETS:
        raise ValueError(
            f"Unknown augmentation preset: {preset!r}. "
            f"Valid presets: {sorted(VALID_PRESETS)}"
        )

    # Extract components from CLIP's eval transform.
    # CLIP's preprocess is typically: Compose([
    #   Resize(224, interpolation=BICUBIC),
    #   CenterCrop(224),
    #   ToTensor(),
    #   Normalize(mean, std),
    # ])
    # We need to replace the Resize+CenterCrop with augmentation equivalents
    # while keeping the Normalize at the end.
    #
    # Strategy: Extract the Normalize transform from the CLIP eval pipeline,
    # then build our own Compose that ends with it.

    # Find the Normalize transform in the CLIP eval pipeline
    normalize_transform = None
    clip_size = 224

    if isinstance(clip_eval_transform, T.Compose):
        transforms_list = list(clip_eval_transform.transforms)
    else:
        transforms_list = [clip_eval_transform]

    for t in transforms_list:
        if isinstance(t, T.Normalize):
            normalize_transform = t
        if isinstance(t, T.Resize):
            clip_size = t.size if isinstance(t.size, int) else t.size[0]

    if normalize_transform is None:
        raise ValueError(
            "Could not find T.Normalize in clip_eval_transform. "
            "The CLIP eval transform must contain Normalize."
        )

    if preset == "a0":
        # A0: Deterministic — same as eval
        return clip_eval_transform

    elif preset == "a1":
        # A1: RandomResizedCrop + RandomHorizontalFlip
        return T.Compose([
            T.RandomResizedCrop(clip_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalize_transform,
        ])

    elif preset == "a2":
        # A2: A1 + ColorJitter
        return T.Compose([
            T.RandomResizedCrop(clip_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            normalize_transform,
        ])

    elif preset == "a3":
        # A3: A2 + RandomErasing (applied AFTER Normalize)
        return T.Compose([
            T.RandomResizedCrop(clip_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            normalize_transform,
            T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ])
```

- [ ] **Step 2: Create `tests/test_transforms.py`**

```python
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
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_transforms.py -v
```

- [ ] **Step 4: Commit**

```bash
git add common/transforms.py tests/test_transforms.py
git commit -m "feat: add common/transforms.py with A0-A3 augmentation presets

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Seed Separation & Deterministic DataLoader

**Files:**
- Modify: `common/utils.py` — add `set_train_seed()`, keep `set_seed()` for split_seed
- Modify: `common/dataset.py` — support `seed_worker` + `generator` for deterministic loading
- Modify: `scripts/split_data.py` — use `split_seed` (from `data.split_seed`), not `data.seed`
- Modify: `experiments/baseline/train.py` — use `set_train_seed()`, `drop_last=False`, deterministic loader

**Interfaces:**
- Produces: `seed_worker(worker_id) -> None` in common/dataset.py
- Produces: `set_train_seed(seed) -> None` in common/utils.py
- Consumes: config keys `data.split_seed` (for splits), `data.train_seed` (for training)

- [ ] **Step 1: Add `set_train_seed` to `common/utils.py`**

In `common/utils.py`, add after the existing `set_seed` function:

```python
def set_train_seed(seed: int) -> None:
    """Set random seeds for training reproducibility.

    Unlike set_seed (used for split generation), this does NOT set
    cudnn.deterministic=True because it significantly slows training.
    It sets the core seeds that ensure DataLoader shuffles and model
    initialization are reproducible.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
```

- [ ] **Step 2: Add `seed_worker` to `common/dataset.py`**

At the top of `common/dataset.py`, add after imports:

```python
import numpy as np

def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker for deterministic augmentation.
    
    Uses the initial seed from torch.utils.data.get_worker_info().
    Caller must set a torch.Generator on the DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

Note: `random` is already imported in `common/dataset.py` — but we need to add it. Actually, `random` is NOT imported in `common/dataset.py`. Let me check... No it's not. We need to add `import random` at the top of `common/dataset.py`.

Edit `common/dataset.py`:
- Add `import random` to the imports
- Add `import numpy as np` to the imports
- Add the `seed_worker` function

- [ ] **Step 3: Modify `scripts/split_data.py` to use `split_seed`**

In `scripts/split_data.py`, change the seed resolution:

Instead of reading `data.seed`, read `data.split_seed` (with fallback to `data.seed` for backward compatibility):

```python
# In main(), change:
if seed is None:
    seed = data_cfg.get("split_seed", data_cfg.get("seed", 42))
```

Also update the `parse_args` help text for `--seed`:
```
help="Random seed for the split (overrides config data.split_seed).",
```

- [ ] **Step 4: Modify `experiments/baseline/train.py` for train_seed, drop_last=False, deterministic loader**

Changes needed:
1. `set_seed(seed)` → `set_train_seed(train_seed)` 
2. DataLoader: `drop_last=False`, add `worker_init_fn=seed_worker`, add `generator=g`
3. Read `train_seed` from config

```python
# In main(), replace set_seed call:
train_seed = config["data"].get("train_seed", config["data"].get("seed", 42))
set_train_seed(train_seed)

# In _build_dataloaders, change DataLoader construction:
g = torch.Generator()
g.manual_seed(train_seed)

train_loader = DataLoader(
    train_dataset,
    batch_size=train_cfg["batch_size"],
    shuffle=True,
    num_workers=train_cfg["num_workers"],
    pin_memory=True,
    drop_last=False,           # Changed from True
    worker_init_fn=seed_worker,
    generator=g,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["eval"]["batch_size"],
    shuffle=False,
    num_workers=train_cfg["num_workers"],
    pin_memory=True,
    drop_last=False,           # Already False, but ensure
    worker_init_fn=seed_worker,
    generator=g,
)
```

- [ ] **Step 5: Update config baseline.yaml**

Add to `data:` section:
```yaml
data:
  split_seed: 42
  train_seed: 42
```

- [ ] **Step 6: Run existing tests to verify no regressions**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 7: Commit**

```bash
git add common/utils.py common/dataset.py scripts/split_data.py experiments/baseline/train.py configs/baseline.yaml
git commit -m "feat: separate split_seed/train_seed, drop_last=False, deterministic DataLoader

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 2: Feature Caching (Tasks 5-7)

### Task 5: Feature Cache Builder & Manifest

**Files:**
- Create: `scripts/cache_features.py`
- Create: `common/cache.py` — `FeatureCacheBuilder`, manifest generation, dual fingerprint

**Interfaces:**
- Produces: `FeatureCacheBuilder` class with `build_cache()` method
- Produces: Cache output at `cache/{stage}/clip_vit_b32_openai/` with `features.pt`, `manifest.json`, `class_to_idx.json`, `idx_to_class.json`
- Produces: `compute_quick_fingerprint(dataset_root) -> str`
- Produces: `compute_full_fingerprint(dataset_root) -> str`

- [ ] **Step 1: Create `common/cache.py`**

```python
"""
Feature caching — encode the FULL training set once with frozen CLIP and
store the features on disk so E0/E1 experiments can train on cached features
instead of re-running CLIP encoding every epoch.

Output per stage: cache/{stage}/clip_vit_b32_openai/
  features.pt            # (N, 512) float32 normalized tensor
  image_paths.json       # [str, ...] POSIX relative paths
  labels.json            # [int, ...] label index per sample
  manifest.json          # Full metadata (backbone, fingerprints, versions)
  class_to_idx.json      # Canonical mapping
  idx_to_class.json      # Inverse mapping
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .class_mapping import load_or_generate_mapping
from .clip_utils import encode_frozen_clip_features, load_openai_clip
from .dataset import IMAGE_EXTENSIONS, _find_images_in_dir

logger = logging.getLogger(__name__)


def _get_package_version(pkg_name):
    """Get version of an installed package, or None."""
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg_name)
    except Exception:
        return None


def _get_clip_info():
    """Get CLIP package information for the manifest."""
    info = {"clip_package": "openai-clip", "clip_version": None, "clip_commit": None, "clip_source_path": None}
    try:
        import clip
        info["clip_source_path"] = os.path.dirname(os.path.abspath(clip.__file__))
        # Try to get git commit from clip installation
        import subprocess
        clip_dir = info["clip_source_path"]
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=clip_dir,
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["clip_commit"] = result.stdout.strip()
        except Exception:
            pass
    except ImportError:
        pass
    return info


def compute_quick_fingerprint(dataset_root):
    """Compute a quick fingerprint from file metadata only (no content read).

    Hashes (rel_path, class_name, file_size) for every image.
    Fast but won't detect content-level corruption.
    """
    dataset_root = Path(dataset_root)
    class_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    hasher = hashlib.sha256()

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = _find_images_in_dir(class_dir)
        for img_path in images:
            rel_path = str(img_path.relative_to(dataset_root))
            file_size = img_path.stat().st_size
            entry = f"{rel_path}|{class_name}|{file_size}"
            hasher.update(entry.encode())

    return hasher.hexdigest()


def compute_full_fingerprint(dataset_root):
    """Compute a full fingerprint from file content SHA256.

    Reads every image file and hashes (rel_path, class_name, file_size, content_sha256).
    Slow but detects any image change.
    """
    dataset_root = Path(dataset_root)
    class_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    hasher = hashlib.sha256()

    for class_dir in tqdm(class_dirs, desc="Full fingerprint"):
        class_name = class_dir.name
        images = _find_images_in_dir(class_dir)
        for img_path in images:
            rel_path = str(img_path.relative_to(dataset_root))
            file_size = img_path.stat().st_size
            content_hash = hashlib.sha256(img_path.read_bytes()).hexdigest()
            entry = f"{rel_path}|{class_name}|{file_size}|{content_hash}"
            hasher.update(entry.encode())

    return hasher.hexdigest()


class FeatureCacheBuilder:
    """Encode full training set with frozen CLIP and cache to disk."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        data_cfg = config["data"]
        self.train_dir = Path(data_cfg["train_dir"])
        self.stage = data_cfg.get("stage", "preliminary")
        self.cache_dir = Path(f"cache/{self.stage}/clip_vit_b32_openai")
        self.expected_num_classes = data_cfg["expected_num_classes"]

    def build(self):
        """Run the full cache build pipeline."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Building feature cache at {self.cache_dir}")

        # Step 1: Canonical class mapping
        class_to_idx, idx_to_class = load_or_generate_mapping(
            metadata_dir=self.cache_dir,
            train_dir=self.train_dir,
            expected_num_classes=self.expected_num_classes,
        )

        # Step 2: Scan all images
        all_images, all_labels, all_rel_paths = self._scan_images(class_to_idx)
        dataset_size = len(all_images)
        logger.info(f"Found {dataset_size} images across {len(class_to_idx)} classes")

        # Step 3: Compute fingerprints (quick first, then full)
        logger.info("Computing quick fingerprint...")
        quick_fp = compute_quick_fingerprint(self.train_dir)
        logger.info(f"Quick fingerprint: {quick_fp[:16]}...")

        logger.info("Computing full fingerprint (this may take a while)...")
        full_fp = compute_full_fingerprint(self.train_dir)
        logger.info(f"Full fingerprint: {full_fp[:16]}...")

        # Step 4: Load CLIP model
        clip_model, preprocess = load_openai_clip(self.device)
        clip_model.visual = clip_model.visual.float()
        clip_model.eval()

        # Step 5: Encode all images
        all_features = self._encode_all(clip_model, preprocess, all_images)

        # Step 6: Save features and labels
        torch.save(all_features, self.cache_dir / "features.pt")
        with open(self.cache_dir / "image_paths.json", "w") as f:
            json.dump(all_rel_paths, f, ensure_ascii=False)
        with open(self.cache_dir / "labels.json", "w") as f:
            json.dump(all_labels, f)

        # Step 7: Save canonical mapping
        with open(self.cache_dir / "class_to_idx.json", "w") as f:
            json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
        with open(self.cache_dir / "idx_to_class.json", "w") as f:
            json.dump(idx_to_class, f, indent=2, ensure_ascii=False)

        # Step 8: Write manifest
        manifest = self._build_manifest(dataset_size, quick_fp, full_fp)
        with open(self.cache_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(f"Cache built: {dataset_size} features saved to {self.cache_dir}")
        return self.cache_dir

    def _scan_images(self, class_to_idx):
        """Scan all class directories and build image/label lists."""
        all_images = []
        all_labels = []
        all_rel_paths = []

        class_dirs = sorted([d for d in self.train_dir.iterdir() if d.is_dir()])
        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in class_to_idx:
                continue
            label = class_to_idx[class_name]
            images = _find_images_in_dir(class_dir)
            for img_path in images:
                all_images.append(img_path)
                all_labels.append(label)
                all_rel_paths.append(str(img_path.relative_to(self.train_dir)))

        return all_images, all_labels, all_rel_paths

    @torch.no_grad()
    def _encode_all(self, clip_model, preprocess, image_paths):
        """Encode all images through frozen CLIP."""
        batch_size = self.config["eval"].get("batch_size", 256)
        all_features = []

        # Simple loop — process one batch at a time
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_indices = []

            for j, p in enumerate(batch_paths):
                try:
                    from PIL import Image
                    img = Image.open(p).convert("RGB")
                    img = preprocess(img)
                    batch_images.append(img)
                    valid_indices.append(j)
                except Exception as e:
                    logger.warning(f"Skipping {p}: {e}")
                    # Use a zero tensor as placeholder
                    img = torch.zeros(3, 224, 224)
                    batch_images.append(img)
                    valid_indices.append(j)

            if not batch_images:
                continue

            images = torch.stack(batch_images).to(self.device)
            features = encode_frozen_clip_features(clip_model, images, self.device, use_amp=False)
            all_features.append(features.cpu())

        result = torch.cat(all_features, dim=0)
        logger.info(f"Encoded features: shape={result.shape}, dtype={result.dtype}")
        return result

    def _build_manifest(self, dataset_size, quick_fp, full_fp):
        """Build the manifest dictionary."""
        import platform
        import sys

        import torch as _torch
        import torchvision as _tv

        clip_info = _get_clip_info()

        class_mapping_hash = hashlib.sha256(
            json.dumps(
                json.loads((self.cache_dir / "class_to_idx.json").read_text()),
                sort_keys=True
            ).encode()
        ).hexdigest()

        return {
            "backbone": "ViT-B/32",
            "pretrained_source": "openai",
            "feature_dim": 512,
            "normalized": True,
            "dtype": "float32",
            "preprocess": "clip_deterministic",
            "dataset_size": dataset_size,
            "num_classes": self.expected_num_classes,
            "dataset_root": str(self.train_dir.resolve()),
            "class_mapping_hash": class_mapping_hash,
            "dataset_quick_fingerprint": quick_fp,
            "dataset_full_fingerprint": full_fp,
            "torch_version": _torch.__version__,
            "torchvision_version": _tv.__version__,
            "clip_package": clip_info["clip_package"],
            "clip_version": clip_info["clip_version"],
            "clip_commit": clip_info["clip_commit"],
            "clip_source_path": clip_info["clip_source_path"],
            "pillow_version": _get_package_version("Pillow") or _get_package_version("PIL"),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "feature_encode_amp": False,
            "autocast_dtype": None,
            "encode_device_type": str(self.device.type),
            "clip_parameter_dtype": "float16",
            "image_resolution": 224,
            "interpolation": "bicubic",
            "clip_mean": [0.48145466, 0.4578275, 0.40821073],
            "clip_std": [0.26862954, 0.26130258, 0.27577711],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
```

- [ ] **Step 2: Create `scripts/cache_features.py`**

```python
#!/usr/bin/env python3
"""
Build the deterministic CLIP feature cache for a given stage.

Encodes the FULL training set once with frozen CLIP ViT-B/32 and saves
features to cache/{stage}/clip_vit_b32_openai/.

Usage:
    python scripts/cache_features.py --config configs/baseline.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from common.cache import FeatureCacheBuilder
from common.utils import load_config

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build CLIP feature cache")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    config = load_config(args.config)

    device_str = args.device or config.get("train", {}).get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    builder = FeatureCacheBuilder(config, device)
    cache_dir = builder.build()
    logger.info(f"Cache complete: {cache_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add common/cache.py scripts/cache_features.py
git commit -m "feat: add FeatureCacheBuilder and cache_features script

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: CachedFeatureDataset with Full Validation

**Files:**
- Modify: `common/cache.py` — append `CachedFeatureDataset` class
- Create: `tests/test_cache.py`

**Interfaces:**
- Produces: `CachedFeatureDataset(cache_dir, split_csv, class_to_idx_path, dataset_root, verification="full") -> Dataset`
  - Returns `(feature_tensor, label)` per sample
  - Validates manifest hard fields → ValueError on mismatch
  - Warns on version field differences
  - Verifies fingerprint (quick or full)
  - Validates tensor: ndim==2, shape[1]==512, dtype==float32, finite, no NaN
  - Per-sample label consistency check

- [ ] **Step 1: Append `CachedFeatureDataset` to `common/cache.py`**

```python
# Append to common/cache.py:

class CachedFeatureDataset(torch.utils.data.Dataset):
    """Dataset that loads pre-computed CLIP features from cache.

    Instead of loading and encoding images online, this dataset reads frozen
    CLIP features directly from disk. Only valid for A0 (deterministic) +
    freeze_clip=True experiments.

    Performs comprehensive validation on init:
      1. Manifest hard-field validation (backbone, pretrained_source, etc.) → ValueError
      2. Version field comparison → warning
      3. Fingerprint verification (quick or full)
      4. class_mapping_hash check
      5. Tensor validation (shape, dtype, finite)
      6. Per-sample label consistency
    """

    HARD_FIELDS = {
        "backbone", "pretrained_source", "feature_dim", "normalized",
        "dtype", "preprocess", "feature_encode_amp", "autocast_dtype",
    }

    EXPECTED_HARD_VALUES = {
        "backbone": "ViT-B/32",
        "pretrained_source": "openai",
        "feature_dim": 512,
        "normalized": True,
        "dtype": "float32",
        "preprocess": "clip_deterministic",
        "feature_encode_amp": False,
        "autocast_dtype": None,
    }

    def __init__(self, cache_dir, split_csv, class_to_idx_path,
                 dataset_root, verification="full"):
        self.cache_dir = Path(cache_dir)
        self.dataset_root = Path(dataset_root)

        # 1. Load manifest
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        # 2. Validate hard fields
        self._validate_hard_fields()

        # 3. Warn on version differences
        self._check_version_fields()

        # 4. Verify fingerprint
        self._verify_fingerprint(verification)

        # 5. Load features and metadata
        self.features = torch.load(self.cache_dir / "features.pt")
        with open(self.cache_dir / "image_paths.json", "r") as f:
            self.all_paths = json.load(f)
        with open(self.cache_dir / "labels.json", "r") as f:
            self.all_labels = json.load(f)

        # 6. Tensor validation
        self._validate_tensors()

        # 7. Load class mapping and verify hash
        with open(class_to_idx_path, "r") as f:
            self.class_to_idx = json.load(f)
        self._validate_class_mapping_hash()

        # 8. Filter by split CSV and check per-sample labels
        self._load_split(split_csv)

        logger.info(
            f"CachedFeatureDataset: {len(self.sample_indices)} samples "
            f"from split {split_csv}"
        )

    def _validate_hard_fields(self):
        """Check that hard compatibility fields match expected values."""
        for field in self.HARD_FIELDS:
            expected = self.EXPECTED_HARD_VALUES.get(field)
            actual = self.manifest.get(field)
            if expected is not None and actual != expected:
                raise ValueError(
                    f"Cache manifest field '{field}' mismatch: "
                    f"expected {expected!r}, got {actual!r}. "
                    f"Rebuild cache with: python scripts/cache_features.py"
                )

    def _check_version_fields(self):
        """Warn if environment version fields differ from cache."""
        import sys
        import torch as _torch
        import torchvision as _tv

        version_checks = {
            "torch_version": _torch.__version__,
            "torchvision_version": _tv.__version__,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
        for field, current in version_checks.items():
            cached = self.manifest.get(field)
            if cached and cached != current:
                logger.warning(
                    f"Cache was built with {field}={cached}, "
                    f"current environment has {field}={current}"
                )

    def _verify_fingerprint(self, verification):
        """Verify dataset fingerprint against the current dataset_root."""
        if verification not in ("full", "quick"):
            raise ValueError(f"verification must be 'full' or 'quick', got {verification!r}")

        fingerprint_key = f"dataset_{verification}_fingerprint"
        cached_fp = self.manifest.get(fingerprint_key)
        if cached_fp is None:
            raise ValueError(f"Manifest missing fingerprint field: {fingerprint_key}")

        logger.info(f"Computing {verification} fingerprint for verification...")
        if verification == "full":
            current_fp = compute_full_fingerprint(self.dataset_root)
        else:
            current_fp = compute_quick_fingerprint(self.dataset_root)

        if current_fp != cached_fp:
            raise ValueError(
                f"{verification.capitalize()} fingerprint mismatch! "
                f"Cache: {cached_fp[:16]}..., Current: {current_fp[:16]}... "
                f"The dataset has changed since the cache was built. "
                f"Rebuild cache with: python scripts/cache_features.py"
            )
        logger.info(f"{verification.capitalize()} fingerprint verified.")

    def _validate_tensors(self):
        """Validate feature tensor integrity."""
        if not isinstance(self.features, torch.Tensor):
            raise ValueError(f"Features must be a torch.Tensor, got {type(self.features)}")
        if self.features.ndim != 2:
            raise ValueError(f"Features must be 2D (N, D), got shape {self.features.shape}")
        if self.features.shape[0] != len(self.all_paths):
            raise ValueError(
                f"Feature count ({self.features.shape[0]}) != path count ({len(self.all_paths)})"
            )
        if self.features.shape[1] != self.manifest["feature_dim"]:
            raise ValueError(
                f"Feature dim ({self.features.shape[1]}) != manifest ({self.manifest['feature_dim']})"
            )
        if self.features.dtype != torch.float32:
            raise ValueError(f"Features must be float32, got {self.features.dtype}")
        if not torch.isfinite(self.features).all():
            raise ValueError("Features contain NaN or Inf values")

        # Check for duplicate paths
        if len(set(self.all_paths)) != len(self.all_paths):
            raise ValueError("Duplicate image paths found in cache")

        logger.info(
            f"Features validated: {self.features.shape}, "
            f"dtype={self.features.dtype}, finite=True"
        )

    def _validate_class_mapping_hash(self):
        """Verify class_mapping_hash matches the cached mapping."""
        cached_hash = self.manifest.get("class_mapping_hash")
        if cached_hash is None:
            return  # Old cache without hash — warn but don't fail

        canonical_str = json.dumps(
            {k: self.class_to_idx[k] for k in sorted(self.class_to_idx.keys())},
            sort_keys=True,
        )
        current_hash = hashlib.sha256(canonical_str.encode()).hexdigest()
        if current_hash != cached_hash:
            raise ValueError(
                f"class_mapping_hash mismatch! "
                f"The class mapping has changed since the cache was built. "
                f"Rebuild cache with: python scripts/cache_features.py"
            )

    def _load_split(self, split_csv):
        """Load split CSV and select corresponding cached features."""
        import pandas as pd
        df = pd.read_csv(split_csv)
        self.sample_indices = []

        path_to_idx = {p: i for i, p in enumerate(self.all_paths)}

        for _, row in df.iterrows():
            img_path = Path(row["image_path"])
            if img_path.is_absolute():
                try:
                    rel_path = str(img_path.relative_to(self.dataset_root))
                except ValueError:
                    # Path not under dataset_root — try using just the filename
                    rel_path = str(img_path)
            else:
                rel_path = str(img_path)

            # Try both the relative path and just the filename
            if rel_path in path_to_idx:
                cache_idx = path_to_idx[rel_path]
            elif img_path.name in path_to_idx:
                cache_idx = path_to_idx[img_path.name]
            else:
                raise ValueError(
                    f"Image path from split CSV not found in cache: {rel_path}"
                )

            # Per-sample label consistency check
            csv_label = int(row["label"])
            cache_label = self.all_labels[cache_idx]
            if csv_label != cache_label:
                raise ValueError(
                    f"Label mismatch for {rel_path}: CSV={csv_label}, cache={cache_label}"
                )

            self.sample_indices.append(cache_idx)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        cache_idx = self.sample_indices[idx]
        return self.features[cache_idx], self.all_labels[cache_idx]
```

- [ ] **Step 2: Create `tests/test_cache.py`**

```python
"""Test feature caching infrastructure."""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from common.cache import (
    compute_quick_fingerprint,
    compute_full_fingerprint,
    CachedFeatureDataset,
)


def make_dummy_cache_dir(base_dir):
    """Create a minimal valid cache directory for testing."""
    cache_dir = Path(base_dir) / "cache" / "preliminary" / "clip_vit_b32_openai"
    cache_dir.mkdir(parents=True)

    # features.pt — 10 samples, 512-dim
    features = torch.randn(10, 512, dtype=torch.float32)
    features = features / features.norm(dim=1, keepdim=True)
    torch.save(features, cache_dir / "features.pt")

    # image_paths.json
    paths = [f"0000/img_{i:04d}.jpg" for i in range(5)] + \
            [f"0001/img_{i:04d}.jpg" for i in range(5)]
    with open(cache_dir / "image_paths.json", "w") as f:
        json.dump(paths, f)

    # labels.json
    labels = [0] * 5 + [1] * 5
    with open(cache_dir / "labels.json", "w") as f:
        json.dump(labels, f)

    # class_to_idx.json
    class_to_idx = {"0000": 0, "0001": 1}
    with open(cache_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f)

    # idx_to_class.json
    with open(cache_dir / "idx_to_class.json", "w") as f:
        json.dump({"0": "0000", "1": "0001"}, f)

    return cache_dir, paths, labels, class_to_idx


def test_cached_dataset_rejects_missing_manifest():
    """CachedFeatureDataset should fail without manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="manifest"):
            CachedFeatureDataset(cache_dir, "fake.csv", "fake.json", tmpdir)


# Note: Full integration tests require actual cached features which
# need CLIP. These tests validate the validation logic.
```

- [ ] **Step 3: Commit**

```bash
git add common/cache.py tests/test_cache.py
git commit -m "feat: add CachedFeatureDataset with full validation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 3: Cosine Classifier (Tasks 7-8)

### Task 7: Cosine Classifier Model

**Files:**
- Create: `experiments/cosine/__init__.py`
- Create: `experiments/cosine/model.py`
- Create: `tests/test_cosine.py`

**Interfaces:**
- Produces: `CosineClassifier` class (extends nn.Module)
- Produces: `build_cosine_model(config, device) -> Tuple[CosineClassifier, preprocess_fn]`

Key design points:
- No bias term
- Weight normalized: `self.weight = nn.Parameter(F.normalize(torch.randn(...), dim=1) * init_scale)`
- `learnable_scale` parameter: if True, `self.logit_scale = nn.Parameter(torch.tensor(init_scale))`; if False, registered as buffer
- `clamp_scale()`: no-op when fixed, clamps to [1, 100] when learnable
- Forward: `F.normalize(features) @ weight.T * clamp(logit_scale)`
- optimizer param_groups: conditionally exclude logit_scale when learnable_scale=False

- [ ] **Step 1: Create `experiments/cosine/__init__.py`**

Empty file.

- [ ] **Step 2: Create `experiments/cosine/model.py`**

```python
"""
Cosine classifier — replaces the linear head with a cosine-similarity-based
classification layer. No bias term. Input features and class prototypes are
both L2-normalized before computing logits.

Supports:
  - Fixed scale: logit_scale is a buffer, not optimized
  - Learnable scale: logit_scale is a nn.Parameter, optimized separately
  - Clamping: scale clamped to [1.0, 100.0] after each optimizer step (learnable only)
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.clip_utils import load_openai_clip

logger = logging.getLogger(__name__)


class CosineClassifier(nn.Module):
    """CLIP ViT-B/32 encoder + cosine classification head.

    Args:
        clip_model: CLIP model from load_openai_clip.
        num_classes: Number of output classes.
        feature_dim: CLIP feature dimensionality (512 for ViT-B/32).
        freeze_clip: Whether to freeze the CLIP backbone.
        init_scale: Initial value for the logit scale (temperature-like).
        learnable_scale: If True, logit_scale is a trainable parameter.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        num_classes: int = 500,
        feature_dim: int = 512,
        freeze_clip: bool = True,
        init_scale: float = 10.0,
        learnable_scale: bool = True,
    ):
        super().__init__()

        if init_scale <= 0:
            raise ValueError(f"init_scale must be positive, got {init_scale}")
        if init_scale > 100:
            raise ValueError(f"init_scale must be <= 100, got {init_scale}")

        self.visual = clip_model.visual
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.freeze_clip = freeze_clip
        self.learnable_scale = learnable_scale
        self.head_type = "cosine"

        # Freeze CLIP backbone
        if freeze_clip:
            for param in self.visual.parameters():
                param.requires_grad = False
            logger.info("CLIP image encoder frozen.")

        # Cosine classifier weight (class prototypes) — no bias
        weight = torch.randn(num_classes, feature_dim)
        weight = F.normalize(weight, dim=1) * init_scale
        self.weight = nn.Parameter(weight)

        # Logit scale (temperature)
        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_buffer("logit_scale", torch.tensor(float(init_scale)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Extract and L2-normalize image features."""
        conv1_dtype = self.visual.conv1.weight.dtype
        images = images.to(dtype=conv1_dtype)

        with torch.set_grad_enabled(not self.freeze_clip):
            features = self.visual(images)

        if features.dim() > 2:
            features = (
                features.mean(dim=[2, 3]) if features.dim() == 4 else features[:, 0]
            )

        features = features.float()
        features = F.normalize(features, p=2, dim=-1)
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode, normalize, compute cosine similarity, scale.

        Returns logits of shape (B, num_classes).
        """
        features = self.encode_image(images)
        # Normalize weight on each forward pass
        weight_norm = F.normalize(self.weight, dim=1)
        logits = features @ weight_norm.T * self.clamp_scale()
        return logits

    def clamp_scale(self):
        """Clamp logit_scale to [1.0, 100.0]. No-op when learnable_scale=False."""
        if self.learnable_scale:
            with torch.no_grad():
                self.logit_scale.clamp_(min=1.0, max=100.0)
        return self.logit_scale

    def get_param_groups(self, lr, weight_decay):
        """Return optimizer param groups.

        When learnable_scale=True: logit_scale gets lr*0.1, no weight decay.
        When learnable_scale=False: only weight is optimized.
        """
        groups = [
            {
                "params": [self.weight],
                "lr": lr,
                "weight_decay": weight_decay,
            },
        ]
        if self.learnable_scale:
            groups.append({
                "params": [self.logit_scale],
                "lr": lr * 0.1,
                "weight_decay": 0.0,
            })
        return groups

    def train(self, mode: bool = True):
        """Override train() to keep CLIP backbone in eval mode when frozen."""
        super().train(mode)
        if self.freeze_clip:
            self.visual.eval()
        return self


def build_cosine_model(config: dict, device: torch.device) -> Tuple[CosineClassifier, callable]:
    """Build CosineClassifier and return CLIP preprocessing function.

    Args:
        config: Must contain model.cos_init_scale, model.cos_learnable_scale,
                model.num_classes, model.feature_dim, model.freeze_clip.
        device: torch device.

    Returns:
        Tuple of (model, preprocess_fn).
    """
    clip_model, preprocess = load_openai_clip(device)

    # Convert visual encoder to float32
    clip_model.visual = clip_model.visual.float()

    model_cfg = config["model"]
    model = CosineClassifier(
        clip_model=clip_model,
        num_classes=model_cfg.get("num_classes", 500),
        feature_dim=model_cfg.get("feature_dim", 512),
        freeze_clip=model_cfg.get("freeze_clip", True),
        init_scale=model_cfg.get("cos_init_scale", 10.0),
        learnable_scale=model_cfg.get("cos_learnable_scale", True),
    )
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"CosineClassifier: {total:,} params, {trainable:,} trainable")

    return model, preprocess
```

- [ ] **Step 3: Create `tests/test_cosine.py`**

```python
"""Test cosine classifier."""
import pytest
import torch
from experiments.cosine.model import CosineClassifier


class MockVisual(torch.nn.Module):
    """Mock CLIP visual encoder."""
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 768, 1)  # Minimal conv for dtype check

    def forward(self, x):
        return torch.randn(x.size(0), 512)  # Return features directly


class MockCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = MockVisual()


def test_cosine_no_bias():
    """Cosine classifier should have no bias parameter."""
    model = CosineClassifier(MockCLIP(), num_classes=10, feature_dim=512)
    assert model.weight is not None
    assert not hasattr(model, 'bias')


def test_cosine_fixed_scale():
    """Fixed scale: logit_scale should be a buffer, not parameter."""
    model = CosineClassifier(
        MockCLIP(), num_classes=10, feature_dim=512,
        init_scale=10.0, learnable_scale=False
    )
    assert not isinstance(model.logit_scale, torch.nn.Parameter)
    assert model.logit_scale.item() == 10.0
    # clamp_scale should be no-op
    s = model.clamp_scale()
    assert s.item() == 10.0


def test_cosine_learnable_scale():
    """Learnable scale: logit_scale should be a Parameter."""
    model = CosineClassifier(
        MockCLIP(), num_classes=10, feature_dim=512,
        init_scale=10.0, learnable_scale=True
    )
    assert isinstance(model.logit_scale, torch.nn.Parameter)
    assert model.logit_scale.item() == 10.0


def test_cosine_clamp():
    """Clamping should work when scale is learnable."""
    model = CosineClassifier(
        MockCLIP(), num_classes=10, feature_dim=512,
        init_scale=10.0, learnable_scale=True
    )
    # Manually set scale to extreme values
    with torch.no_grad():
        model.logit_scale.fill_(200.0)
    clamped = model.clamp_scale()
    assert clamped.item() == 100.0

    with torch.no_grad():
        model.logit_scale.fill_(0.1)
    clamped = model.clamp_scale()
    assert clamped.item() == 1.0


def test_cosine_init_scale_validation():
    """Invalid init_scale -> ValueError."""
    with pytest.raises(ValueError, match="positive"):
        CosineClassifier(MockCLIP(), num_classes=10, init_scale=-1.0)
    with pytest.raises(ValueError, match="<= 100"):
        CosineClassifier(MockCLIP(), num_classes=10, init_scale=200.0)


def test_cosine_forward_shape():
    """Forward pass should produce (B, num_classes) logits."""
    model = CosineClassifier(MockCLIP(), num_classes=10, feature_dim=512)
    model = model.float()
    images = torch.randn(4, 3, 224, 224)
    logits = model(images)
    assert logits.shape == (4, 10)


def test_cosine_param_groups():
    """Check param group structure."""
    model = CosineClassifier(
        MockCLIP(), num_classes=10, feature_dim=512,
        learnable_scale=True
    )
    groups = model.get_param_groups(lr=0.001, weight_decay=0.0001)
    assert len(groups) == 2  # weight group + scale group
    # Scale group has lower lr, no wd
    assert groups[1]["lr"] == 0.0001
    assert groups[1]["weight_decay"] == 0.0


def test_cosine_param_groups_fixed_scale():
    """Fixed scale: only weight group, no scale group."""
    model = CosineClassifier(
        MockCLIP(), num_classes=10, feature_dim=512,
        learnable_scale=False
    )
    groups = model.get_param_groups(lr=0.001, weight_decay=0.0001)
    assert len(groups) == 1  # Only weight group


def test_cosine_train_mode_keeps_backbone_eval():
    """When freeze_clip=True, calling train() should keep visual in eval."""
    model = CosineClassifier(MockCLIP(), freeze_clip=True)
    model.train()
    assert model.training  # Classifier head in train mode
    assert not model.visual.training  # Backbone stays in eval
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_cosine.py -v
```

- [ ] **Step 5: Commit**

```bash
git add experiments/cosine/ tests/test_cosine.py
git commit -m "feat: add CosineClassifier with learnable/fixed scale and param groups

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Refactored Training Infrastructure with Modes

**Files:**
- Rewrite: `experiments/baseline/train.py` — add mode support (dev/confirm/final_fit), epoch freezing, checkpoint metadata, backbone eval enforcement, guard enforcement
- Rewrite: `experiments/baseline/model.py` — use `common/clip_utils.py`, add `train()` override
- Modify: `experiments/baseline/evaluate.py` — use canonical mapping, checkpoint metadata
- Modify: `experiments/baseline/infer.py` — load idx_to_class from checkpoint

**Interfaces:**
- `model.train()` keeps backbone in eval when `freeze_clip=True`
- Checkpoint keys: `model_state_dict`, `class_to_idx`, `idx_to_class`, `head_type`,
  `augmentation_preset`, `trained_epochs`, `epoch_selection_split`,
  `epoch_selection_policy`, `training_mode`, `split_seed`, `dev_best_epoch`,
  `source_dev_best_epoch`, `frozen_train_epochs`, `config`
- Training modes: `dev` (train+val, record best_epoch), `confirm` (train frozen epochs, eval),
  `final_fit` (train frozen epochs, no val, full dataset)
- Guard enforcement: B0+cached, preset!=a0+cached, freeze_clip=False+cached all → ValueError

- [ ] **Step 1: Update `experiments/baseline/model.py` — use clip_utils, add train() override**

Replace `build_model` to use `load_openai_clip` from common:

```python
# In build_model(), replace the clip.load() section:
from common.clip_utils import load_openai_clip

def build_model(config: dict, device: torch.device) -> Tuple[CLIPLinearClassifier, callable]:
    clip_model_name = config["model"]["clip_model_name"]
    num_classes = config["model"]["num_classes"]
    feature_dim = config["model"].get("feature_dim", 512)
    freeze_clip = config["model"].get("freeze_clip", True)

    clip_model, preprocess = load_openai_clip(device)
    clip_model.visual = clip_model.visual.float()

    model = CLIPLinearClassifier(
        clip_model=clip_model,
        num_classes=num_classes,
        feature_dim=feature_dim,
        freeze_clip=freeze_clip,
    )
    model = model.to(device)
    model.head_type = "linear"  # For checkpoint metadata

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model built: {total:,} total params, {trainable:,} trainable")

    return model, preprocess
```

Add `train()` override to `CLIPLinearClassifier`:

```python
def train(self, mode: bool = True):
    """Override train() to keep CLIP backbone in eval when frozen."""
    super().train(mode)
    if self.freeze_clip:
        self.visual.eval()
    return self
```

- [ ] **Step 2: Rewrite `experiments/baseline/train.py` with mode support**

This is the largest single file change. Key additions:

```python
# New imports
from common.class_mapping import load_or_generate_mapping
from common.transforms import build_train_transform, VALID_PRESETS
from common.cache import CachedFeatureDataset

# New arg: --mode (dev/confirm/final_fit)
# New arg: --experiment-id (B0/E0/etc for guard enforcement)
# New arg: --use-cached-features
# New arg: --frozen-epochs (for confirm/final_fit)
# New arg: --augmentation-preset (a0/a1/a2/a3)
# New arg: --source-dev-best-epoch (for confirm/final_fit, carries forward)

def _enforce_guards(experiment_id, use_cached_features, augmentation_preset, freeze_clip):
    """Hard enforcement of feature caching rules."""
    if experiment_id == "B0" and use_cached_features:
        raise ValueError("B0 regression must use the original online encoding path")
    if use_cached_features and augmentation_preset != "a0":
        raise ValueError("Cached features only valid for deterministic A0 preprocessing")
    if use_cached_features and not freeze_clip:
        raise ValueError("Cached features require freeze_clip=True")

def _build_checkpoint_metadata(model, config, mode, args, best_epoch=None):
    """Build the unified checkpoint metadata."""
    meta = {
        "class_to_idx": getattr(model, 'class_to_idx_', None),
        "idx_to_class": getattr(model, 'idx_to_class_', None),
        "head_type": getattr(model, 'head_type', 'linear'),
        "augmentation_preset": args.augmentation_preset,
        "training_mode": mode,
        "split_seed": config["data"].get("split_seed", None),
    }
    
    if mode == "dev":
        meta.update({
            "dev_best_epoch": best_epoch,
            "frozen_train_epochs": best_epoch,
            "trained_epochs": config["train"]["epochs"],
            "epoch_selection_policy": "dev_best_epoch_frozen_before_confirm",
            "epoch_selection_split": config["data"].get("split_seed", 42),
        })
    elif mode in ("confirm", "final_fit"):
        meta.update({
            "source_dev_best_epoch": args.source_dev_best_epoch,
            "frozen_train_epochs": args.frozen_epochs,
            "trained_epochs": args.frozen_epochs,
            "epoch_selection_policy": "dev_best_epoch_frozen_before_confirm",
            "epoch_selection_split": config["data"].get("split_seed", 42),
        })
    
    return meta

def main():
    # ... existing setup ...
    
    # NEW: Read experiment metadata
    experiment_id = args.experiment_id
    mode = args.mode  # "dev", "confirm", "final_fit"
    use_cached = args.use_cached_features
    aug_preset = args.augmentation_preset
    
    # NEW: Enforce guards
    _enforce_guards(experiment_id, use_cached, aug_preset, 
                    config["model"].get("freeze_clip", True))
    
    # NEW: Canonical class mapping
    class_to_idx, idx_to_class = load_or_generate_mapping(
        metadata_dir=config["data"]["class_mapping_path"],
        train_dir=config["data"]["train_dir"],
        expected_num_classes=config["data"]["expected_num_classes"],
    )
    
    # NEW: Build transform with augmentation preset
    _, preprocess = build_model(config, device)
    if aug_preset != "a0" and not use_cached:
        train_transform = build_train_transform(aug_preset, preprocess)
    else:
        train_transform = preprocess
    
    # NEW: Dataset selection (full or split, cached or online)
    if mode == "final_fit":
        dataset = TrainImageDataset(
            data_root=config["data"]["train_dir"],
            split_csv=None,  # Full dataset
            class_to_idx=class_to_idx,
            transform=train_transform,
        )
        val_loader = None  # No validation in final_fit
    elif use_cached:
        dataset = CachedFeatureDataset(
            cache_dir=config["cache"]["cache_dir"],
            split_csv=train_csv,
            class_to_idx_path=config["data"]["class_mapping_path"] + "/class_to_idx.json",
            dataset_root=config["data"]["train_dir"],
            verification=config["cache"].get("verification", "full"),
        )
    else:
        dataset = TrainImageDataset(
            data_root=config["data"]["train_dir"],
            split_csv=train_csv,
            class_to_idx=class_to_idx,
            transform=train_transform,
        )
    
    # NEW: Deterministic DataLoader with drop_last=False
    g = torch.Generator()
    g.manual_seed(config["data"]["train_seed"])
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    # ... setup optimizer/scheduler ...
    
    # NEW: Track best epoch in dev mode
    dev_best_epoch = None
    best_val_acc = 0.0
    
    # NEW: Frozen epochs for confirm/final_fit
    if mode in ("confirm", "final_fit"):
        epochs = args.frozen_epochs  # Use frozen value
    else:
        epochs = train_cfg["epochs"]
    
    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        # ... train_one_epoch ...
        
        if val_loader is not None:
            val_loss, val_acc = validate(...)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                dev_best_epoch = epoch
                # Save best.pt
                ...
    
    # NEW: Save final checkpoint with full metadata
    checkpoint_meta = _build_checkpoint_metadata(
        model, config, mode, args, best_epoch=dev_best_epoch
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        **checkpoint_meta,
        "config": config,
    }
    torch.save(checkpoint, save_dir / "last.pt")
    
    # NEW: Save evaluation JSON for dev/confirm modes
    if mode in ("dev", "confirm"):
        eval_results = {
            "experiment_id": experiment_id,
            "mode": mode,
            "split_seed": config["data"].get("split_seed"),
            "train_seed": config["data"].get("train_seed"),
            "best_val_acc": float(best_val_acc),
            "dev_best_epoch": dev_best_epoch,
            "trained_epochs": epochs,
            "head_type": model.head_type,
            "augmentation_preset": aug_preset,
        }
        with open(save_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
```

- [ ] **Step 3: Update `experiments/baseline/evaluate.py`**

- Use canonical mapping from `common.class_mapping`
- Load `idx_to_class` from checkpoint
- Save evaluation results as JSON

- [ ] **Step 4: Update `experiments/baseline/infer.py`**

- Load `class_to_idx` and `idx_to_class` from checkpoint (not from split_dir)
- Verify checkpoint has the required metadata
- Use `csv.writer` format (already correct)

- [ ] **Step 5: Commit**

```bash
git add experiments/baseline/
git commit -m "feat: refactor training with mode support, epoch freezing, checkpoint metadata

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Cosine Experiment Scripts (train/evaluate/infer)

**Files:**
- Create: `experiments/cosine/train.py`
- Create: `experiments/cosine/evaluate.py`
- Create: `experiments/cosine/infer.py`

**Approach:** Cosine scripts mirror baseline's refactored scripts. Key differences:
- Uses `build_cosine_model` instead of `build_model`
- Optimizer uses `model.get_param_groups(lr, wd)` instead of `get_trainable_parameters()`
- Calls `model.clamp_scale()` after each optimizer step
- Saves `head_type: "cosine"` in checkpoints

- [ ] **Step 1: Create `experiments/cosine/train.py`**

Copy the refactored baseline train.py and modify:
- Import `build_cosine_model` from `.model`
- `model.head_type = "cosine"` (set by the model itself)
- Use `model.get_param_groups(lr, wd)` for optimizer
- Call `model.clamp_scale()` after `optimizer.step()` in the training loop
- Same mode support (dev/confirm/final_fit), epoch freezing, checkpoint metadata
- Same guard enforcement

- [ ] **Step 2: Create `experiments/cosine/evaluate.py`**

Same structure as baseline/evaluate.py but loads `CosineClassifier`.

- [ ] **Step 3: Create `experiments/cosine/infer.py`**

Same as baseline/infer.py but loads `CosineClassifier` and reads mapping from checkpoint.

- [ ] **Step 4: Commit**

```bash
git add experiments/cosine/
git commit -m "feat: add cosine experiment train/evaluate/infer scripts

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Augmentation Experiments

**Files:**
- Create: `experiments/augmentation/__init__.py`
- Create: `experiments/augmentation/train.py`
- Create: `experiments/augmentation/evaluate.py`
- Create: `experiments/augmentation/infer.py`

**Approach:** Augmentation experiments reuse `CLIPLinearClassifier` (linear head) with A1/A2/A3 transforms. The train script is nearly identical to baseline's refactored train.py — the only difference is `augmentation_preset` is set to a1/a2/a3 and `use_cached_features=False` (random augmentation must be re-applied each epoch).

- [ ] **Step 1: Create augmentation experiment scripts**

Create simple wrappers that import from `experiments.baseline` and set `augmentation_preset`. The train script enforces:
- `use_cached_features=False` (random augmentations must be recomputed)
- `augmentation_preset` in {a1, a2, a3}
- Uses E0's tuned lr/wd/scheduler/batch_size from config
- Independent epoch freezing

- [ ] **Step 2: Commit**

```bash
git add experiments/augmentation/
git commit -m "feat: add augmentation experiments (E2-E4) reusing linear head

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 4: Multi-Split & Evaluation (Tasks 11-12)

### Task 11: Multi-Split Support & Evaluation Module

**Files:**
- Modify: `scripts/split_data.py` — support multiple split_seeds
- Create: `common/evaluation.py` — paired delta, confirmation reports
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Add multi-seed split support to `scripts/split_data.py`**

Add `--split-seeds` argument (comma-separated list of seeds). Default generates split_seed=42. When multiple seeds are provided, generates `outputs/{experiment}/splits/seed_{N}/` directories.

- [ ] **Step 2: Create `common/evaluation.py`**

```python
"""
Multi-split evaluation and paired delta reporting.

Computes paired deltas vs E0 (tuned linear baseline) across confirm splits
and produces pooled/confirmation statistics for candidate selection.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_eval_results(results_path):
    """Load eval_results.json from a training run."""
    with open(results_path, "r") as f:
        return json.load(f)


def compute_paired_deltas(
    e0_results: Dict[int, float],  # {split_seed: val_acc}
    candidate_results: Dict[int, float],  # {split_seed: val_acc}
) -> Dict[str, float]:
    """Compute paired deltas: Δ_i = Acc_candidate,i - Acc_E0,i for each split.

    Args:
        e0_results: E0 validation accuracy per split seed.
        candidate_results: Candidate validation accuracy per split seed.

    Returns:
        Dict with keys:
          - deltas: {split_seed: delta}
          - mean_delta: mean across splits
          - std_delta: sample std (ddof=1)
          - min_delta: worst-split delta
          - confirmation_wins: X/2 (number of splits where delta > -0.002)
          - pooled_win: whether mean_delta > 0
    """
    deltas = {}
    for split_seed in e0_results:
        if split_seed in candidate_results:
            deltas[split_seed] = candidate_results[split_seed] - e0_results[split_seed]

    n = len(deltas)
    if n == 0:
        return {"deltas": {}, "mean_delta": 0.0, "confirmation_wins": "0/0"}

    mean_delta = sum(deltas.values()) / n
    std_delta = (
        (sum((d - mean_delta) ** 2 for d in deltas.values()) / (n - 1)) ** 0.5
        if n > 1 else 0.0
    )
    min_delta = min(deltas.values())
    wins = sum(1 for d in deltas.values() if d > -0.002)

    return {
        "deltas": deltas,
        "mean_delta": round(mean_delta, 6),
        "std_delta": round(std_delta, 6) if std_delta else 0.0,
        "min_delta": round(min_delta, 6),
        "confirmation_wins": f"{wins}/{n}",
        "pooled_delta": round(mean_delta, 6),
    }


def apply_candidate_rules(
    candidates: Dict[str, Dict],  # {candidate_id: paired_delta_report}
    elimination_threshold: float = -0.002,
    tie_threshold: float = 0.001,
) -> Tuple[str, Dict]:
    """Apply the pre-specified candidate selection rules.

    Returns:
        (selected_id, selection_report)
    """
    # Step 2: Eliminate candidates that degrade >0.2pp on any split
    survivors = {}
    for cid, report in candidates.items():
        if report["min_delta"] > -0.002:
            survivors[cid] = report

    # Step 3: Eliminate if mean delta <= 0
    survivors = {k: v for k, v in survivors.items() if v["mean_delta"] > 0}

    # Step 4: Fallback to E0
    if not survivors:
        return "E0", {"reason": "no_candidates_survived", "fallback": "E0"}

    # Step 5: Exactly one survivor
    if len(survivors) == 1:
        winner = list(survivors.keys())[0]
        return winner, {"reason": "sole_survivor", "winner": winner}

    # Step 6: Tiebreaker
    sorted_ids = sorted(survivors.keys())
    delta_diff = abs(survivors[sorted_ids[0]]["mean_delta"] - survivors[sorted_ids[1]]["mean_delta"])

    if delta_diff < tie_threshold:
        # Apply deterministic tiebreaker:
        # 1. Fewer inference-time parameters (cosine < linear since no bias)
        # 2. Fewer augmentation components (a0 < a1 < a2 < a3)
        # 3. Higher worst-split delta
        # 4. Lower delta std
        # 5. Lexicographic experiment ID
        # Simplified: just pick by higher mean_delta then lexicographic
        winner = max(survivors.keys(), key=lambda k: (survivors[k]["mean_delta"], k))
        return winner, {"reason": "tiebreaker", "winner": winner, "delta_diff": delta_diff}

    # Otherwise higher mean delta wins
    winner = max(survivors.keys(), key=lambda k: survivors[k]["mean_delta"])
    return winner, {"reason": "higher_mean_delta", "winner": winner}
```

- [ ] **Step 3: Commit**

```bash
git add common/evaluation.py tests/test_evaluation.py scripts/split_data.py
git commit -m "feat: add multi-split evaluation and paired delta reporting

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: B0 Regression Fixture

**Files:**
- Create: `experiments/baseline/b0_regression.py` — B0 regression protocol
- Create: `configs/b0_regression.yaml`

**Key requirements from spec:**
- B0 uses original hyperparameters (lr=1e-3, wd=1e-4, epochs=20, AdamW, CosineAnnealingLR, warmup=1, AMP, batch_size=128, max_grad_norm=1.0)
- B0 does NOT adopt new tuning rules (no 9-trial search, no per-method epoch freezing)
- B0 MUST use online encoding (not cached features) — enforced with ValueError
- Saves resolved config as JSON fixture for regression comparison
- Records `observed_best_epoch` (not `frozen_train_epochs`)

- [ ] **Step 1: Create B0 regression fixture script**

The B0 regression is implemented as a thin wrapper that calls the refactored train.py with B0-specific settings. It saves a resolved config fixture:

```python
# experiments/baseline/b0_regression.py
"""
B0 regression protocol: reproduce the original baseline with exact hyperparameters.

This verifies that infrastructure refactoring didn't change the baseline results.
B0 MUST use online encoding and the original training protocol.
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

B0_FIXTURE = {
    "optimizer": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "epochs": 20,
    "scheduler": "CosineAnnealingLR",
    "warmup_epochs": 1,
    "amp": True,
    "max_grad_norm": 1.0,
    "checkpoint_policy": "best_val",
    "split_seed": 42,
    "train_seed": 42,
}

def save_b0_fixture(output_dir):
    """Save the resolved B0 config for regression comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = output_dir / "b0_regression_fixture.json"
    with open(fixture_path, "w") as f:
        json.dump(B0_FIXTURE, f, indent=2)
    logger.info(f"B0 regression fixture saved to {fixture_path}")
    return fixture_path
```

- [ ] **Step 2: B0-specific config (`configs/b0_regression.yaml`)**

Same as baseline.yaml but with explicit B0 fields and `experiment.id: B0`.

- [ ] **Step 3: Commit**

```bash
git add experiments/baseline/b0_regression.py configs/b0_regression.yaml
git commit -m "feat: add B0 regression fixture and config

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 13: Submission with Explicit Exceptions

**Files:**
- Modify: `common/submission.py` — full coverage checks with explicit exceptions

- [ ] **Step 1: Add pre-submission checks to `common/submission.py`**

Add a new function `validate_submission_coverage` that performs all the checks from the spec using explicit exceptions:

```python
def validate_submission_coverage(test_dir, results_csv_path):
    """Validate submission coverage with explicit exceptions.

    Checks:
      1. Test basename uniqueness
      2. Prediction count == test image count
      3. No duplicate image names in predictions
      4. Set equality (no missing, no extra)
      5. Class name format (4-digit, no whitespace)

    Args:
        test_dir: Path to test image directory.
        results_csv_path: Path to pred_results.csv.

    Raises:
        ValueError: On any validation failure.
        FileNotFoundError: If paths don't exist.
    """
    test_dir = Path(test_dir)
    results_csv_path = Path(results_csv_path)

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if not results_csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv_path}")

    # Collect test image basenames
    test_image_paths = []
    for ext in IMAGE_EXTENSIONS:
        test_image_paths.extend(test_dir.glob(f"*{ext}"))
        test_image_paths.extend(test_dir.glob(f"*{ext.upper()}"))

    test_names = [p.name for p in test_image_paths]

    # Check basename uniqueness
    if len(test_names) != len(set(test_names)):
        raise ValueError("Test set contains duplicate basenames")

    expected_names = set(test_names)

    # Read predictions
    import csv
    with open(results_csv_path, "r") as f:
        reader = csv.reader(f)
        submission_rows = list(reader)

    predicted_names = [row[0].strip() for row in submission_rows if row]

    # Check count
    if len(predicted_names) != len(expected_names):
        raise ValueError(
            f"Prediction count mismatch: "
            f"got {len(predicted_names)}, expected {len(expected_names)}"
        )

    # Check duplicates
    if len(predicted_names) != len(set(predicted_names)):
        raise ValueError("Submission contains duplicate image names")

    # Check set equality
    predicted_set = set(predicted_names)
    if predicted_set != expected_names:
        missing = sorted(expected_names - predicted_set)
        extra = sorted(predicted_set - expected_names)
        raise ValueError(
            f"Submission coverage mismatch: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    # Check class name format
    for row in submission_rows:
        if not row:
            continue
        image_name = row[0].strip()
        class_name = row[1].strip() if len(row) > 1 else ""

        if class_name != class_name.strip():
            raise ValueError(f"Class name contains whitespace: {class_name!r}")
        if len(class_name) != 4 or not class_name.isdigit():
            raise ValueError(f"Invalid class name for {image_name}: {class_name!r}")

    logger.info(f"Submission validation passed: {len(predicted_names)} predictions")
```

- [ ] **Step 2: Update `generate_submission` to use csv.writer instead of manual format**

```python
def generate_submission(raw_csv_path: str, out_dir: str) -> tuple:
    # ... existing validation ...
    
    # Generate pred_results.csv using csv.writer (no header)
    results_path = out_dir / "pred_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        for _, row in df.iterrows():
            img_name = row["image_name"]
            pred_label = str(row["pred_label"]).zfill(4)
            writer.writerow([img_name, pred_label])
    # Note: csv.writer adds comma+space by default, writes "img.jpg,0001"
```

- [ ] **Step 3: Commit**

```bash
git add common/submission.py
git commit -m "feat: add submission coverage validation with explicit exceptions

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 5: Configs & Integration (Tasks 14-15)

### Task 14: Configuration Files for All Experiments

**Files:**
- Modify: `configs/baseline.yaml` — add new fields (split_seed, train_seed, stage, expected_num_classes, class_mapping_path, augmentation_preset, cache section)
- Create: `configs/e0_hyper_search.yaml`
- Create: `configs/e1_hyper_search.yaml`
- Create: `configs/e2_augmentation.yaml`
- Create: `configs/e3_augmentation.yaml`
- Create: `configs/e4_augmentation.yaml`
- Create: `configs/e5_combined.yaml`
- Create: `configs/c0_cosine_scale.yaml`
- Create: `configs/c1_cosine_scale.yaml`
- Create: `configs/c2_cosine_scale.yaml`

- [ ] **Step 1: Update `configs/baseline.yaml` with all new fields**

```yaml
experiment:
  id: B0
  mode: dev
  head_type: linear
  augmentation_preset: a0

data:
  stage: preliminary
  image_extensions: [.jpg, .jpeg, .png, .bmp, .webp]
  split_seed: 42
  train_seed: 42
  split_dir: outputs/baselines/baseline/splits
  test_dir: data/preliminary/test
  train_dir: data/preliminary/train
  val_ratio: 0.1
  expected_num_classes: 500
  class_mapping_path: data/preliminary/metadata
  use_full_training_set: false

model:
  clip_model_name: ViT-B/32
  feature_dim: 512
  freeze_clip: true
  num_classes: 500
  use_cached_features: false

cache:
  cache_dir: cache/preliminary/clip_vit_b32_openai
  verification: full

eval:
  batch_size: 256

output:
  log_dir: outputs/baselines/baseline/logs
  submission_dir: outputs/baselines/baseline/submissions

train:
  amp: true
  batch_size: 128
  device: cuda
  epochs: 20
  image_size: 224
  lr: 0.001
  max_grad_norm: 1.0
  num_workers: 8
  save_dir: outputs/baselines/baseline/checkpoints
  scheduler: cosine
  warmup_epochs: 1
  weight_decay: 0.0001
```

- [ ] **Step 2: Create E0 config**

```yaml
# configs/e0_hyper_search.yaml
# E0: Linear+A0, best of 9-trial lr x wd search
experiment:
  id: E0
  mode: dev
  head_type: linear
  augmentation_preset: a0

data:
  stage: preliminary
  split_seed: 42
  train_seed: 42
  split_dir: outputs/e0/splits
  test_dir: data/preliminary/test
  train_dir: data/preliminary/train
  val_ratio: 0.1
  expected_num_classes: 500
  class_mapping_path: data/preliminary/metadata
  use_full_training_set: false

model:
  clip_model_name: ViT-B/32
  feature_dim: 512
  freeze_clip: true
  num_classes: 500
  use_cached_features: true  # E0 can use cached features (A0 + frozen CLIP)

cache:
  cache_dir: cache/preliminary/clip_vit_b32_openai
  verification: full

# Hyperparameter search grid (3 lr x 3 wd = 9 trials)
hyper_search:
  lr_values: [0.0001, 0.0005, 0.001]
  wd_values: [0.00001, 0.0001, 0.001]

# ... rest same as baseline
```

- [ ] **Step 3: Create E1-E5, C0-C2 configs**

Each config sets the appropriate `experiment.id`, `experiment.head_type`, `experiment.augmentation_preset`, `model.use_cached_features`, `model.cos_init_scale`, `model.cos_learnable_scale`.

E2-E4 inherit E0's lr/wd. C0-C2 inherit E1's lr/wd and only vary `cos_init_scale` and `cos_learnable_scale`.

- [ ] **Step 4: Commit**

```bash
git add configs/
git commit -m "feat: add configs for all experiments (B0, E0-E5, C0-C2)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 15: Integration Tests & Acceptance Verification

**Files:**
- Create: `tests/test_integration.py` — smoke test that exercises the full pipeline
- Modify: `tests/test_label_mapping.py` — update to use canonical mapping
- Create: `scripts/run_acceptance.py` — script that runs all acceptance criteria checks

- [ ] **Step 1: Create integration smoke test**

```python
"""Integration smoke test: tiny dataset, train one epoch, infer, generate submission."""
import subprocess
import sys
from pathlib import Path


def test_full_pipeline_on_tiny_dataset():
    """Smoke test: make_tiny_dataset -> split -> train(1 epoch) -> infer -> submission."""
    repo_root = Path(__file__).resolve().parent.parent
    
    # Step 1: Create tiny dataset
    result = subprocess.run([
        sys.executable, "tools/make_tiny_dataset.py",
        "--train_dir", "data/tiny_smoke/train",
        "--test_dir", "data/tiny_smoke/test",
        "--num_classes", "5",
        "--images_per_class", "4",
        "--num_test", "3",
    ], cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, f"make_tiny_dataset failed: {result.stderr}"
    
    # Step 2: Generate canonical mapping
    # (tests common/class_mapping.py)
    
    # Step 3: Split data
    # (tests split_seed support)
    
    # Step 4: Train 1 epoch
    # (tests training loop with new infrastructure)
    
    # Step 5: Infer
    # (tests inference, checkpoint metadata)
    
    # Step 6: Generate submission
    # (tests submission with explicit exceptions)
    
    # Step 7: Validate submission
    # (tests check_submission.py)
```

- [ ] **Step 2: Create acceptance verification script**

`scripts/run_acceptance.py` runs all acceptance criteria from the spec:
- AC-1.x: Feature caching (cache build, manifest validation, guard enforcement)
- AC-2.x: Seeds & multi-split (split_seed≠train_seed, output isolation, determinism)
- AC-3.x: Cosine classifier (no bias, grad flow, clamp, param groups)
- AC-4.x: Data augmentation (preset validation, randomness, lr/wd inheritance)
- AC-5.x: Engineering & regression (B0 fixture, backbone eval, importability, submission compliance)

- [ ] **Step 3: Run all tests**

```bash
python -m pytest tests/ -v --tb=short
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py scripts/run_acceptance.py
git commit -m "feat: add integration tests and acceptance verification script

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Execution Order

Tasks should be implemented in dependency order:

```
Phase 1 (parallel): Task 1 (clip_utils), Task 2 (class_mapping)
Phase 2 (parallel): Task 3 (transforms), Task 4 (seed separation)
Phase 3: Task 5 (cache builder) → Task 6 (CachedFeatureDataset)
Phase 4: Task 7 (cosine model) → Task 8 (refactored training)
Phase 5 (parallel): Task 9 (cosine scripts), Task 10 (augmentation scripts)
Phase 6: Task 11 (multi-split eval) → Task 12 (B0 regression)
Phase 7: Task 13 (submission) → Task 14 (configs) → Task 15 (integration)
```

Tasks within a phase can run in parallel. Tasks 1+2 are independent. Tasks 3+4 are independent (but both depend on 1+2 being complete). Tasks 9+10 depend on Task 8.

---
