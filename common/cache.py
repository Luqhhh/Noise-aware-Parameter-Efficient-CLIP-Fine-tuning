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
import sys
import time
from pathlib import Path
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from .class_mapping import load_or_generate_mapping
from .clip_utils import encode_frozen_clip_features, load_openai_clip
from .dataset import _find_images_in_dir

logger = logging.getLogger(__name__)

_MISSING = object()  # Sentinel for "key not present" in EXPECTED_HARD_VALUES


def _get_package_version(pkg_name):
    """Get version of an installed package, or None."""
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg_name)
    except Exception:
        return None


def _get_clip_info():
    """Get CLIP package information for the manifest."""
    info = {
        "clip_package": "openai-clip",
        "clip_version": None,
        "clip_commit": None,
        "clip_source_path": None,
    }
    try:
        import clip
        import subprocess

        info["clip_source_path"] = os.path.dirname(os.path.abspath(clip.__file__))
        # Try to get git commit from clip installation
        clip_dir = info["clip_source_path"]
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=clip_dir,
                capture_output=True,
                text=True,
                timeout=5,
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
        self.expected_num_classes = data_cfg.get(
            "expected_num_classes",
            config.get("model", {}).get("num_classes", 500),
        )
        if "expected_num_classes" not in data_cfg and "num_classes" not in config.get("model", {}):
            logger.warning("expected_num_classes not configured, defaulting to 500")

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
        manifest = self._build_manifest(dataset_size, quick_fp, full_fp, class_to_idx)
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
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    img = preprocess(img)
                    batch_images.append(img)
                except Exception as e:
                    logger.warning(f"Skipping {p}: {e}")
                    # Use a zero tensor as placeholder
                    img = torch.zeros(3, 224, 224)
                    batch_images.append(img)

            if not batch_images:
                continue

            images = torch.stack(batch_images).to(self.device)
            features = encode_frozen_clip_features(
                clip_model, images, self.device, use_amp=False
            )
            all_features.append(features.cpu())

        result = torch.cat(all_features, dim=0)
        logger.info(f"Encoded features: shape={result.shape}, dtype={result.dtype}")
        return result

    def _build_manifest(self, dataset_size, quick_fp, full_fp, class_to_idx):
        """Build the manifest dictionary."""
        clip_info = _get_clip_info()

        class_mapping_hash = hashlib.sha256(
            json.dumps(class_to_idx, sort_keys=True).encode()
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
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "clip_package": clip_info["clip_package"],
            "clip_version": clip_info["clip_version"],
            "clip_commit": clip_info["clip_commit"],
            "clip_source_path": clip_info["clip_source_path"],
            "pillow_version": _get_package_version("Pillow")
            or _get_package_version("PIL"),
            "python_version": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
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
        if not Path(class_to_idx_path).exists():
            raise FileNotFoundError(
                f"class_to_idx_path not found: {class_to_idx_path}"
            )
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
            expected = self.EXPECTED_HARD_VALUES.get(field, _MISSING)
            actual = self.manifest.get(field)
            if expected is not _MISSING and actual != expected:
                raise ValueError(
                    f"Cache manifest field '{field}' mismatch: "
                    f"expected {expected!r}, got {actual!r}. "
                    f"Rebuild cache with: python scripts/cache_features.py"
                )

    def _check_version_fields(self):
        """Warn if environment version fields differ from cache."""
        version_checks = {
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
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
        if split_csv is None:
            self.sample_indices = list(range(len(self.all_paths)))
            return

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
                # Defense-in-depth: bare filename lookup as a final fallback
                # even though cached paths always contain directory prefixes.
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
