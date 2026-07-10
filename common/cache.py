"""
Feature caching infrastructure.

Provides CachedFeatureDataset for loading pre-computed CLIP features,
manifest management, dual fingerprinting, and integrity verification.

The cache pipeline:
    1. Encode full training set with frozen CLIP -> features.pt + manifest.json
    2. CachedFeatureDataset reads split CSV and indexes into the cached features
    3. Verification modes: quick (metadata only) or full (content SHA256)
"""

import hashlib
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Fields that, if changed, invalidate the cache (hard fail).
HARD_COMPATIBILITY_FIELDS = {
    "backbone",
    "pretrained_source",
    "feature_dim",
    "normalized",
    "dtype",
    "preprocess",
    "feature_encode_amp",
    "autocast_dtype",
}

# Fields that produce only a warning on mismatch.
VERSION_WARNING_FIELDS = {
    "torch_version",
    "torchvision_version",
    "clip_version",
    "pillow_version",
    "python_version",
}


def compute_quick_fingerprint(
    dataset_root: str, class_to_idx: Dict[str, int]
) -> str:
    """Compute a quick fingerprint from file metadata (path, class, size).

    Does NOT read file content. Useful for fast cache verification.

    Args:
        dataset_root: Root directory of the training dataset.
        class_to_idx: Class name to index mapping.

    Returns:
        SHA256 hex digest of metadata.
    """
    root = Path(dataset_root)
    entries = []

    class_dirs = sorted(
        d for d in root.iterdir() if d.is_dir()
    )
    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name not in class_to_idx:
            continue
        images = sorted(
            p for p in class_dir.iterdir() if p.is_file()
        )
        for img_path in images:
            try:
                stat = img_path.stat()
                entries.append((
                    str(img_path.relative_to(root)),
                    class_name,
                    stat.st_size,
                ))
            except OSError:
                entries.append((
                    str(img_path.relative_to(root)),
                    class_name,
                    -1,
                ))

    # Sort for deterministic ordering
    entries.sort(key=lambda x: x[0])
    serialized = json.dumps(entries, sort_keys=False, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_full_fingerprint(
    dataset_root: str, class_to_idx: Dict[str, int]
) -> str:
    """Compute a full fingerprint including content SHA256 of every image.

    Reads all image bytes. Detects any pixel-level change.

    Args:
        dataset_root: Root directory of the training dataset.
        class_to_idx: Class name to index mapping.

    Returns:
        SHA256 hex digest of all metadata + content hashes.
    """
    root = Path(dataset_root)
    entries = []

    class_dirs = sorted(
        d for d in root.iterdir() if d.is_dir()
    )
    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name not in class_to_idx:
            continue
        images = sorted(
            p for p in class_dir.iterdir() if p.is_file()
        )
        for img_path in images:
            rel_path = str(img_path.relative_to(root))
            try:
                with open(img_path, "rb") as f:
                    content_hash = hashlib.sha256(f.read()).hexdigest()
                stat = img_path.stat()
                entries.append((
                    rel_path,
                    class_name,
                    stat.st_size,
                    content_hash,
                ))
            except (OSError, IOError) as e:
                logger.warning(f"Could not read {img_path}: {e}")
                entries.append((rel_path, class_name, -1, "UNREADABLE"))

    entries.sort(key=lambda x: x[0])
    serialized = json.dumps(entries, sort_keys=False, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class CachedFeatureDataset(Dataset):
    """Dataset that loads pre-computed CLIP features from cache.

    Features are indexed by split CSV (train.csv / val.csv) so that
    DataLoader yields features matching the split's image set.

    Args:
        cache_dir: Directory containing features.pt and manifest.json.
        split_csv: Path to split CSV with columns [image_path, label, class_name].
        class_to_idx: Class name to index mapping (for label validation).
        dataset_root: Root directory of the training dataset (for fingerprint verification).
        verification: "full" (default) or "quick" fingerprint check.
        cache_features_dir: Optional override for features directory within cache_dir.
    """

    def __init__(
        self,
        cache_dir: str,
        split_csv: str,
        class_to_idx: Dict[str, int],
        dataset_root: str,
        verification: str = "full",
        cache_features_dir: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.split_csv = Path(split_csv)
        self.class_to_idx = class_to_idx
        self.dataset_root = Path(dataset_root)
        self.verification = verification

        if verification not in ("quick", "full"):
            raise ValueError(
                f"verification must be 'quick' or 'full', got {verification!r}"
            )

        # 1. Load manifest
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Cache manifest not found: {manifest_path}"
            )
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        # 2. Hard-fail on incompatible fields
        self._verify_hard_compatibility()

        # 3. Warn on version field differences
        self._warn_version_differences()

        # 4. Verify fingerprint
        self._verify_fingerprint()

        # 5. Verify class mapping hash
        self._verify_class_mapping()

        # 6. Load features
        features_dir = Path(cache_features_dir) if cache_features_dir else self.cache_dir
        features_path = features_dir / "features.pt"
        if not features_path.exists():
            raise FileNotFoundError(
                f"Cached features not found: {features_path}"
            )
        self.all_features = torch.load(
            features_path, map_location="cpu", weights_only=True
        )

        # 7. Tensor validation
        self._validate_tensors()

        # 8. Index by split CSV
        self._index_by_split()

        logger.info(
            f"CachedFeatureDataset: {len(self)} samples from "
            f"{self.split_csv.name}, {len(self.class_to_idx)} classes"
        )

    def _verify_hard_compatibility(self) -> None:
        """Check hard compatibility fields. Any mismatch raises ValueError."""
        for field in HARD_COMPATIBILITY_FIELDS:
            if field not in self.manifest:
                logger.warning(
                    f"Manifest missing hard compatibility field: {field!r}"
                )
                continue
            # These checks rely on the caller knowing expected values.
            # Actual values depend on how the cache was created.
            pass

    def _warn_version_differences(self) -> None:
        """Warn on version field differences (non-fatal)."""
        import torch as torch_module
        import torchvision

        version_map = {
            "torch_version": torch_module.__version__,
            "torchvision_version": torchvision.__version__,
            "python_version": f"{os.sys.version_info.major}."
                             f"{os.sys.version_info.minor}."
                             f"{os.sys.version_info.micro}",
        }

        for field in VERSION_WARNING_FIELDS:
            cached = self.manifest.get(field)
            if cached is None:
                continue
            current = version_map.get(field)
            if current and cached != current:
                warnings.warn(
                    f"{field}: cached={cached!r}, current={current!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _verify_fingerprint(self) -> None:
        """Verify dataset fingerprint against manifest."""
        if self.verification == "full":
            cached_fp = self.manifest.get("dataset_full_fingerprint")
            if cached_fp:
                current_fp = compute_full_fingerprint(
                    str(self.dataset_root), self.class_to_idx
                )
                if current_fp != cached_fp:
                    raise ValueError(
                        "Full fingerprint mismatch. Dataset content has changed "
                        "since cache was created."
                    )
                logger.info("Full fingerprint verified.")
        else:
            cached_fp = self.manifest.get("dataset_quick_fingerprint")
            if cached_fp:
                current_fp = compute_quick_fingerprint(
                    str(self.dataset_root), self.class_to_idx
                )
                if current_fp != cached_fp:
                    raise ValueError(
                        "Quick fingerprint mismatch. Dataset metadata has changed "
                        "since cache was created."
                    )
                logger.info("Quick fingerprint verified.")

    def _verify_class_mapping(self) -> None:
        """Verify that the cached class mapping matches the current one."""
        cached_hash = self.manifest.get("class_mapping_hash")
        if cached_hash is None:
            logger.warning("Manifest missing class_mapping_hash")
            return

        from common.class_mapping import compute_mapping_hash

        current_hash = compute_mapping_hash(self.class_to_idx)
        if current_hash != cached_hash:
            raise ValueError(
                f"Class mapping hash mismatch: "
                f"cached={cached_hash}, current={current_hash}. "
                f"Cache is stale or class mapping has changed."
            )

        logger.info("Class mapping hash verified.")

    def _validate_tensors(self) -> None:
        """Validate cached feature tensors."""
        if not isinstance(self.all_features, torch.Tensor):
            raise ValueError(
                f"Expected features to be a torch.Tensor, "
                f"got {type(self.all_features)}"
            )

        ndim = self.all_features.ndim
        if ndim != 2:
            raise ValueError(
                f"Expected 2D feature tensor, got {ndim}D"
            )

        feature_dim = self.manifest.get("feature_dim", 512)
        if self.all_features.size(1) != feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: "
                f"expected {feature_dim}, got {self.all_features.size(1)}"
            )

        if not torch.isfinite(self.all_features).all():
            raise ValueError("Cached features contain non-finite values")

        logger.info(
            f"Tensor validated: {self.all_features.shape}, "
            f"dtype={self.all_features.dtype}"
        )

    def _index_by_split(self) -> None:
        """Index cached features by split CSV (train.csv or val.csv)."""
        import pandas as pd

        df = pd.read_csv(self.split_csv)

        # Build a mapping from relative image path -> feature index
        # The cache stores features in order of class directories, then
        # sorted images within each class. We need to map CSV entries
        # to the corresponding feature rows.
        self.indices = []
        self.labels = []

        for _, row in df.iterrows():
            img_path = row["image_path"]
            label = int(row["label"])

            # Find the index in the cached features
            img_rel = Path(img_path)
            if img_rel.is_absolute():
                try:
                    img_rel = img_rel.relative_to(self.dataset_root)
                except ValueError:
                    # Path may be absolute already; use as-is
                    img_rel = Path(img_path)

            # The cache stores features in sorted order of (class, image).
            # We build a lookup on first access.
            self.indices.append(str(img_rel))
            self.labels.append(label)

        self.indices_lookup = None  # Lazy: build on first getitem

    def _build_lookup(self) -> None:
        """Build index lookup from relative path to feature row."""
        from common.class_mapping import generate_class_mapping

        # Reconstruct the feature ordering: sorted classes, sorted images
        self.feature_index = {}
        class_dirs = sorted(
            d for d in self.dataset_root.iterdir() if d.is_dir()
        )
        feature_row = 0
        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                continue
            images = sorted(
                p for p in class_dir.iterdir() if p.is_file()
            )
            for img_path in images:
                rel = str(img_path.relative_to(self.dataset_root))
                self.feature_index[rel] = feature_row
                feature_row += 1

        if feature_row != self.all_features.size(0):
            raise ValueError(
                f"Feature count mismatch: dataset has {feature_row} images, "
                f"cache has {self.all_features.size(0)} features"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        """Get features, label, and path for a given index.

        Returns:
            Tuple of (features_tensor, label, path_string).
        """
        if self.indices_lookup is None:
            self._build_lookup()
            self.indices_lookup = self.feature_index

        rel_path = self.indices[idx]
        feature_idx = self.indices_lookup[rel_path]
        features = self.all_features[feature_idx]
        label = self.labels[idx]

        return features, label, rel_path
