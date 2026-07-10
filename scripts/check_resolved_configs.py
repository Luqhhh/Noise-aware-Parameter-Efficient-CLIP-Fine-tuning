from argparse import Namespace
from pathlib import Path

from common.runtime_config import resolve_runtime_args
from common.utils import load_config


EXPECTED = {
    "b0_regression.yaml": ("B0", "linear", "a0", False),
    "e0_hyper_search.yaml": ("E0", "linear", "a0", True),
    "e1_hyper_search.yaml": ("E1", "cosine", "a0", True),
    "e2_augmentation.yaml": ("E2", "linear", "a1", False),
    "e3_augmentation.yaml": ("E3", "linear", "a2", False),
    "e4_augmentation.yaml": ("E4", "linear", "a3", False),
    "e5_combined.yaml": ("E5", "cosine", "a3", False),
}


def empty_args():
    return Namespace(
        experiment_id=None,
        mode=None,
        augmentation_preset=None,
        head_type=None,
        use_cached_features=None,
    )


def main():
    for filename, expected in EXPECTED.items():
        path = Path("configs") / filename
        config = load_config(str(path))
        args = resolve_runtime_args(empty_args(), config)

        actual = (
            args.experiment_id,
            args.head_type,
            args.augmentation_preset,
            args.use_cached_features,
        )

        if actual != expected:
            raise AssertionError(
                f"{filename}: expected={expected}, actual={actual}"
            )

        print(f"[PASS] {filename}: {actual}")


if __name__ == "__main__":
    main()
