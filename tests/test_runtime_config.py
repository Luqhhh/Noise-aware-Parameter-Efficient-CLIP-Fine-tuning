from argparse import Namespace

from common.runtime_config import resolve_runtime_args


def make_args(**kwargs):
    values = {
        "experiment_id": None,
        "mode": None,
        "augmentation_preset": None,
        "head_type": None,
        "use_cached_features": None,
    }
    values.update(kwargs)
    return Namespace(**values)


def test_yaml_values_are_used_when_cli_is_absent():
    config = {
        "experiment": {
            "id": "E2",
            "mode": "dev",
            "head_type": "linear",
            "augmentation_preset": "a1",
        },
        "model": {
            "use_cached_features": False,
        },
    }

    args = resolve_runtime_args(make_args(), config)

    assert args.experiment_id == "E2"
    assert args.mode == "dev"
    assert args.head_type == "linear"
    assert args.augmentation_preset == "a1"
    assert args.use_cached_features is False


def test_cli_explicit_values_override_yaml():
    config = {
        "experiment": {
            "id": "E2",
            "mode": "dev",
            "head_type": "linear",
            "augmentation_preset": "a1",
        },
        "model": {
            "use_cached_features": False,
        },
    }

    args = resolve_runtime_args(
        make_args(
            experiment_id="E1",
            head_type="cosine",
            augmentation_preset="a0",
            use_cached_features=True,
        ),
        config,
    )

    assert args.experiment_id == "E1"
    assert args.head_type == "cosine"
    assert args.augmentation_preset == "a0"
    assert args.use_cached_features is True
