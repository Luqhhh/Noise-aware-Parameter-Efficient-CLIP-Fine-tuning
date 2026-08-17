from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from aegis_clip.pace_protocol import (
    EXPECTED_PROTOCOL_SECTIONS,
    PacePreflightError,
    build_exact_group_artifact,
    freeze_group_sha_in_config,
    load_pace_protocol,
    verify_protocol_assets,
)
from aegis_clip.cli.prepare_pace_group_artifact import prepare_pace_group_artifact
from aegis_clip.runtime import sha256_file


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixed_sections(tmp_path: Path) -> dict[str, object]:
    return {
        "spec": {
            "path": str(tmp_path / "pace-k2-design.md"),
            "sha256": "668eca56452b3e2dc55ae1e8a9fceea626765f518b65688a8454d697e8f64fc9",
            "commit": "84f4705",
        },
        "parent_assets": {
            "checkpoint": str(tmp_path / "best.pt"),
            "checkpoint_sha256": "26916fd3ec96311dcab7a637f416ad3455cf7c78087844d408a38958f168962a",
            "trust_bundle": str(tmp_path / "trust.pt"),
            "trust_bundle_sha256": (
                "ff8688a818219d3737715cd7dc9d11d014e7a5e973fa68f966c0ce370a88a246"
            ),
            "submission_dir": str(tmp_path / "submission"),
            "prediction_csv_sha256": (
                "c6ed3e6a7f63c49a9b821f0e09222a153d926702de6f4c42505781aa7ae89fdd"
            ),
            "submission_zip_sha256": (
                "6333375eea0f0b7575b833de16daf89c897df521c9eaa3f64a71e546c5ec4dc6"
            ),
            "submission_manifest_sha256": (
                "0ddff9c4e03b0bfefe3c8671d388679a4a33dbbcc547b9f88d1b440fdca1c06e"
            ),
        },
        "parent": {
            "peft_mode": "visual_lora_mlp_lora",
            "classifier_mode": "linear",
            "feature_adapter": "Identity",
            "batch_size": 128,
            "amp": False,
            "crop_sizes": [128, 144, 160],
            "local_scale_weights": [0.45, 0.50, 0.05],
            "flip_weight": 0.50,
            "local_weight": 0.40,
            "global_temperature": 1.5,
            "local_temperature": 1.5,
            "local_top_k": 5,
            "part_top_patches": 8,
            "part_temperature": 0.07,
            "prior": {
                "target": "uniform",
                "strength": 0.85,
                "max_iterations": 50,
                "tolerance": 1.0e-6,
                "damping": 0.5,
            },
        },
        "evidence": {
            "view_order": [
                "original_128",
                "original_144",
                "original_160",
                "flipped_128",
                "flipped_144",
                "flipped_160",
            ],
            "tail_size": 7,
            "weight_norm_epsilon": 1.0e-12,
            "minimum_positive_views": 4,
            "require_orientation_positive": True,
            "require_leave_one_scale_positive": True,
            "linear_gate_atol": 1.0e-5,
            "linear_gate_rtol": 1.0e-5,
            "antisymmetry_atol": 1.0e-6,
            "antisymmetry_rtol": 1.0e-6,
        },
        "crossfit": {
            "seed": 42,
            "outer_folds": 5,
            "inner_folds": 3,
            "sklearn_version": "1.9.0",
            "numpy_version": "2.5.1",
            "sort_key": "canonical_path",
        },
        "beta_solver": {
            "dtype": "float64",
            "initial_upper": 1.0,
            "maximum_upper": 1048576.0,
            "maximum_iterations": 100,
            "interval_tolerance": 1.0e-12,
            "allow_zero": True,
            "intercept": False,
            "margin_coefficient": 1.0,
        },
        "threshold": {
            "modes": ["all_switch", "finite", "no_switch"],
            "strict_comparator": ">",
            "minimum_accuracy_changing_precision": 0.6,
            "minimum_wilson_lower": 0.5,
            "wilson_z": 1.959963984540054,
            "no_switch_reasons": [
                "no_eligible",
                "no_qualified_candidate",
                "inner_fit_failed",
                "full_refit_failed",
                "finite_mapping_failed",
                "final_oof_no_switch",
            ],
        },
        "promotion": {
            "minimum_raw_micro_delta_pp": 0.2,
            "minimum_clean_core_micro_delta_pp": 0.2,
            "minimum_net_correct_fraction": 0.002,
            "minimum_nonnegative_outer_folds": 4,
            "minimum_worst_fold_delta_pp": -0.1,
            "minimum_macro_delta_pp": -0.05,
            "bootstrap_draws": 10000,
            "bootstrap_seed": 42,
            "bootstrap_rng": "PCG64",
            "bootstrap_quantile_method": "linear",
        },
        "execution": {
            "device": "cuda",
            "amp": False,
            "batch_size": 128,
            "expected_validation_samples": 10316,
            "expected_test_samples": 24967,
        },
        "outputs": {"root": str(tmp_path / "outputs")},
    }


def _write_protocol(tmp_path: Path, *, cross_split_duplicate: bool = False) -> Path:
    train_root = tmp_path / "train"
    (train_root / "0000").mkdir(parents=True)
    (train_root / "0001").mkdir(parents=True)
    (train_root / "0000" / "a.jpg").write_bytes(b"duplicate")
    (train_root / "0000" / "b.jpg").write_bytes(b"duplicate")
    val_bytes = b"duplicate" if cross_split_duplicate else b"validation"
    (train_root / "0001" / "c.jpg").write_bytes(val_bytes)

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(
        [
            {"image_path": "train/0000/a.jpg", "label": 0},
            {"image_path": "train/0000/b.jpg", "label": 0},
        ]
    ).to_csv(train_csv, index=False, lineterminator="\n")
    pd.DataFrame(
        [{"image_path": "train/0001/c.jpg", "label": 1}]
    ).to_csv(val_csv, index=False, lineterminator="\n")

    class_to_idx = tmp_path / "class_to_idx.json"
    class_to_idx.write_text('{"0000": 0, "0001": 1}\n', encoding="utf-8")
    idx_to_class = tmp_path / "idx_to_class.json"
    idx_to_class.write_text('{"0": "0000", "1": "0001"}\n', encoding="utf-8")
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text('{"train_count": 2, "val_count": 1}\n', encoding="utf-8")

    config = {
        "protocol_id": "pace_k2_r2_parttoken_v1",
        "assets": {
            "train_csv": str(train_csv),
            "train_csv_sha256": _digest(train_csv),
            "val_csv": str(val_csv),
            "val_csv_sha256": _digest(val_csv),
            "class_to_idx": str(class_to_idx),
            "class_to_idx_sha256": _digest(class_to_idx),
            "idx_to_class": str(idx_to_class),
            "idx_to_class_sha256": _digest(idx_to_class),
            "split_manifest": str(manifest),
            "split_manifest_sha256": _digest(manifest),
            "train_root": str(train_root),
        },
        "group_artifact": {
            "state": "bootstrap_unfrozen",
            "sha256": None,
            "output_path": str(
                tmp_path
                / "protocol_artifacts"
                / "pace_k2_r2_parttoken"
                / "content_groups.json"
            ),
            "report_path": str(
                tmp_path
                / "protocol_artifacts"
                / "pace_k2_r2_parttoken"
                / "group_artifact_report.json"
            ),
            "expected_total_rows": 3,
            "expected_unique_groups": 2 if not cross_split_duplicate else 1,
            "expected_duplicate_extras": 1 if not cross_split_duplicate else 2,
        },
    }
    config.update(_fixed_sections(tmp_path))
    path = tmp_path / "pace.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_group_builder_is_deterministic_and_freezes_digest(tmp_path: Path) -> None:
    config_path = _write_protocol(tmp_path)
    protocol = load_pace_protocol(config_path, allow_unfrozen_group=True)

    first = build_exact_group_artifact(protocol, hash_workers=2)
    first_bytes = protocol.group_artifact.output_path.read_bytes()
    second = build_exact_group_artifact(protocol, hash_workers=1)

    assert first.map_sha256 == second.map_sha256
    assert protocol.group_artifact.output_path.read_bytes() == first_bytes
    assert first.total_rows == 3
    assert first.unique_groups == 2
    assert first.duplicate_extras == 1
    assert first.cross_split_overlap == 0
    mapping = json.loads(first_bytes)
    assert list(mapping) == ["0000/a.jpg", "0000/b.jpg", "0001/c.jpg"]
    assert mapping["0000/a.jpg"] == mapping["0000/b.jpg"]
    assert all(len(value) == 64 and value == value.lower() for value in mapping.values())

    freeze_group_sha_in_config(config_path, first.map_sha256)
    frozen = load_pace_protocol(config_path)
    assert frozen.group_artifact.state == "frozen"
    assert frozen.group_artifact.sha256 == first.map_sha256
    assert sha256_file(frozen.group_artifact.output_path) == first.map_sha256


def test_group_builder_rejects_duplicate_group_across_split(tmp_path: Path) -> None:
    protocol = load_pace_protocol(
        _write_protocol(tmp_path, cross_split_duplicate=True),
        allow_unfrozen_group=True,
    )

    with pytest.raises(PacePreflightError, match="crosses train and validation"):
        build_exact_group_artifact(protocol, hash_workers=1)


def test_protocol_rejects_changed_asset_bytes(tmp_path: Path) -> None:
    config_path = _write_protocol(tmp_path)
    protocol = load_pace_protocol(config_path, allow_unfrozen_group=True)
    protocol.assets.val_csv.write_bytes(b"changed")

    with pytest.raises(PacePreflightError, match="val_csv SHA-256 mismatch"):
        build_exact_group_artifact(protocol, hash_workers=1)


def test_public_preflight_verifies_split_assets_and_gates_model_assets(
    tmp_path: Path,
) -> None:
    protocol = load_pace_protocol(
        _write_protocol(tmp_path), allow_unfrozen_group=True
    )

    audit = verify_protocol_assets(protocol, require_model_assets=False)
    assert audit.protocol_sha256 == sha256_file(protocol.config_path)
    assert audit.split_assets_verified is True
    assert audit.model_assets_verified is False

    with pytest.raises(PacePreflightError, match="checkpoint is missing"):
        verify_protocol_assets(protocol, require_model_assets=True)


def test_downstream_loader_rejects_unfrozen_group_state(tmp_path: Path) -> None:
    config_path = _write_protocol(tmp_path)

    with pytest.raises(PacePreflightError, match="group artifact is not frozen"):
        load_pace_protocol(config_path)


def test_group_output_must_use_tracked_protocol_directory(tmp_path: Path) -> None:
    config_path = _write_protocol(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["group_artifact"]["output_path"] = str(tmp_path / "artifacts" / "groups.json")
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PacePreflightError, match="protocol_artifacts/pace_k2_r2_parttoken"):
        load_pace_protocol(config_path, allow_unfrozen_group=True)


def test_protocol_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    config_path = _write_protocol(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PacePreflightError, match="protocol keys mismatch"):
        load_pace_protocol(config_path, allow_unfrozen_group=True)


def test_protocol_rejects_missing_asset_key(tmp_path: Path) -> None:
    config_path = _write_protocol(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del payload["assets"]["val_csv"]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PacePreflightError, match="assets keys mismatch"):
        load_pace_protocol(config_path, allow_unfrozen_group=True)


def test_protocol_rejects_malformed_asset_digest(tmp_path: Path) -> None:
    config_path = _write_protocol(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["assets"]["val_csv_sha256"] = "ABC"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PacePreflightError, match="64 lowercase hex"):
        load_pace_protocol(config_path, allow_unfrozen_group=True)


def test_protocol_declares_all_required_sections() -> None:
    assert EXPECTED_PROTOCOL_SECTIONS == {
        "protocol_id",
        "spec",
        "assets",
        "parent_assets",
        "group_artifact",
        "parent",
        "evidence",
        "crossfit",
        "beta_solver",
        "threshold",
        "promotion",
        "execution",
        "outputs",
    }


def test_protocol_exposes_fixed_sections(tmp_path: Path) -> None:
    protocol = load_pace_protocol(
        _write_protocol(tmp_path), allow_unfrozen_group=True
    )
    assert protocol.fixed["parent"]["batch_size"] == 128


def test_formal_protocol_preregisters_core_constants(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "pace_k2_r2_parttoken.yaml"
    protocol = load_pace_protocol(source, allow_unfrozen_group=True)

    assert protocol.fixed["spec"]["sha256"] == (
        "668eca56452b3e2dc55ae1e8a9fceea626765f518b65688a8454d697e8f64fc9"
    )
    assert protocol.fixed["parent"] == {
        "peft_mode": "visual_lora_mlp_lora",
        "classifier_mode": "linear",
        "feature_adapter": "Identity",
        "batch_size": 128,
        "amp": False,
        "crop_sizes": [128, 144, 160],
        "local_scale_weights": [0.45, 0.50, 0.05],
        "flip_weight": 0.50,
        "local_weight": 0.40,
        "global_temperature": 1.5,
        "local_temperature": 1.5,
        "local_top_k": 5,
        "part_top_patches": 8,
        "part_temperature": 0.07,
        "prior": {
            "target": "uniform", "strength": 0.85, "max_iterations": 50,
            "tolerance": 1.0e-6, "damping": 0.5,
        },
    }
    assert protocol.fixed["evidence"]["tail_size"] == 7
    assert protocol.fixed["crossfit"]["outer_folds"] == 5
    assert protocol.fixed["crossfit"]["inner_folds"] == 3
    assert protocol.fixed["threshold"]["strict_comparator"] == ">"
    assert protocol.fixed["promotion"]["bootstrap_draws"] == 10000

    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["parent"]["batch_size"] = 64
    changed = tmp_path / "changed.yaml"
    changed.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(PacePreflightError, match="parent constants mismatch"):
        load_pace_protocol(changed, allow_unfrozen_group=True)


@pytest.mark.parametrize(
    ("section", "key", "changed_value"),
    [
        ("parent", "amp", True),
        ("parent", "amp", 0),
        ("evidence", "tail_size", 8),
        ("crossfit", "outer_folds", 4),
        ("beta_solver", "maximum_iterations", 99),
        ("threshold", "strict_comparator", ">="),
        ("promotion", "bootstrap_draws", 9999),
        ("execution", "batch_size", 64),
        ("parent_assets", "checkpoint_sha256", "0" * 64),
    ],
)
def test_protocol_rejects_mutated_fixed_constant(
    tmp_path: Path,
    section: str,
    key: str,
    changed_value: object,
) -> None:
    config_path = _write_protocol(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload[section][key] = changed_value
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PacePreflightError, match=f"{section} constants mismatch"):
        load_pace_protocol(config_path, allow_unfrozen_group=True)


def test_prepare_cli_double_builds_then_freezes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_protocol(tmp_path)
    calls: list[list[str]] = []

    def checked_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(subprocess, "run", checked_run)
    result = prepare_pace_group_artifact(config_path, hash_workers=2)

    frozen = load_pace_protocol(config_path)
    report = json.loads(frozen.group_artifact.report_path.read_text(encoding="utf-8"))
    assert result["first_sha256"] == result["second_sha256"]
    assert report["independent_builds_match"] is True
    assert report["git_check_ignore_exit_code"] == 1
    assert frozen.group_artifact.sha256 == result["first_sha256"]
    assert calls == [[
        "git", "check-ignore", "--quiet", "--",
        str(frozen.group_artifact.output_path),
    ]]


def test_frozen_loader_rejects_changed_group_bytes(tmp_path: Path) -> None:
    config_path = _write_protocol(tmp_path)
    protocol = load_pace_protocol(config_path, allow_unfrozen_group=True)
    audit = build_exact_group_artifact(protocol, hash_workers=1)
    freeze_group_sha_in_config(config_path, audit.map_sha256)
    protocol.group_artifact.output_path.write_text(
        "{}\n", encoding="utf-8"
    )

    with pytest.raises(
        PacePreflightError, match="group artifact SHA-256 mismatch"
    ):
        load_pace_protocol(config_path)
