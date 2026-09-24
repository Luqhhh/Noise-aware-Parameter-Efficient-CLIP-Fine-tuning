"""Fail-closed semantic declaration layer for REMATCH750_SEARCH_V5.

The V5 manifest is a *conditional candidate list*, not a promise that every
operator is implemented.  This module makes that distinction executable:

* :func:`validate_declaration` is cheap and asset-free.  It validates the
  declared V5 semantics and raises :class:`MechanismBlockedError` for a trial
  whose runtime implementation has not landed.
* :func:`validate_training` first runs the declaration check, then delegates
  dataset/cache/checkpoint provenance to the shared V4/V5 lineage validator.

Nothing here silently treats an unknown search field as harmless.
"""
from __future__ import annotations

from typing import Any

from aegis_clip import rematch_search_v4 as _lineage


PROTOCOL = "rematch750_search_v5"

IMPLEMENTED = "implemented"
BLOCKED_IMPLEMENTATION = "blocked_implementation"
PENDING_ARTIFACT = "pending_artifact"
CONDITIONAL_PENDING_EVIDENCE = "conditional_pending_evidence"
BLOCKED_DEPENDENCY = "blocked_dependency"

_IMPLEMENTATION_STATUSES = {
    IMPLEMENTED,
    BLOCKED_IMPLEMENTATION,
    PENDING_ARTIFACT,
    CONDITIONAL_PENDING_EVIDENCE,
    BLOCKED_DEPENDENCY,
}

_ALLOWED_FAMILIES = {"R", "S", "K", "V", "L", "Q", "N", "H", "X"}

_COMMON_SEARCH_KEYS = {
    "micro_batch_candidates",
    "smoke_required",
    "reference_resolution",
    "notes",
}

_FAMILY_SEARCH_KEYS = {
    "R": {"resolution", "staged_resolution"},
    "S": {"resolution", "precision", "optimizer", "sam_rho"},
    "K": {"resolution", "anchor_weight", "anchor_schedule", "teacher"},
    "V": {"resolution", "geometry", "rrc_scale_min", "rrc_scale_max",
          "late_rrc_scale_min", "late_rrc_scale_max"},
    "L": {"resolution", "area_fraction", "local_supervision_weight",
          "local_start_epoch", "confidence_gate"},
    "Q": {"resolution", "gsam_alpha", "sam_rho"},
    "N": {"resolution", "repair_rho", "confidence_gate",
          "max_relabel_fraction", "evidence_required"},
    "H": {"feature_source", "classifier_mode", "sampler_mode", "init"},
    "X": {"component_trials"},
}


class MechanismBlockedError(RuntimeError):
    """Raised when a declared V5 trial is not implemented or not ready."""


def is_v5(config: dict[str, Any]) -> bool:
    return str(config.get("project", {}).get("protocol", "")) == PROTOCOL


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"REMATCH750_SEARCH_V5 declaration: {message}")


def _search(config: dict[str, Any]) -> dict[str, Any]:
    search = config.get("project", {}).get("search", {})
    _require(isinstance(search, dict), "project.search must be a mapping")
    return search


def _family(config: dict[str, Any]) -> str:
    family = str(config.get("project", {}).get("family", ""))
    _require(family in _ALLOWED_FAMILIES, f"unknown family {family!r}")
    return family


def validate_declaration(config: dict[str, Any], *, require_implemented: bool = False) -> dict[str, Any]:
    """Validate V5-only semantics and return the declaration.

    Unknown search keys fail closed.  A blocked status is a valid declaration
    but cannot be executed unless ``require_implemented`` is false.
    """
    _require(is_v5(config), "project.protocol is not rematch750_search_v5")
    project = config["project"]
    trial_id = str(project.get("trial_id", ""))
    _require(bool(trial_id), "project.trial_id is required")
    family = _family(config)
    search = _search(config)
    allowed_keys = _COMMON_SEARCH_KEYS | _FAMILY_SEARCH_KEYS[family]
    unknown = set(search) - allowed_keys
    _require(not unknown, f"unknown project.search keys for {trial_id}: {sorted(unknown)}")

    status = str(project.get("implementation_status", ""))
    _require(
        status in _IMPLEMENTATION_STATUSES,
        f"project.implementation_status must be one of {sorted(_IMPLEMENTATION_STATUSES)}",
    )

    if family == "R":
        resolution = int(search.get("resolution", config["model"].get("input_resolution", 0)))
        _require(resolution in _lineage.V5_ALLOWED_RESOLUTIONS,
                 f"R resolution must be one of {_lineage.V5_ALLOWED_RESOLUTIONS}")
        if bool(search.get("staged_resolution", False)):
            staged = config.get("train", {}).get("staged_resolution", {})
            _require(isinstance(staged, dict) and bool(staged.get("enabled", False)),
                     "R staged point requires train.staged_resolution.enabled=true")
            early = int(staged.get("early_resolution", 0))
            late = int(staged.get("late_resolution", 0))
            switch = int(staged.get("switch_epoch", 0))
            _require(early in _lineage.V5_ALLOWED_RESOLUTIONS, "bad staged early_resolution")
            _require(late in _lineage.V5_ALLOWED_RESOLUTIONS, "bad staged late_resolution")
            _require(0 < switch <= int(config["train"].get("epochs", 16)),
                     "staged switch_epoch must be inside the training horizon")
            _require(int(config["model"].get("input_resolution", -1)) == early,
                     "R staged config model.input_resolution must equal early_resolution")
        else:
            _require(int(config["model"].get("input_resolution", -1)) == resolution,
                     "R config input_resolution must equal project.search.resolution")
    elif family == "S":
        resolution = int(search.get("resolution", 0))
        _require(resolution in {320, 384, 448}, "S resolution must be 320, 384 or 448")
        _require(str(search.get("precision", "")) == "fp32", "S group must declare precision=fp32")
        optimizer = str(search.get("optimizer", ""))
        _require(optimizer in {"adamw", "sam"}, "S optimizer must be adamw or sam")
        _require(not bool(config.get("train", {}).get("amp", True)),
                 "S group requires train.amp=false")
        if optimizer == "sam":
            sam = config.get("train", {}).get("sam", {})
            _require(
                isinstance(sam, dict) and bool(sam.get("enabled", False)),
                "S SAM point requires train.sam.enabled=true",
            )
            _require(
                sam.get("mode", "standard_global_l2") == "standard_global_l2",
                "S SAM point requires train.sam.mode=standard_global_l2",
            )
            _require(
                int(config["train"].get("grad_accum_steps", 1)) > 1,
                "V5 S SAM points must use the effective-batch accumulation path",
            )
            if status != IMPLEMENTED:
                raise MechanismBlockedError("S SAM point is not implemented")
    elif family == "K":
        resolution = int(search.get("resolution", 0))
        _require(resolution in {320, 384, 448}, "K resolution must be 320, 384 or 448")
        if bool(search.get("anchor_schedule", False)):
            schedule = config.get("loss", {}).get("anchor_schedule", {})
            _require(isinstance(schedule, dict) and schedule, "anchor schedule config missing")
            _require(float(schedule.get("start_weight", -1.0)) >= 0.0,
                     "anchor schedule start_weight must be non-negative")
            _require(float(schedule.get("end_weight", -1.0)) >= 0.0,
                     "anchor schedule end_weight must be non-negative")
            hold = int(schedule.get("hold_epochs", 0))
            end = int(schedule.get("end_epoch", 0))
            _require(end > hold >= 0, "anchor schedule requires end_epoch > hold_epochs >= 0")
            if status == BLOCKED_IMPLEMENTATION:
                raise MechanismBlockedError("anchor schedule declaration is marked blocked")
        teacher = str(search.get("teacher", "fixed_reference"))
        _require(teacher in {"fixed_reference", "same_view_official", "none"},
                 "K teacher must be fixed_reference, same_view_official or none")
        if teacher == "same_view_official":
            teacher_config = config.get("train", {}).get("same_view_teacher", {})
            _require(
                isinstance(teacher_config, dict)
                and bool(teacher_config.get("enabled", False))
                and str(teacher_config.get("source", "")) == "official_clip"
                and int(teacher_config.get("resolution", 0)) == 224,
                "same-view official teacher requires train.same_view_teacher "
                "{enabled: true, source: official_clip, resolution: 224}",
            )
    elif family == "V":
        resolution = int(search.get("resolution", 0))
        _require(resolution in {320, 384}, "V resolution must be 320 or 384")
        geometry = str(search.get("geometry", ""))
        _require(
            geometry in {"rrc_area_min", "letterbox", "late_rrc_window"},
            "V geometry must be rrc_area_min, letterbox or late_rrc_window",
        )
        if geometry == "letterbox":
            _require(
                config.get("data", {}).get("train_augmentation") == "clip_letterbox",
                "V letterbox requires data.train_augmentation=clip_letterbox",
            )
        if geometry == "late_rrc_window":
            late_rrc = config.get("data", {}).get("late_rrc", {})
            _require(
                isinstance(late_rrc, dict) and bool(late_rrc.get("enabled", False)),
                "V late_rrc requires data.late_rrc.enabled=true",
            )
            _require(
                0.0 < float(late_rrc.get("scale_min", 0.0))
                <= float(late_rrc.get("scale_max", 0.0)) <= 1.0,
                "V late_rrc scale bounds are invalid",
            )
            _require(
                int(late_rrc.get("start_epoch", 0)) > 1,
                "V late_rrc start_epoch must be positive",
            )
    elif family == "L":
        resolution = int(search.get("resolution", 0))
        _require(resolution in {320, 384}, "L resolution must be 320 or 384")
        local_config = config.get("loss", {}).get("attention_local_training", {})
        _require(
            isinstance(local_config, dict) and bool(local_config.get("enabled", False)),
            "L requires loss.attention_local_training.enabled=true",
        )
        _require(
            int(local_config.get("start_epoch", 0)) >= 1,
            "L attention local start_epoch must be positive",
        )
        _require(
            0.0 <= float(local_config.get("confidence_gate", -1.0)) <= 1.0,
            "L confidence_gate must be in [0,1]",
        )
        crop_size = int(local_config.get("crop_size", 0))
        _require(0 < crop_size < int(config["model"]["input_resolution"]),
                 "L crop_size must be smaller than input resolution")
    elif family == "Q":
        resolution = int(search.get("resolution", 0))
        _require(resolution in {320, 384}, "Q resolution must be 320 or 384")
        sam = config.get("train", {}).get("sam", {})
        mode = str(sam.get("mode", ""))
        _require(mode == "gsam_constant_rho", "Q config must declare train.sam.mode=gsam_constant_rho")
        _require(bool(sam.get("enabled", False)), "Q config must enable train.sam")
        alpha = float(sam.get("gsam_alpha", -1.0))
        _require(0.0 <= alpha <= 1.0, "Q gsam_alpha must be in [0,1]")
        _require(
            int(config["train"].get("grad_accum_steps", 1)) > 1,
            "Q GSAM requires effective-batch accumulation",
        )
        _require(not bool(config["train"].get("amp", True)), "Q GSAM requires FP32")
    elif family == "N":
        _require(status in {CONDITIONAL_PENDING_EVIDENCE, BLOCKED_IMPLEMENTATION},
                 "N group is conditional and cannot be marked implemented without new evidence")
    elif family == "H":
        _require(str(search.get("feature_source", "")) == "F05_encoder",
                 "H feature_source must be F05_encoder")
        _require(int(search.get("reference_resolution", 0)) == 320,
                 "H reference_resolution must be 320 for the F05 encoder")
        classifier = str(search.get("classifier_mode", config["model"].get("classifier_mode", "")))
        _require(classifier in {"linear", "cosine"}, "H classifier_mode must be linear or cosine")
        sampler = str(search.get("sampler_mode", ""))
        _require(sampler in {"natural", "sqrt_class_balanced", "class_balanced"},
                 "H sampler_mode must be natural, sqrt_class_balanced or class_balanced")
        init = str(search.get("init", ""))
        _require(init in {"inherit", "reinit"}, "H init must be inherit or reinit")
        # Executable H rows still pass through the shared cache/checkpoint
        # lineage validator, which verifies the F05 320px feature manifest.
    elif family == "X":
        components = search.get("component_trials", [])
        _require(isinstance(components, list) and len(components) >= 2,
                 "X component_trials must be a list of at least two trial ids")
        if status == IMPLEMENTED:
            raise MechanismBlockedError("X combinations are blocked until component winners exist")

    if require_implemented and status != IMPLEMENTED:
        raise MechanismBlockedError(
            f"{trial_id} is declared {status}; it cannot be executed as an implemented trial"
        )
    return {"trial_id": trial_id, "family": family, "status": status, "search": search}


def validate_training(
    config: dict[str, Any],
    resume: str | None = None,
    init_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Validate declaration, then delegate provenance lineage to V4/V5 core."""
    validate_declaration(config, require_implemented=True)
    # Shared dataset/cache/checkpoint provenance implementation.  The core
    # module recognizes both protocol strings through ``is_rematch_search``.
    return _lineage.validate_training(config, resume=resume, init_checkpoint=init_checkpoint)
