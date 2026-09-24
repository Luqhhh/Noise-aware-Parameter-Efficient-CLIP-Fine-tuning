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
            _require(status != IMPLEMENTED,
                     "staged-resolution semantics are not implemented")
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
            accum = int(config["train"].get("grad_accum_steps", 1))
            if accum > 1 and status == IMPLEMENTED:
                raise MechanismBlockedError(
                    "effective-batch SAM with gradient accumulation is not implemented; "
                    "microbatch-SAM is forbidden by the V5 specification"
                )
            if accum == 1 and status == IMPLEMENTED:
                raise MechanismBlockedError(
                    "microbatch-SAM is forbidden for V5 S-group points; "
                    "use the effective-batch SAM implementation"
                )
    elif family == "K":
        resolution = int(search.get("resolution", 0))
        _require(resolution in {320, 384, 448}, "K resolution must be 320, 384 or 448")
        if bool(search.get("anchor_schedule", False)):
            _require(status != IMPLEMENTED, "anchor schedule is not implemented")
        teacher = str(search.get("teacher", "fixed_reference"))
        _require(teacher in {"fixed_reference", "same_view_official", "none"},
                 "K teacher must be fixed_reference, same_view_official or none")
        if teacher == "same_view_official":
            _require(status != IMPLEMENTED, "same-view official teacher is not implemented")
    elif family == "V":
        resolution = int(search.get("resolution", 0))
        _require(resolution in {320, 384}, "V resolution must be 320 or 384")
        geometry = str(search.get("geometry", ""))
        _require(
            geometry in {"rrc_area_min", "letterbox", "late_rrc_window"},
            "V geometry must be rrc_area_min, letterbox or late_rrc_window",
        )
        if geometry in {"letterbox", "late_rrc_window"}:
            _require(status != IMPLEMENTED, f"V geometry {geometry!r} is not implemented")
    elif family == "L":
        resolution = int(search.get("resolution", 0))
        _require(resolution in {320, 384}, "L resolution must be 320 or 384")
        if status == IMPLEMENTED:
            raise MechanismBlockedError("L attention-local V5 semantics are not implemented")
    elif family == "Q":
        resolution = int(search.get("resolution", 0))
        _require(resolution in {320, 384}, "Q resolution must be 320 or 384")
        mode = str(config.get("train", {}).get("sam", {}).get("mode", ""))
        _require(mode == "gsam_constant_rho", "Q config must declare train.sam.mode=gsam_constant_rho")
        if status == IMPLEMENTED:
            raise MechanismBlockedError("GSAM is not implemented")
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
        if status == IMPLEMENTED:
            raise MechanismBlockedError(
                "H implementation is gated on the F05 320px feature cache and binding"
            )
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
