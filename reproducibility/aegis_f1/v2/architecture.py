"""CPU-only identity of the legacy projected and explicit 768-dim V2 routes."""
from __future__ import annotations


def architecture_spec(recipe):
    dimension = recipe.get("feature_dim", 512)
    if type(dimension) is not int or dimension not in (512, 768):
        raise ValueError("V2 feature_dim must be 512 or 768")
    return dict(feature_dim=dimension,
                feature_source="preprojection_cls" if dimension == 768 else "projected_cls",
                readout_precision="float32" if dimension == 768 else "autocast")


def check_plan_architecture(plan, config=None):
    expected = architecture_spec(plan["recipe"])
    if "feature_dim" not in plan["recipe"]:
        if "architecture" in plan:
            raise ValueError("Legacy V2 recipe cannot carry an unbound architecture override")
        return
    if plan.get("architecture") != expected:
        raise ValueError("Prepared V2 architecture differs from recipe")
    if config is not None and config["model"].get("architecture") != expected:
        raise ValueError("Stage V2 architecture differs from recipe")
