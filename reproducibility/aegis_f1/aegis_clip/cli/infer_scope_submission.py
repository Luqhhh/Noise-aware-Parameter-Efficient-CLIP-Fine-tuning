"""Replay a promoted frozen SCOPE deployment into submission artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from aegis_clip.data import load_class_mapping
from aegis_clip.runtime import sha256_file
from aegis_clip.scope_cache import (
    load_scope_cache,
    semantic_sha256,
    validate_evidence_cache,
    validate_parent_cache,
)
from aegis_clip.scope_protocol import load_scope_protocol, verify_scope_assets
from aegis_clip.submission import create_submission


def require_promoted_decision(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "scope_decision_v1":
        raise ValueError("SCOPE decision schema mismatch")
    if payload.get("promoted") is not True:
        raise ValueError("SCOPE-K2 was not promoted; test inference is forbidden")
    return payload


def replay_scope_decisions(
    candidates: torch.Tensor,
    margin: torch.Tensor,
    evidence: torch.Tensor,
    eligible: torch.Tensor,
    *,
    beta: float,
    threshold: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = torch.as_tensor(candidates, dtype=torch.int64).cpu()
    margin = torch.as_tensor(margin, dtype=torch.float64).flatten().cpu()
    evidence = torch.as_tensor(evidence, dtype=torch.float64).flatten().cpu()
    eligible = torch.as_tensor(eligible, dtype=torch.bool).flatten().cpu()
    n = margin.numel()
    if candidates.shape != (n, 2) or evidence.shape != (n,) or eligible.shape != (n,):
        raise ValueError("deployment decision inputs are misaligned")
    if float(beta) < 0.0 or not torch.isfinite(margin).all() or not torch.isfinite(evidence).all():
        raise ValueError("deployment scorer is invalid")
    score = margin + float(beta) * evidence
    mode = threshold.get("mode")
    if mode == "all_switch":
        switch = eligible
    elif mode == "finite":
        gamma = threshold.get("gamma")
        if gamma is None:
            raise ValueError("finite deployment threshold is missing gamma")
        switch = eligible & (score > float(gamma))
    elif mode == "no_switch":
        switch = torch.zeros(n, dtype=torch.bool)
    else:
        raise ValueError("deployment threshold mode is invalid")
    prediction = candidates[:, 0].clone()
    prediction[switch] = candidates[switch, 1]
    return prediction, switch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-cache", type=Path, required=True)
    parser.add_argument("--evidence-cache", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # This check intentionally precedes every test-cache read or output mkdir.
    decision = require_promoted_decision(args.decision)
    protocol = load_scope_protocol(args.config)
    verify_scope_assets(protocol)
    checkpoint = args.checkpoint.resolve()
    if checkpoint != protocol.assets.checkpoint or sha256_file(checkpoint) != protocol.assets.checkpoint_sha256:
        raise ValueError("deployment checkpoint is not the frozen parent")
    deployment = json.loads(args.deployment.read_text(encoding="utf-8"))
    if deployment.get("schema") != "scope_deployment_v1" or deployment.get("method") != "scope":
        raise ValueError("SCOPE deployment payload is invalid")
    if decision.get("deployment") != str(args.deployment.resolve()):
        raise ValueError("decision/deployment path binding mismatch")
    parent = load_scope_cache(args.parent_cache)
    evidence = load_scope_cache(args.evidence_cache)
    validate_parent_cache(parent, protocol, "test")
    validate_evidence_cache(evidence, parent, protocol, "test")
    binding_checks = {
        "checkpoint": deployment["lineage"]["checkpoint_sha256"] == protocol.assets.checkpoint_sha256,
        "protocol": deployment["protocol_sha256"] == sha256_file(protocol.config_path),
        "view_order": deployment["view_order"] == evidence["view_order"],
        "view_weights": deployment["view_weights"] == evidence["view_weights"],
        "edges": deployment["edge_sha256"] == evidence["edges_sha256"],
        "test_parent_evidence": evidence["parent_semantic_sha256"] == semantic_sha256(parent),
    }
    if not all(binding_checks.values()):
        raise ValueError(f"deployment binding mismatch: {binding_checks}")
    prediction, switch = replay_scope_decisions(
        parent["candidate_indices"], parent["parent_margin"], evidence["scope"]["aggregate"],
        evidence["scope"]["eligibility"], beta=float(deployment["beta"]),
        threshold=deployment["mapped_threshold"],
    )
    replay, replay_switch = replay_scope_decisions(
        parent["candidate_indices"], parent["parent_margin"], evidence["scope"]["aggregate"],
        evidence["scope"]["eligibility"], beta=float(deployment["beta"]),
        threshold=deployment["mapped_threshold"],
    )
    if not torch.equal(prediction, replay) or not torch.equal(switch, replay_switch):
        raise ValueError("deployment decision replay is not deterministic")
    _, idx_to_class = load_class_mapping(protocol.assets.class_to_idx)
    labels = [str(idx_to_class[int(value)]).zfill(4) for value in prediction.tolist()]
    names = [Path(path).name for path in parent["paths"]]
    predictions = list(zip(names, labels))
    manifest = create_submission(
        predictions, names, args.output_dir, checkpoint,
        inference_mode="scope_k2_frozen_parent_tta", tta_risk_acknowledged=True,
        valid_labels={str(value).zfill(4) for value in idx_to_class.values()},
        extra_manifest={
            "scope_decision_sha256": sha256_file(args.decision),
            "scope_deployment_sha256": sha256_file(args.deployment),
            "scope_test_parent_cache_sha256": parent["_cache_sha256"],
            "scope_test_evidence_cache_sha256": evidence["_cache_sha256"],
            "scope_switch_count": int(switch.sum()),
            "scope_replay_equal": True,
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
