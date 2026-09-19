#!/usr/bin/env python3
"""Run the frozen v8 snapshot after auditing one volatile repr-only mismatch.

The original v8 producer included the process-local address from
``repr(preprocess)`` in its protocol hash.  This compatibility entry point does
not alter that producer record.  It requires the current consumer descriptor
to be identical after removing only Python function addresses, then returns
the original producer descriptor to the existing version-2 validator.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "outputs/prelim75_v8_20260919/execution_source"
if not SNAPSHOT.is_dir():
    raise FileNotFoundError(f"Missing immutable v8 execution snapshot: {SNAPSHOT}")
sys.path.insert(0, str(SNAPSHOT))

from aegis_clip.calibration_binding import protocol_sha256  # noqa: E402
from aegis_clip.runtime import atomic_json_dump, sha256_file  # noqa: E402
import aegis_clip.cli.infer as infer_cli  # noqa: E402


_FUNCTION_ADDRESS = re.compile(
    r"(<function\s+[^>]+?)\s+at\s+0x[0-9a-fA-F]+(>)"
)


def _normalized(descriptor: dict) -> dict:
    result = copy.deepcopy(descriptor)
    preprocess = result.get("preprocess")
    if not isinstance(preprocess, str):
        raise ValueError("Protocol repair requires a string preprocess descriptor")
    result["preprocess"] = _FUNCTION_ADDRESS.sub(r"\1\2", preprocess)
    return result


def _resolve(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


_original_builder = infer_cli.build_inference_protocol_descriptor


def _bound_builder(args, checkpoint, config, preprocess, device):
    actual = _original_builder(args, checkpoint, config, preprocess, device)
    if not args.prior_config:
        return actual
    prior_path = Path(args.prior_config).resolve()
    record = json.loads(prior_path.read_text(encoding="utf-8"))
    source_binding = record.get("source_audit", {})
    audit_path = _resolve(source_binding.get("path", ""), prior_path.parent)
    if not audit_path.is_file() or sha256_file(audit_path) != source_binding.get(
        "sha256"
    ):
        raise ValueError("Protocol repair source audit is missing or changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    logits_binding = audit.get("validation_logits", {})
    logits_path = _resolve(logits_binding.get("path", ""), audit_path.parent)
    if not logits_path.is_file() or sha256_file(logits_path) != logits_binding.get(
        "sha256"
    ):
        raise ValueError("Protocol repair frozen score cache is missing or changed")
    payload = torch.load(logits_path, map_location="cpu", weights_only=True)
    producer = payload.get("inference_protocol")
    producer_hash = payload.get("inference_protocol_sha256")
    if not isinstance(producer, dict):
        raise ValueError("Frozen score cache lacks its producer protocol descriptor")
    if (
        protocol_sha256(producer) != producer_hash
        or producer_hash != record.get("inference_protocol_sha256")
        or producer_hash != audit.get("inference_protocol_sha256")
    ):
        raise ValueError("Producer protocol bindings changed before repair")
    normalized_actual = _normalized(actual)
    normalized_producer = _normalized(producer)
    if normalized_actual != normalized_producer:
        differing = sorted(
            key
            for key in set(normalized_actual) | set(normalized_producer)
            if normalized_actual.get(key) != normalized_producer.get(key)
        )
        raise ValueError(
            "Protocol mismatch is not limited to a volatile function address: "
            + ",".join(differing)
        )
    raw_differences = sorted(
        key
        for key in set(actual) | set(producer)
        if actual.get(key) != producer.get(key)
    )
    if raw_differences != ["preprocess"]:
        raise ValueError(
            "Protocol repair expected exactly one raw preprocess difference, got "
            + ",".join(raw_differences)
        )
    receipt_path = os.environ.get("PRELIM75_V8_PROTOCOL_REPAIR_RECEIPT")
    if not receipt_path:
        raise ValueError("Protocol repair requires an explicit receipt output path")
    atomic_json_dump(
        {
            "status": "verified_repr_address_only",
            "producer_protocol_sha256": producer_hash,
            "consumer_raw_protocol_sha256": protocol_sha256(actual),
            "normalized_protocol_sha256": protocol_sha256(normalized_actual),
            "raw_differing_fields": raw_differences,
            "normalization": "remove_only_process_local_hex_address_from_function_repr",
            "validation_logits": str(logits_path),
            "validation_logits_sha256": logits_binding["sha256"],
            "producer_preprocess": producer["preprocess"],
            "consumer_preprocess": actual["preprocess"],
            "all_other_protocol_fields_exact": True,
            "snapshot_package_unchanged": True,
        },
        receipt_path,
    )
    return producer


def main() -> None:
    infer_cli.build_inference_protocol_descriptor = _bound_builder
    infer_cli.main()


if __name__ == "__main__":
    main()
