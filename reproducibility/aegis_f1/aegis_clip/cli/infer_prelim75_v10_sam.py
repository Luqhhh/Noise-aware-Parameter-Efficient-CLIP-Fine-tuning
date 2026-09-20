"""Launch generic inference under the archived L1 CUDA numerical context."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


SETTINGS = {
    "batch_size": 64,
    "matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "float32_matmul_precision": "highest",
}


def _argument_value(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Missing required PRELIM75 v10 inference argument: {name}") from exc


def main() -> None:
    if int(_argument_value("--batch-size")) != SETTINGS["batch_size"]:
        raise ValueError("PRELIM75 v10 inference batch size changed")
    torch.backends.cuda.matmul.allow_tf32 = SETTINGS["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = SETTINGS["cudnn_allow_tf32"]
    torch.set_float32_matmul_precision(SETTINGS["float32_matmul_precision"])
    from aegis_clip.cli.infer import main as generic_main

    generic_main()
    output = Path(_argument_value("--output-dir"))
    actual = {
        "batch_size": SETTINGS["batch_size"],
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if actual != SETTINGS:
        raise RuntimeError("Generic inference changed PRELIM75 v10 numerical settings")
    (output / "prelim75_v10_inference_numerics.json").write_text(
        json.dumps({"status": "passed", "settings": actual}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
