#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment missing. Run: uv sync --extra dev --locked" >&2
  exit 2
fi

stage="${1:-}"
case "${stage}" in
  test)
    "${PYTHON}" -m pytest -q -p no:cacheprovider
    ;;
  audit)
    "${PYTHON}" -m aegis_clip.cli.audit --config configs/a0_fulldata_anchor.yaml
    ;;
  prepare)
    "${PYTHON}" -m aegis_clip.cli.prepare_stage \
      --train-root /home/x28639/projects/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning/train \
      --output-root artifacts/stages \
      --stage preliminary \
      --seed 42 \
      --val-ratio 0.10 \
      --expected-classes 500 \
      --expected-samples 103218 \
      --hash-workers "${HASH_WORKERS:-8}"
    ;;
  cache)
    "${PYTHON}" -m aegis_clip.cli.cache_features \
      --config configs/a1_cvt_soft.yaml \
      --device "${DEVICE}" \
      --batch-size "${CACHE_BATCH_SIZE:-256}" \
      --workers "${CACHE_WORKERS:-8}"
    ;;
  groups)
    "${PYTHON}" -m aegis_clip.cli.build_groups \
      --config configs/a1_cvt_soft.yaml \
      --workers "${HASH_WORKERS:-8}"
    ;;
  trust)
    "${PYTHON}" -m aegis_clip.cli.build_trust \
      --config configs/a1_cvt_soft.yaml \
      --device "${DEVICE}"
    ;;
  smoke|a0|a1|a2|a3|b0|b1|b2|c0|c1|c2|c3|d0|d1|f1|f6|f7)
    config="$(
      case "${stage}" in
        smoke) echo configs/s0_cvt_cached_smoke.yaml ;;
        a0) echo configs/a0_fulldata_anchor.yaml ;;
        a1) echo configs/a1_cvt_soft.yaml ;;
        a2) echo configs/a2_cvt_cap.yaml ;;
        a3) echo configs/a3_cvt_ln_distill.yaml ;;
        b0) echo configs/b0_cached_nomix.yaml ;;
        b1) echo configs/b1_cached_cvt_soft.yaml ;;
        b2) echo configs/b2_cached_cvt_cap.yaml ;;
        c0) echo configs/c0_feature_adapter.yaml ;;
        c1) echo configs/c1_gated_feature_adapter.yaml ;;
        c2) echo configs/c2_gated_trust_weight.yaml ;;
        c3) echo configs/c3_lnpost_distill.yaml ;;
        d0) echo configs/d0_full_train_anchor.yaml ;;
        d1) echo configs/d1_full_train_gated_adapter.yaml ;;
        f1) echo configs/f1_visual_lora_clean_core.yaml ;;
        f6) echo configs/f6_a2_disjoint_lora_gate.yaml ;;
        f7) echo configs/f7_a2_fixed_fullfit.yaml ;;
      esac
    )"
    train_args=(--config "${config}")
    if [[ "${OVERWRITE:-0}" == "1" ]]; then
      train_args+=(--overwrite)
    fi
    if [[ -n "${RESUME:-}" ]]; then
      train_args+=(--resume "${RESUME}")
    fi
    "${PYTHON}" -m aegis_clip.cli.train "${train_args[@]}"
    ;;
  *)
    echo "Usage: $0 {test|prepare|cache|audit|groups|trust|smoke|a0|a1|a2|a3|b0|b1|b2|c0|c1|c2|c3|d0|d1|f1|f6|f7}" >&2
    exit 2
    ;;
esac
