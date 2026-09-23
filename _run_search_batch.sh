#!/usr/bin/env bash
# 搜索批次串行驱动（本机只有一张 GPU，按 GPT-6 方案给定顺序串行）
# 用法: _run_search_batch.sh <config 基名> [<config 基名> ...]
# 每个点各写一份日志；任一点非零退出即记录并继续下一点，不静默跳过。

set -u

WT=/home/clairvoyant/code/worktrees/search-mixup-anchor
MAIN=/home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
LOGDIR="$WT/outputs/search_20260923/_logs"
PY="$MAIN/.venv/bin/python"

export PYTHONPATH="$WT/reproducibility/aegis_f1"
cd "$WT" || exit 1
mkdir -p "$LOGDIR"

for name in "$@"; do
    cfg="configs/$name.yaml"
    log="$LOGDIR/$name.log"
    if [ ! -f "$cfg" ]; then
        echo "MISSING $cfg" | tee -a "$LOGDIR/_driver.log"
        continue
    fi
    echo "=== START $name $(date -Is) ===" | tee -a "$LOGDIR/_driver.log"
    start=$(date +%s)
    "$PY" -u -m aegis_clip.cli.rematch train --config "$cfg" >"$log" 2>&1
    rc=$?
    end=$(date +%s)
    echo "=== END $name rc=$rc elapsed=$((end - start))s $(date -Is) ===" | tee -a "$LOGDIR/_driver.log"
done
