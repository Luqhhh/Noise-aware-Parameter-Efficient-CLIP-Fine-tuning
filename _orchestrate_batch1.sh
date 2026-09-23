#!/usr/bin/env bash
# 第一批编排：训练与推理交替，避免推理和训练抢同一张卡。
# 顺序：等 A1 训完 → 出 A1 包 → 训 A2 → 出 A2 包 → 训 B1 → 出 B1 包 → 训 B2
#
# 出包不是为了立刻上传，是为了在今晚 22:00 决策时手上有全部选项。
# 提交额度只有一个，谁上由那时的本地成绩决定。

set -u

WT=/home/clairvoyant/code/worktrees/search-mixup-anchor
MAIN=/home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
LOGDIR="$WT/outputs/search_20260923/_logs"
PY="$MAIN/.venv/bin/python"

export PYTHONPATH="$WT/reproducibility/aegis_f1"
cd "$WT" || exit 1
mkdir -p "$LOGDIR"

log() { echo "=== $* $(date -Is) ===" | tee -a "$LOGDIR/_driver.log"; }

wait_idle() {
    if [ -n "${1:-}" ]; then
        while pgrep -f "configs/$1.yaml" >/dev/null 2>&1; do sleep 30; done
    fi
    while pgrep -f "aegis_clip.cli.rematch train" >/dev/null 2>&1; do sleep 30; done
}

train_one() {
    local name="$1" exp="$2"
    log "TRAIN START $name"
    "$PY" -u -m aegis_clip.cli.rematch train --config "configs/$name.yaml" \
        >"$LOGDIR/$name.log" 2>&1
    log "TRAIN END $name rc=$?"
    local sr="$WT/outputs/search_20260923/$exp/seed42/checkpoints/selected_report.json"
    if [ -f "$sr" ]; then
        echo "selected_report $exp: $(cat "$sr")" >>"$LOGDIR/_driver.log"
    fi
}

infer_one() {
    local name="$1" exp="$2"
    log "INFER START $name"
    "$PY" -u -m aegis_clip.cli.rematch infer --config "configs/$name.yaml" \
        >"$LOGDIR/infer_$name.log" 2>&1
    log "INFER END $name rc=$?"
    local sub="$WT/outputs/search_20260923/$exp/seed42/submission"
    if [ -d "$sub" ]; then
        echo "PACKAGE $exp: $(ls "$sub")" >>"$LOGDIR/_driver.log"
    else
        echo "PACKAGE MISSING $exp" >>"$LOGDIR/_driver.log"
    fi
}

# --- 1. 等 A1 训完，出 A1 的包 ---
wait_idle search_a1_mixup_a02
infer_one search_a1_mixup_a02 SEARCH_A1_MIXUP_A02

# --- 2. A2：训练 + 出包 ---
train_one search_a2_mixup_a04 SEARCH_A2_MIXUP_A04
infer_one search_a2_mixup_a04 SEARCH_A2_MIXUP_A04

# --- 3. B1：训练 + 出包（方案提名的第一对之一）---
train_one search_b1_anchor05_lr1e5 SEARCH_B1_ANCHOR05_LR1E5
infer_one search_b1_anchor05_lr1e5 SEARCH_B1_ANCHOR05_LR1E5

# --- 4. B2：只训练（约 21:50 结束，赶不上今晚决策）---
train_one search_b2_anchor00_lr3e6 SEARCH_B2_ANCHOR00_LR3E6

log "BATCH1 DONE"
