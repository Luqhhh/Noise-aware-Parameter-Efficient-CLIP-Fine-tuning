#!/usr/bin/env bash
# 第一批编排：训练与推理交替，避免推理和训练抢同一张卡。
# 顺序：等 A1 训完 → 出 A1 包 → 训 A2 → 出 A2 包 → 训 B1 → 训 B2
# 出包这一步是为了用掉今天的提交额度，并把 α=0.2 / α=0.4 两个强度都送上平台。

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
    # 等指定配置的训练进程退出，再等全场没有任何 rematch train 在跑
    if [ -n "${1:-}" ]; then
        while pgrep -f "configs/$1.yaml" >/dev/null 2>&1; do sleep 30; done
    fi
    while pgrep -f "aegis_clip.cli.rematch train" >/dev/null 2>&1; do sleep 30; done
}

train_one() {
    local name="$1"
    log "TRAIN START $name"
    "$PY" -u -m aegis_clip.cli.rematch train --config "configs/$name.yaml" \
        >"$LOGDIR/$name.log" 2>&1
    log "TRAIN END $name rc=$?"
    # 记录被选中的轮次，供人工核对（last_epoch 下应等于总轮数）
    local sr="$WT/outputs/search_20260923/$2/seed42/checkpoints/selected_report.json"
    if [ -f "$sr" ]; then
        echo "selected_report: $(cat "$sr")" >>"$LOGDIR/_driver.log"
    fi
}

infer_one() {
    local name="$1"
    log "INFER START $name"
    "$PY" -u -m aegis_clip.cli.rematch infer --config "configs/$name.yaml" \
        >"$LOGDIR/infer_$name.log" 2>&1
    log "INFER END $name rc=$?"
    local sub="$WT/outputs/search_20260923/$2/seed42/submission"
    if [ -d "$sub" ]; then
        echo "PACKAGE: $(ls -la "$sub" | tail -n +2)" >>"$LOGDIR/_driver.log"
    else
        echo "PACKAGE MISSING for $name" >>"$LOGDIR/_driver.log"
    fi
}

# --- 1. 等 A1 训完，立刻出 A1 的包 ---
wait_idle search_a1_mixup_a02
infer_one search_a1_mixup_a02 SEARCH_A1_MIXUP_A02

# --- 2. A2：训练 + 出包 ---
train_one search_a2_mixup_a04 SEARCH_A2_MIXUP_A04
infer_one search_a2_mixup_a04 SEARCH_A2_MIXUP_A04

# --- 3. B 轴两点只训练，本轮不出包 ---
train_one search_b1_anchor05_lr1e5 SEARCH_B1_ANCHOR05_LR1E5
train_one search_b2_anchor00_lr3e6 SEARCH_B2_ANCHOR00_LR3E6

log "BATCH1 DONE"
