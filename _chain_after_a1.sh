#!/usr/bin/env bash
# 等 A1 训练进程退出后，串行跑完第一批剩余三个点。
# 本机只有一张 GPU，不并发，避免显存竞争。

set -u

WT=/home/clairvoyant/code/worktrees/search-mixup-anchor
LOGDIR="$WT/outputs/search_20260923/_logs"

mkdir -p "$LOGDIR"

echo "=== chain waiting for A1 $(date -Is) ===" >>"$LOGDIR/_driver.log"

# 等 A1 的 python 训练进程退出
while pgrep -f "configs/search_a1_mixup_a02.yaml" >/dev/null 2>&1; do
    sleep 60
done

# 再等任何 rematch train 进程清空，确保不重叠
while pgrep -f "aegis_clip.cli.rematch train" >/dev/null 2>&1; do
    sleep 60
done

echo "=== A1 done, chaining rest $(date -Is) ===" >>"$LOGDIR/_driver.log"

exec bash "$WT/_run_search_batch.sh" \
    search_a2_mixup_a04 \
    search_b1_anchor05_lr1e5 \
    search_b2_anchor00_lr3e6
