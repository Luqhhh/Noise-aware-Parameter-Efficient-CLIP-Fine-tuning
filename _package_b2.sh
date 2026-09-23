#!/usr/bin/env bash
# 等 B2 训练结束后为它出包。
#
# 为什么需要这个独立脚本：_orchestrate_batch1.sh 的最后一步只训练不出包，
# 依据是我当时（按 0.169 s/step）错误估计 B2 约 21:50 才结束、赶不上 22:00。
# 实测速率是 0.117~0.125 s/step，B2 约 19:30 结束 —— 能赶上，应该出包。
# 编排脚本此刻正在运行（bash 增量读取，改它会让正在执行的脚本错乱），
# 所以另起一个等待进程补这一步，不去动正在跑的编排。
#
# 不重启任何训练；只在 B2 的训练进程退出后追加一次 infer。

set -u

WT=/home/clairvoyant/code/worktrees/search-mixup-anchor
MAIN=/home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
LOGDIR="$WT/outputs/search_20260923/_logs"
PY="$MAIN/.venv/bin/python"
CFG=search_b2_anchor00_lr3e6
EXP=SEARCH_B2_ANCHOR00_LR3E6

export PYTHONPATH="$WT/reproducibility/aegis_f1"
cd "$WT" || exit 1

echo "=== B2 PACKAGER waiting $(date -Is) ===" >>"$LOGDIR/_driver.log"

# 等 B2 的训练进程退出（自己的 pgrep 不会匹配到本脚本）
while pgrep -f "rematch train --config configs/$CFG.yaml" >/dev/null 2>&1; do
    sleep 30
done

echo "=== B2 PACKAGER train gone, running infer $(date -Is) ===" >>"$LOGDIR/_driver.log"

"$PY" -u -m aegis_clip.cli.rematch infer --config "configs/$CFG.yaml" \
    >"$LOGDIR/infer_$CFG.log" 2>&1
rc=$?

echo "=== B2 PACKAGER INFER END rc=$rc $(date -Is) ===" >>"$LOGDIR/_driver.log"

sub="$WT/outputs/search_20260923/$EXP/seed42/submission"
if [ -d "$sub" ]; then
    echo "PACKAGE $EXP: $(ls "$sub" | tr '\n' ' ')" >>"$LOGDIR/_driver.log"
else
    echo "PACKAGE MISSING $EXP" >>"$LOGDIR/_driver.log"
fi
