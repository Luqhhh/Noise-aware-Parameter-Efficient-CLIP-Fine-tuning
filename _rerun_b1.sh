#!/usr/bin/env bash
# B1（锚点 0.5 + backbone_lr 1e-5）在 2026-09-23 15:03 以 CUDA unknown error 崩溃，
# 判断为环境级故障，需重跑取结果。本脚本等待 GPU 空出来后重跑 B1 训练 + 出包。
#
# 关键安全点：本机只有一张 8GB 卡，一次只能跑一个 full-FT（实测约 4.7GB）。
# 若与 B2 的出包推理并发，会显存不足 —— 那本身就会制造出与待查故障同类
# 的 CUDA 错误，把「环境故障」和「自伤」混为一谈。因此不靠「当前没有进程」
# 这种有竞态的判断（B2 训练退出到其 packager 启动 infer 之间有窗口），
# 而是等 _driver.log 出现 "B2 PACKAGER INFER END" 这个确定性标志。
#
# 日志写 .2 后缀，不覆盖原始崩溃日志（看门程序按行号去重，覆盖会让它重报）。

set -u

WT=/home/clairvoyant/code/worktrees/search-mixup-anchor
MAIN=/home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
LOGDIR="$WT/outputs/search_20260923/_logs"
DRIVER="$LOGDIR/_driver.log"
PY="$MAIN/.venv/bin/python"
CFG=search_b1_anchor05_lr1e5
EXP=SEARCH_B1_ANCHOR05_LR1E5

export PYTHONPATH="$WT/reproducibility/aegis_f1"
cd "$WT" || exit 1

echo "=== B1 RERUN waiting for B2 packaging to finish $(date -Is) ===" >>"$DRIVER"

# 等 B2 的 packager 出包结束（确定性标志），而非「当前无进程」的有竞态判断
while ! grep -q "B2 PACKAGER INFER END" "$DRIVER" 2>/dev/null; do
    sleep 30
done

# 双保险：确认确实没有 search_ 的训练/推理进程在跑
while pgrep -f "rematch (train|infer) --config configs/search_" >/dev/null 2>&1; do
    sleep 30
done

echo "=== B1 RERUN TRAIN START $(date -Is) ===" >>"$DRIVER"

"$PY" -u -m aegis_clip.cli.rematch train --config "configs/$CFG.yaml" \
    >"$LOGDIR/$CFG.log.2" 2>&1
rc=$?

echo "=== B1 RERUN TRAIN END rc=$rc $(date -Is) ===" >>"$DRIVER"

if [ "$rc" -eq 0 ]; then
    "$PY" -u -m aegis_clip.cli.rematch infer --config "configs/$CFG.yaml" \
        >"$LOGDIR/infer_$CFG.log.2" 2>&1
    irc=$?
    echo "=== B1 RERUN INFER END rc=$irc $(date -Is) ===" >>"$DRIVER"
    sub="$WT/outputs/search_20260923/$EXP/seed42/submission"
    if [ -d "$sub" ]; then
        echo "PACKAGE $EXP: $(ls "$sub" | tr '\n' ' ')" >>"$DRIVER"
    else
        echo "PACKAGE MISSING $EXP" >>"$DRIVER"
    fi
else
    echo "=== B1 RERUN 训练再次失败 rc=$rc，不出包 $(date -Is) ===" >>"$DRIVER"
fi
