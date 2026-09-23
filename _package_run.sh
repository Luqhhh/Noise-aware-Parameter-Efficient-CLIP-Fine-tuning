#!/usr/bin/env bash
# 通用出包器：等某个 run 的训练进程退出后跑 infer，然后**验证包是不是预定终点的模型**。
#
# 为什么需要它（2026-09-23 的教训）：B2 在 16:00 以 CUDA unknown error 崩溃于第 5 轮，
# 但 _package_b2.sh 只断言了 submission 目录存在，于是照样报了 PACKAGE —— 那个包是
# 16 轮计划的第 4 轮模型。平台 9 项结构校验全过（单 checkpoint、确定性推理，合规），
# 结构上完全像个好包，只有 selected_report.json 能戳穿它。
#
# 因此本脚本的判据不是「目录在不在」，而是三条：
#   1. 训练进程真的退出了（自己的 pgrep 不会匹配到本脚本）
#   2. selected_report.json 存在 —— 它是训练跑到终点才写的；崩溃的 run 没有
#   3. selected_epoch == schedule_epochs —— last_epoch 语义下的「预定终点」
# 任一条不满足就打 INVALID，不出包，不声称 PACKAGE。
#
# 用法： bash _package_run.sh <cfg_name> <EXP_ID>

set -u

WT=/home/clairvoyant/code/worktrees/search-mixup-anchor
MAIN=/home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
LOGDIR="$WT/outputs/search_20260923/_logs"
DRIVER="$LOGDIR/_driver.log"
PY="$MAIN/.venv/bin/python"

CFG="${1:?用法: _package_run.sh <cfg_name> <EXP_ID>}"
EXP="${2:?用法: _package_run.sh <cfg_name> <EXP_ID>}"
RUN="$WT/outputs/search_20260923/$EXP/seed42"

export PYTHONPATH="$WT/reproducibility/aegis_f1"
cd "$WT" || exit 1

echo "=== PACKAGER[$EXP] waiting $(date -Is) ===" >>"$DRIVER"

while pgrep -f "rematch train --config configs/$CFG.yaml" >/dev/null 2>&1; do
    sleep 30
done

echo "=== PACKAGER[$EXP] train gone, running infer $(date -Is) ===" >>"$DRIVER"

"$PY" -u -m aegis_clip.cli.rematch infer --config "configs/$CFG.yaml" \
    >"$LOGDIR/infer_$CFG.log" 2>&1
rc=$?
echo "=== PACKAGER[$EXP] INFER END rc=$rc $(date -Is) ===" >>"$DRIVER"

# --- 终点语义验证：这才是「包能不能用」的判据 ---
report="$RUN/checkpoints/selected_report.json"
if [ ! -f "$report" ]; then
    echo "PACKAGE INVALID $EXP: selected_report.json 不存在 —— 训练未跑到终点，包不是预定终点的模型" >>"$DRIVER"
    exit 1
fi

verdict=$("$PY" - "$report" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
sel, sched = r.get("selected_epoch"), r.get("schedule_epochs")
if sel != sched:
    print(f"MISMATCH selected_epoch={sel} schedule_epochs={sched}")
else:
    print(f"OK selected_epoch={sel} micro={r.get('raw_micro'):.4f} macro={r.get('raw_macro'):.4f}")
EOF
)

sub="$RUN/submission"
if [ -z "$(ls -A "$sub" 2>/dev/null)" ]; then
    echo "PACKAGE INVALID $EXP: submission 目录为空（infer rc=$rc）" >>"$DRIVER"
    exit 1
fi

case "$verdict" in
    OK*)
        echo "PACKAGE $EXP: $verdict :: $(ls "$sub" | tr '\n' ' ')" >>"$DRIVER"
        ;;
    *)
        echo "PACKAGE INVALID $EXP: $verdict —— 被选轮次与总轮数不符，不得上传" >>"$DRIVER"
        exit 1
        ;;
esac
