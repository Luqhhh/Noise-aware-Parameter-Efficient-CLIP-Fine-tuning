#!/usr/bin/env bash
# 通用重跑器：训练崩溃后 run_dir 残留，trainer.py:144 的
#   if run_dir.exists() and not (resume or overwrite): raise FileExistsError
# 会把重跑挡在第一步（B1 于 16:03:40 启动、16:03:58 以 rc=1 退出，18 秒，一个梯度步都没跑）。
#
# 而 `--overwrite` / `--resume` **在 CLI 上传不进去**：train 子命令只注册了 --config
# （cli/rematch.py:136-143），第 108 行是 train(config) 单参数调用；trainer.py:116/118 的
# 那两个是 Python 形参，配置里也没有对应键。所以出路只有自己腾路径。
#
# 本脚本不删除任何东西：把崩溃残留的 seed42 **改名**为 seed42.crashed-<ts>。
# 同名同盘 rename 是瞬时的，既腾出 run_dir 路径，又把崩溃现场（checkpoint、metrics.csv、
# 部分评测）原样留下，可回滚、可复查。这比 shutil.rmtree（--overwrite 内部的行为）安全。
#
# 单卡安全：本机只有一张 8GB 卡，一次只能跑一个 full-FT（实测 4.7GB）。与另一个
# full-FT 并发会显存不足 —— 那会制造出与待查故障同类的 CUDA 错误，把「环境故障」
# 和「自伤」混为一谈。因此等到确实没有任何 search_ 训练/推理进程才动手。
#
# 用法： bash _rerun_point.sh <cfg_name> <EXP_ID>

set -u

WT=/home/clairvoyant/code/worktrees/search-mixup-anchor
MAIN=/home/clairvoyant/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning
LOGDIR="$WT/outputs/search_20260923/_logs"
DRIVER="$LOGDIR/_driver.log"
FAILED="$WT/outputs/search_20260923/_failed"
PY="$MAIN/.venv/bin/python"

CFG="${1:?用法: _rerun_point.sh <cfg_name> <EXP_ID>}"
EXP="${2:?用法: _rerun_point.sh <cfg_name> <EXP_ID>}"
RUN="$WT/outputs/search_20260923/$EXP/seed42"

export PYTHONPATH="$WT/reproducibility/aegis_f1"
cd "$WT" || exit 1

echo "=== RERUN[$EXP] waiting for the card to clear $(date -Is) ===" >>"$DRIVER"
while pgrep -f "rematch (train|infer) --config configs/search_" >/dev/null 2>&1; do
    sleep 30
done

# 残留目录改名保全（不删）
if [ -e "$RUN" ]; then
    ts=$(date +%Y%m%dT%H%M%S)
    mkdir -p "$FAILED"
    mv "$RUN" "$FAILED/${EXP}_seed42.crashed-$ts" || {
        echo "=== RERUN[$EXP] 改名失败，放弃（不改名就跑不起来）$(date -Is) ===" >>"$DRIVER"
        exit 1
    }
    echo "=== RERUN[$EXP] 残留已改名为 _failed/${EXP}_seed42.crashed-$ts（保留现场）$(date -Is) ===" >>"$DRIVER"
fi

# 日志取下一个未用的 .N 后缀，避免覆盖崩溃日志（看门程序按「文件:行号」去重）
N=1
while [ -e "$LOGDIR/$CFG.log.$N" ]; do N=$((N+1)); done
LOG="$LOGDIR/$CFG.log.$N"

echo "=== RERUN[$EXP] TRAIN START log=$CFG.log.$N $(date -Is) ===" >>"$DRIVER"
"$PY" -u -m aegis_clip.cli.rematch train --config "configs/$CFG.yaml" >"$LOG" 2>&1
rc=$?
echo "=== RERUN[$EXP] TRAIN END rc=$rc $(date -Is) ===" >>"$DRIVER"

if [ "$rc" -ne 0 ]; then
    echo "=== RERUN[$EXP] 训练再次失败 rc=$rc，不出包 $(date -Is) ===" >>"$DRIVER"
    exit "$rc"
fi

# 训练成功 → 交给严格出包器（它会断言 selected_report.json 与终点轮次）
bash "$WT/_package_run.sh" "$CFG" "$EXP"
