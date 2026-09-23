#!/usr/bin/env bash
# B2 重跑的看门程序：每 30 分钟确认一次没有中断。
#
# 为什么不复用 _watch_batch1.sh v3：它的 check_once 第一句就是
#   if grep -q "BATCH1 DONE" "$DRIVER"; then ... return 0; fi
# 而 _driver.log 里已经有 BATCH1 DONE —— 于是它只扫一遍 _driver.log 就早退，
# 完全不看任何 run 的状态；异常行已去重，所以下一轮 problems 为空，它会写一句
# 「OK 全部完成（BATCH1 DONE）」然后退出。那句话在重跑进行中就是误导。
#
# 本脚本只盯一个 run：B2 重跑。
# 检查项：
#   1. 训练进程是否活着 / 出包进程是否活着
#   2. Progress 是否在推进（每 ~30s 一条，超阈值即卡死）
#   3. 新日志与 _driver.log 的异常行（rc≠0 / CUDA error / OOM / NaN / traceback）
#   4. GPU 是否可达；标称训练中显存却极低
#   5. 磁盘余量
# 训练与出包进程都消失后写一条收尾摘要并退出。

set -u

WT=/home/clairvoyant/code/worktrees/search-mixup-anchor
LOGDIR="$WT/outputs/search_20260923/_logs"
HEALTH="$LOGDIR/_health.log"
ALERT="$LOGDIR/_ALERT.txt"
DRIVER="$LOGDIR/_driver.log"
INCIDENTS="$LOGDIR/_INCIDENTS.log"
SEEN="$LOGDIR/_watch_rerun_seen.txt"

CFG=search_b2_anchor00_lr3e6
EXP=SEARCH_B2_ANCHOR00_LR3E6

INTERVAL=1800
STALL_MINUTES=20
DISK_MIN_FREE_GB=20
BAD_RE='rc=[1-9]|PACKAGE INVALID|PACKAGE MISSING|cuda error|out of memory|nan detected|assertionerror|traceback'

mkdir -p "$LOGDIR"; touch "$SEEN" "$INCIDENTS"

problems=()
note() { echo "$(date -Is) $*" >>"$HEALTH"; }
add()  { problems+=("$1"); }

# 用 [_] 括号技巧，避免 pgrep 匹配到自己所在的命令行
train_alive()   { pgrep -f "rematch train --config configs/$CFG.yaml" >/dev/null 2>&1; }
package_alive() { pgrep -f "[_]package_run.sh $CFG" >/dev/null 2>&1; }

newest_log() { ls -t "$LOGDIR/$CFG.log" "$LOGDIR/$CFG.log".* 2>/dev/null | head -1; }

age_of_progress() {
    local f="$1" ts
    [ -f "$f" ] || { echo 0; return; }
    ts=$(grep Progress "$f" | tail -1 | grep -o '^[0-9-]* [0-9:]*')
    [ -n "$ts" ] || { echo 0; return; }
    echo $(( ($(date +%s) - $(date -d "$ts" +%s 2>/dev/null || date +%s)) / 60 ))
}

step_of() { grep Progress "$1" 2>/dev/null | tail -1 | grep -o '"epoch": [0-9]*, "global_step": [0-9]*'; }

scan_bad() {
    local f="$1" label="$2" m lineno line sig
    [ -f "$f" ] || return 0
    while IFS= read -r m; do
        [ -n "$m" ] || continue
        lineno=${m%%:*}; line=${m#*:}; sig="$f:$lineno"
        if ! grep -qxF "$sig" "$SEEN"; then
            echo "$sig" >>"$SEEN"
            printf '%s\t%s\t%s:%s\t%s\n' "$(date -Is)" "$label" "$f" "$lineno" \
                   "$(echo "$line" | tr -d '\t' | cut -c1-300)" >>"$INCIDENTS"
            add "$label 异常 @$(basename "$f"):$lineno → $(echo "$line" | cut -c1-150)"
        fi
    done < <(grep -inE "$BAD_RE" "$f" 2>/dev/null)
}

check_once() {
    problems=()
    local phase="空闲" log pa

    if train_alive; then
        phase="训练中"
        log=$(newest_log)
        pa=$(age_of_progress "$log")
        [ "$pa" -gt "$STALL_MINUTES" ] && \
            add "$EXP 训练进程存活，但 Progress 已 $pa 分钟未推进（阈值 ${STALL_MINUTES}）"
    elif package_alive; then
        phase="出包中"
    fi

    scan_bad "$DRIVER" "driver"
    local f
    for f in "$LOGDIR/$CFG.log" "$LOGDIR/$CFG.log".* "$LOGDIR/infer_$CFG.log" "$LOGDIR/infer_$CFG.log".*; do
        [ -f "$f" ] && scan_bad "$f" "$EXP"
    done

    local gpu_mem
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -z "$gpu_mem" ]; then
        add "nvidia-smi 无输出 —— GPU 不可达"
    elif [ "$phase" = "训练中" ] && [ "$gpu_mem" -lt 1000 ]; then
        add "$EXP 标称训练中，但显存占用仅 ${gpu_mem}MiB —— 可能没真在算"
    fi

    local free_gb
    free_gb=$(df -BG "$WT" | awk 'NR==2{gsub("G","",$4); print $4}')
    [ -n "$free_gb" ] && [ "$free_gb" -lt "$DISK_MIN_FREE_GB" ] && \
        add "磁盘剩余仅 ${free_gb}G（阈值 ${DISK_MIN_FREE_GB}G）"

    local p
    if [ ${#problems[@]} -eq 0 ]; then
        note "RERUN OK phase=$phase gpu=${gpu_mem:-?}MiB disk=${free_gb:-?}G $([ "$phase" = "训练中" ] && step_of "$(newest_log)" || echo "无进度")"
        rm -f "$ALERT"
    else
        note "RERUN ALERT phase=$phase :: ${problems[*]}"
        {
            echo "时间: $(date -Is)"
            for p in "${problems[@]}"; do echo "  - $p"; done
            echo
            echo "--- _driver.log 尾部 ---"
            tail -12 "$DRIVER" 2>/dev/null
        } >"$ALERT"
    fi
}

note "=== 重跑看门程序启动 pid=$$ 间隔=${INTERVAL}s 目标=$EXP ==="

# 先等重跑真正开始，避免启动瞬间误判「进程消失」
for _ in $(seq 1 20); do
    train_alive && break
    package_alive && break
    sleep 15
done
[ "$(newest_log)" = "$LOGDIR/$CFG.log" ] && note "RERUN 警告：只看到原始崩溃日志，重跑的 .N 日志尚未出现"

while true; do
    check_once
    if ! train_alive && ! package_alive; then
        note "=== 重跑看门程序退出：训练与出包进程均已消失 ==="
        tail -6 "$DRIVER" >>"$HEALTH"
        exit 0
    fi
    sleep "$INTERVAL"
done
