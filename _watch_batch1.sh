#!/usr/bin/env bash
# 第一批搜索批次的看门程序：每 30 分钟确认一次没有中断。
#
# 检查项：
#   1. 编排进程是否还活着（或已正常收尾 BATCH1 DONE）
#   2. 正在训练的 run，日志是否在推进（Progress 每 ~34s 一条，超阈值即卡死）
#   3. 正在训练的 run，日志尾部有无 Traceback / OOM / CUDA error
#   4. GPU 是否可达；训练中显存占用是否异常低
#   5. 磁盘余量
#   6. 编排活着但什么都没在跑、且 _driver.log 很久没更新 —— 编排停滞
#
# 注意：infer 进程的命令行也带 --config，必须按子命令区分 train/infer，
# 否则推理期间会被误判成「训练卡死」。
#
# 输出：$LOGDIR/_health.log 每轮一行；异常时写 $LOGDIR/_ALERT.txt 留明细。
# 全部跑完后自行退出，不留僵尸进程。

set -u

WT=/home/clairvoyant/code/worktrees/search-mixup-anchor
LOGDIR="$WT/outputs/search_20260923/_logs"
HEALTH="$LOGDIR/_health.log"
ALERT="$LOGDIR/_ALERT.txt"
DRIVER="$LOGDIR/_driver.log"

INTERVAL=1800          # 30 分钟
STALL_MINUTES=20       # 训练中超过这么久没有新 Progress 视为卡死
IDLE_MINUTES=15        # 编排活着但无事可做超过这么久视为停滞
DISK_MIN_FREE_GB=20

mkdir -p "$LOGDIR"

RUNS=(
  "search_a1_mixup_a02|SEARCH_A1_MIXUP_A02"
  "search_a2_mixup_a04|SEARCH_A2_MIXUP_A04"
  "search_b1_anchor05_lr1e5|SEARCH_B1_ANCHOR05_LR1E5"
  "search_b2_anchor00_lr3e6|SEARCH_B2_ANCHOR00_LR3E6"
)

problems=()
active=""
active_cfg=""
phase="空闲"

note() { echo "$(date -Is) $*" >>"$HEALTH"; }
add()  { problems+=("$1"); }

age_of_log() {   # 日志 mtime 距今多少分钟
    local f="$1" m
    [ -f "$f" ] || { echo 0; return; }
    m=$(stat -c %Y "$f" 2>/dev/null) || { echo 0; return; }
    echo $(( ($(date +%s) - m) / 60 ))
}

# 最后一条 Progress 的时间戳距今多少分钟
age_of_progress() {
    local f="$1" ts
    [ -f "$f" ] || { echo 0; return; }
    ts=$(grep Progress "$f" | tail -1 | grep -o '^[0-9-]* [0-9:]*')
    [ -n "$ts" ] || { echo 0; return; }
    echo $(( ($(date +%s) - $(date -d "$ts" +%s 2>/dev/null || date +%s)) / 60 ))
}

step_of() { grep Progress "$1" 2>/dev/null | tail -1 | grep -o '"epoch": [0-9]*, "global_step": [0-9]*'; }

check_once() {
    problems=(); active=""; active_cfg=""; phase="空闲"
    local now; now=$(date +%s)

    # --- 1. 编排进程 ---
    if grep -q "BATCH1 DONE" "$DRIVER" 2>/dev/null; then
        note "OK 全部完成（BATCH1 DONE）"
        return 0
    fi
    local orch_alive=0
    pgrep -f "_orchestrate_batch1.sh" >/dev/null 2>&1 && orch_alive=1
    [ "$orch_alive" = "0" ] && add "编排进程已消失，且未见 BATCH1 DONE —— 批次停止推进"

    # --- 2/3. 逐个 run 判断状态（严格区分 train / infer）---
    local entry cfg exp training inferring
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r cfg exp <<<"$entry"
        training=0; inferring=0
        pgrep -f "rematch train --config configs/$cfg.yaml" >/dev/null 2>&1 && training=1
        pgrep -f "rematch infer --config configs/$cfg.yaml" >/dev/null 2>&1 && inferring=1

        if [ "$training" = "1" ]; then
            active="$exp"; active_cfg="$cfg"; phase="训练中"
            local pa
            pa=$(age_of_progress "$LOGDIR/$cfg.log")
            [ "$pa" -gt "$STALL_MINUTES" ] && \
                add "$exp 训练进程存活，但 Progress 已 $pa 分钟未推进（阈值 ${STALL_MINUTES}）"
            if tail -60 "$LOGDIR/$cfg.log" 2>/dev/null | \
               grep -qiE "traceback|out of memory|cuda error|nan detected|assertionerror"; then
                add "$exp 日志尾部出现错误关键字（Traceback/OOM/CUDA error）"
            fi
        elif [ "$inferring" = "1" ]; then
            active="$exp"; active_cfg="$cfg"; phase="推理中"
            local ia
            ia=$(age_of_log "$LOGDIR/infer_$cfg.log")
            [ "$ia" -gt 40 ] && add "$exp 推理已 $ia 分钟未输出（阈值 40）"
        fi
    done

    # --- 6. 编排活着但无事可做 ---
    if [ "$orch_alive" = "1" ] && [ -z "$active" ]; then
        local da; da=$(age_of_log "$DRIVER")
        [ "$da" -gt "$IDLE_MINUTES" ] && \
            add "编排存活但无训练/推理进程，_driver.log 已 $da 分钟未更新（阈值 ${IDLE_MINUTES}）"
    fi

    # --- 4. GPU ---
    local gpu_mem
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -z "$gpu_mem" ]; then
        add "nvidia-smi 无输出 —— GPU 不可达"
    elif [ "$phase" = "训练中" ] && [ "$gpu_mem" -lt 1000 ]; then
        add "$active 标称训练中，但显存占用仅 ${gpu_mem}MiB —— 可能没真正在算"
    fi

    # --- 5. 磁盘 ---
    local free_gb
    free_gb=$(df -BG "$WT" | awk 'NR==2{gsub("G","",$4); print $4}')
    [ -n "$free_gb" ] && [ "$free_gb" -lt "$DISK_MIN_FREE_GB" ] && \
        add "磁盘剩余仅 ${free_gb}G（阈值 ${DISK_MIN_FREE_GB}G）"

    # --- 汇总 ---
    if [ ${#problems[@]} -eq 0 ]; then
        note "OK phase=$phase active=${active:-无} gpu=${gpu_mem:-?}MiB disk=${free_gb:-?}G $([ -n "$active_cfg" ] && step_of "$LOGDIR/$active_cfg.log" || echo "无进度")"
        rm -f "$ALERT"
    else
        local p
        note "ALERT phase=$phase active=${active:-无} :: ${problems[*]}"
        {
            echo "时间: $(date -Is)"
            for p in "${problems[@]}"; do echo "  - $p"; done
            echo
            echo "--- _driver.log 尾部 ---"
            tail -20 "$DRIVER" 2>/dev/null
        } >"$ALERT"
    fi
}

note "=== 看门程序启动 pid=$$ 间隔=${INTERVAL}s ==="

while true; do
    check_once
    grep -q "BATCH1 DONE" "$DRIVER" 2>/dev/null && { note "=== 看门程序正常退出 ==="; exit 0; }
    sleep "$INTERVAL"
done
