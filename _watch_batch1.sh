#!/usr/bin/env bash
# 第一批搜索批次的看门程序：每 30 分钟确认一次没有中断。
#
# 版本 3 —— 增加 newest_log()：B1 重跑写 .2 后缀日志，但重跑进程仍匹配同一个
# config 路径；若只看 .log 会读到 15:03 的旧崩溃日志，把正在推进的重跑误报成
# 「卡死」。stall/进度/异常扫描一律取该 run 最新的那份日志。
#
# 版本 2 —— 2026-09-23 15:03 B1 以 rc=134 (SIGABRT, CUDA error) 崩溃，
# 版本 1 报了 OK。根因：版本 1 只检查「当前活跃的那个 run」，B1 死掉后
# 编排立刻转去跑 B2，于是活跃 run 是健康的 B2，而 B1 的日志与
# _driver.log 里的 `rc=134` / `PACKAGE MISSING` 从来没有被读过。
# 版本 2 改为扫描 _driver.log 与全部 run 日志，不再只看活跃 run。
#
# 检查项：
#   1. 编排进程是否还活着（或已正常收尾 BATCH1 DONE）
#   2. 正在训练的 run，日志是否在推进（Progress 每 ~30s 一条，超阈值即卡死）
#   3. 【新】_driver.log 与全部 run 日志里的异常行：非零 rc、PACKAGE MISSING、
#      CUDA error、OOM、NaN。按「文件:行号」去重，只报一次，并永久记入
#      _INCIDENTS.log，不会因为后续轮次健康而被覆盖抹掉。
#   4. GPU 是否可达；训练中显存占用是否异常低
#   5. 磁盘余量
#   6. 编排活着但什么都没在跑、且 _driver.log 很久没更新 —— 编排停滞
#
# 注意：infer 进程的命令行也带 --config，必须按子命令区分 train/infer，
# 否则推理期间会被误判成「训练卡死」。
#
# 输出：$LOGDIR/_health.log 每轮一行；异常写 $LOGDIR/_ALERT.txt 并追加
# 明细到 $LOGDIR/_INCIDENTS.log（永久）。批次结束后自行退出。

set -u

WT=/home/clairvoyant/code/worktrees/search-mixup-anchor
LOGDIR="$WT/outputs/search_20260923/_logs"
HEALTH="$LOGDIR/_health.log"
ALERT="$LOGDIR/_ALERT.txt"
DRIVER="$LOGDIR/_driver.log"
INCIDENTS="$LOGDIR/_INCIDENTS.log"
SEEN="$LOGDIR/_watch_seen.txt"       # 已报过的「文件:行号」签名

INTERVAL=1800          # 30 分钟
STALL_MINUTES=20       # 训练中超过这么久没有新 Progress 视为卡死
IDLE_MINUTES=15        # 编排活着但无事可做超过这么久视为停滞
DISK_MIN_FREE_GB=20

# 异常行匹配（对 _driver.log 与 run 日志同用）
BAD_RE='rc=[1-9]|PACKAGE MISSING|cuda error|out of memory|nan detected|assertionerror|traceback'

mkdir -p "$LOGDIR"; touch "$SEEN" "$INCIDENTS"

RUNS=(
  "search_a1_mixup_a02|SEARCH_A1_MIXUP_A02"
  "search_a2_mixup_a04|SEARCH_A2_MIXUP_A04"
  "search_b1_anchor05_lr1e5|SEARCH_B1_ANCHOR05_LR1E5"
  "search_b2_anchor00_lr3e6|SEARCH_B2_ANCHOR00_LR3E6"
)

problems=()
active=""; active_cfg=""; phase="空闲"

note() { echo "$(date -Is) $*" >>"$HEALTH"; }
add()  { problems+=("$1"); }

age_of_log() {
    local f="$1" m
    [ -f "$f" ] || { echo 0; return; }
    m=$(stat -c %Y "$f" 2>/dev/null) || { echo 0; return; }
    echo $(( ($(date +%s) - m) / 60 ))
}

age_of_progress() {
    local f="$1" ts
    [ -f "$f" ] || { echo 0; return; }
    ts=$(grep Progress "$f" | tail -1 | grep -o '^[0-9-]* [0-9:]*')
    [ -n "$ts" ] || { echo 0; return; }
    echo $(( ($(date +%s) - $(date -d "$ts" +%s 2>/dev/null || date +%s)) / 60 ))
}

step_of() { grep Progress "$1" 2>/dev/null | tail -1 | grep -o '"epoch": [0-9]*, "global_step": [0-9]*'; }

# 取某个 run 最新的日志文件（含重跑产生的 .2 后缀）。
# B1 重跑时进程仍匹配同一个 config 路径，若只看 .log 会读到 15:03 的旧崩溃日志，
# 从而把正在推进的重跑误报成「卡死」。
newest_log() {
    ls -t "$LOGDIR/$1.log" "$LOGDIR/$1.log".* 2>/dev/null | head -1
}

# 扫描一个文件的异常行，按「文件:行号」去重；新发现的记入 problems 与 _INCIDENTS.log
scan_bad() {
    local f="$1" label="$2" m lineno line sig
    [ -f "$f" ] || return 0
    while IFS= read -r m; do
        [ -n "$m" ] || continue
        lineno=${m%%:*}
        line=${m#*:}
        sig="$f:$lineno"
        if ! grep -qxF "$sig" "$SEEN"; then
            echo "$sig" >>"$SEEN"
            printf '%s\t%s\t%s:%s\t%s\n' "$(date -Is)" "$label" "$f" "$lineno" \
                   "$(echo "$line" | tr -d '\t' | cut -c1-300)" >>"$INCIDENTS"
            add "$label 异常 @$(basename "$f"):$lineno → $(echo "$line" | cut -c1-150)"
        fi
    done < <(grep -inE "$BAD_RE" "$f" 2>/dev/null)
}

check_once() {
    problems=(); active=""; active_cfg=""; phase="空闲"
    local entry cfg exp training inferring

    # --- 1. 编排进程 ---
    if grep -q "BATCH1 DONE" "$DRIVER" 2>/dev/null; then
        # 收尾前仍要扫一遍，避免最后一次异常没被看见
        scan_bad "$DRIVER" "driver"
        if [ ${#problems[@]} -eq 0 ]; then
            note "OK 全部完成（BATCH1 DONE）"
            return 0
        fi
        note "ALERT 批次已结束但存在异常 :: ${problems[*]}"
        { echo "时间: $(date -Is)"; printf '  - %s\n' "${problems[@]}"; } >"$ALERT"
        return 0
    fi
    local orch_alive=0
    pgrep -f "_orchestrate_batch1.sh" >/dev/null 2>&1 && orch_alive=1
    [ "$orch_alive" = "0" ] && add "编排进程已消失，且未见 BATCH1 DONE —— 批次停止推进"

    # --- 2. 逐个 run 判断状态（严格区分 train / infer）---
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r cfg exp <<<"$entry"
        training=0; inferring=0
        pgrep -f "rematch train --config configs/$cfg.yaml" >/dev/null 2>&1 && training=1
        pgrep -f "rematch infer --config configs/$cfg.yaml" >/dev/null 2>&1 && inferring=1

        if [ "$training" = "1" ]; then
            active="$exp"; active_cfg="$cfg"; phase="训练中"
            local pa; pa=$(age_of_progress "$(newest_log "$cfg")")
            [ "$pa" -gt "$STALL_MINUTES" ] && \
                add "$exp 训练进程存活，但 Progress 已 $pa 分钟未推进（阈值 ${STALL_MINUTES}）"
        elif [ "$inferring" = "1" ]; then
            active="$exp"; active_cfg="$cfg"; phase="推理中"
            local ia; ia=$(age_of_log "$(newest_log "infer_$cfg")")
            [ "$ia" -gt 40 ] && add "$exp 推理已 $ia 分钟未输出（阈值 40）"
        fi
    done

    # --- 3. 全量异常扫描（不限于活跃 run）---
    scan_bad "$DRIVER" "driver"
    local f
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r cfg exp <<<"$entry"
        for f in "$LOGDIR/$cfg.log" "$LOGDIR/$cfg.log".*; do
            [ -f "$f" ] && scan_bad "$f" "$exp"
        done
        for f in "$LOGDIR/infer_$cfg.log" "$LOGDIR/infer_$cfg.log".*; do
            [ -f "$f" ] && scan_bad "$f" "$exp(infer)"
        done
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
        add "$active 标称训练中，但显存占用仅 ${gpu_mem}MiB —— 可能没真在算"
    fi

    # --- 5. 磁盘 ---
    local free_gb
    free_gb=$(df -BG "$WT" | awk 'NR==2{gsub("G","",$4); print $4}')
    [ -n "$free_gb" ] && [ "$free_gb" -lt "$DISK_MIN_FREE_GB" ] && \
        add "磁盘剩余仅 ${free_gb}G（阈值 ${DISK_MIN_FREE_GB}G）"

    # --- 已出包计数 ---
    local pkgs=0 e
    for e in "${RUNS[@]}"; do
        IFS='|' read -r cfg exp <<<"$e"
        [ -f "$WT/outputs/search_20260923/$exp/seed42/submission/submission.zip" ] && pkgs=$((pkgs+1))
    done

    # --- 汇总 ---
    local p
    if [ ${#problems[@]} -eq 0 ]; then
        note "OK phase=$phase active=${active:-无} pkgs=$pkgs/4 gpu=${gpu_mem:-?}MiB disk=${free_gb:-?}G $([ -n "$active_cfg" ] && step_of "$(newest_log "$active_cfg")" || echo "无进度")"
        rm -f "$ALERT"
    else
        note "ALERT phase=$phase active=${active:-无} pkgs=$pkgs/4 :: ${problems[*]}"
        {
            echo "时间: $(date -Is)"
            for p in "${problems[@]}"; do echo "  - $p"; done
            echo
            echo "--- _driver.log 尾部 ---"
            tail -12 "$DRIVER" 2>/dev/null
        } >"$ALERT"
    fi
}

note "=== 看门程序 v3 启动 pid=$$ 间隔=${INTERVAL}s ==="

while true; do
    check_once
    grep -q "BATCH1 DONE" "$DRIVER" 2>/dev/null && [ ${#problems[@]} -eq 0 ] && \
        { note "=== 看门程序正常退出 ==="; exit 0; }
    sleep "$INTERVAL"
done
