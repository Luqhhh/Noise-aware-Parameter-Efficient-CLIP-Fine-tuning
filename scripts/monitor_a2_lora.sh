#!/usr/bin/env bash
set -u

repo="${1:-$HOME/code/Noise-aware-Parameter-Efficient-CLIP-Fine-tuning}"
cd "$repo"

echo "=== $(date --iso-8601=seconds) ==="
for unit in a2-lora-min.service a2-lora-full.service; do
  echo "[$unit] $(systemctl --user is-active "$unit" 2>/dev/null || true)"
  journalctl --user -u "$unit" -n 8 --no-pager 2>/dev/null || true
done

for csv in outputs/a2_lora_min_knn/seed42/logs/metrics.csv \
           outputs/a2_lora_full_knn/seed42/logs/metrics.csv; do
  if [[ -f "$csv" ]]; then
    echo "--- $csv ---"
    tail -n 2 "$csv"
  fi
done
