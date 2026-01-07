#!/bin/bash
# ===========================================================
# run_RQ2.sh
# Auto-run script for RQ2 (multi-node, multi-transaction loads)
# Usage: ./run_RQ2.sh
# -----------------------------------------------------------
# Chạy các thí nghiệm RQ2 với nhiều cấu hình nodes/transactions
# và lưu kết quả + log riêng biệt cho từng lần chạy.
# ===========================================================

set -euo pipefail

# ============================
# CONFIGURATION
# ============================
nodes_list=(50 100 250 500)          # số lượng node trong mạng
tx_list=(4096 8192 16384)         # số lượng giao dịch mỗi thí nghiệm
repeats=5                         # số lần lặp để lấy mean/std
outdir="results_RQ2"              # thư mục lưu kết quả
logdir="${outdir}/logs"           # thư mục lưu log
script_name="RQ2.py"              # file Python thực thi
timestamp_root=$(date +%Y%m%d_%H%M%S)

# ============================
# PREPARE OUTPUT FOLDERS
# ============================
mkdir -p "$outdir"
mkdir -p "$logdir"

echo "=== 🚀 RQ2 batch run started: $(date) ==="
echo "Results folder: $outdir"
echo "Logs folder: $logdir"
echo "---------------------------------------------"
echo "Nodes:         ${nodes_list[*]}"
echo "Transactions:  ${tx_list[*]}"
echo "Repeats:       $repeats"
echo "---------------------------------------------"
echo

# ============================
# MAIN EXECUTION LOOP
# ============================
for n in "${nodes_list[@]}"; do
  for tx in "${tx_list[@]}"; do
    for runidx in $(seq 1 $repeats); do
      run_ts=$(date +%Y%m%d_%H%M%S)
      echo "▶ Run: nodes=$n | tx=$tx | run#${runidx} | started at $run_ts"

      # log file for this run
      run_log="${logdir}/run_n${n}_tx${tx}_r${runidx}_${run_ts}.log"

      # execute experiment
      python3 "$script_name" --transactions "$tx" --nodes "$n" > "$run_log" 2>&1 || {
        echo "❌ Run failed: nodes=$n tx=$tx run#${runidx}. Check log: $run_log"
        continue
      }

      # expected output CSV file (e.g., "8192_single.csv")
      generated_csv="${tx}_single.csv"
      if [[ -f "$generated_csv" ]]; then
        dest="${outdir}/RQ2_n${n}_tx${tx}_run${runidx}_${run_ts}.csv"
        mv -f "$generated_csv" "$dest"
        echo "✔ Saved CSV -> $dest"
      else
        echo "⚠️ Expected CSV '$generated_csv' not found (check $run_log)"
      fi

      sleep 0.5
    done
    echo "---------------------------------------------"
  done
done

echo
echo "=== ✅ All RQ2 runs finished: $(date) ==="
echo "Results saved under: $outdir"
echo "Logs saved under: $logdir"
echo "=========================================================="
