#!/bin/bash
# ===============================================
# RQ1.2: Network Partition Robustness Experiments
# ===============================================

configs=(
  "100 2 5 10"
  "150 3 5 10"
  "200 4 10 15"
  "500 5 10 20"
)

for cfg in "${configs[@]}"; do
  set -- $cfg
  nodes=$1
  partitions=$2
  txs=$3
  rounds=$4
  outfile="RQ1.2_n${nodes}_p${partitions}.csv"

  echo "▶ Running: nodes=$nodes | partitions=$partitions | tx-per-node=$txs | rounds=$rounds"
  python3 RQ1.2.py \
    --nodes "$nodes" \
    --partitions "$partitions" \
    --tx-per-node "$txs" \
    --rounds "$rounds"

  # Rename result file để tránh ghi đè
  mv RQ1.2.csv "results_${outfile}"
done
