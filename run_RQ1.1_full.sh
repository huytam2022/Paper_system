#!/bin/bash
# =====================================================
# RQ1.1: Sybil & Collusion Robustness Experiments (PoVI)
# =====================================================

nodes_list=(50 100 250 500)
mal_list=(0.10 0.25 0.40)
tx_list=(4096 8192 16384)

outdir="results_RQ1.1"
mkdir -p "$outdir"

for n in "${nodes_list[@]}"; do
  for m in "${mal_list[@]}"; do
    for tx in "${tx_list[@]}"; do
      outfile="${outdir}/RQ1.1_n${n}_m$(echo $m | awk '{printf "%d", $1*100}')_tx${tx}.csv"
      echo "▶ Running: nodes=$n | malicious=$m | transactions=$tx"
      python3 RQ1.1.py \
        --transactions "$tx" \
        --nodes "$n" \
        --malicious-percent "$m" \
        --invalid-block-rate 0.30 \
        --rounds 200 \
        --proposer-strategy weighted \
        --export-csv "$outfile"
    done
  done
done
