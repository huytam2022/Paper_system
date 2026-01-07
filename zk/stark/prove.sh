#!/usr/bin/env bash
set -euo pipefail
CIRCUIT="$1"; WITNESS="$2"; PUBLIC="$3"; PROOF_OUT="$4"

# Mô phỏng thời gian tạo proof
sleep 0.5
# Ghi file proof giả (512KB)
head -c 524288 </dev/urandom >"$PROOF_OUT"

echo "[STARK stub] Proved $CIRCUIT với witness $WITNESS -> $PROOF_OUT"
