#!/usr/bin/env bash
set -euo pipefail
C="$1"; W="$2"; P="$3"; OUT="$4"
sleep 0.35
head -c 196608 </dev/urandom >"$OUT"  # ~192 KB
echo "[Groth16 stub] Proved -> $OUT"