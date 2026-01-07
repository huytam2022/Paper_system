#!/usr/bin/env bash
set -euo pipefail
P="$1"; PROOF="$2"
sleep 0.06
test -f "$PROOF"
echo "[Groth16 stub] Verified $PROOF"