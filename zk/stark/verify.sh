#!/usr/bin/env bash
set -euo pipefail
PUBLIC="$1"; PROOF="$2"

# Mô phỏng verify nhanh
sleep 0.1
# Kiểm tra file proof tồn tại
test -f "$PROOF"

echo "[STARK stub] Verified $PROOF với public input $PUBLIC"
