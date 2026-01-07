#!/bin/bash
# Kích hoạt venv
source venv/bin/activate

# Danh sách node và transaction
NODE_LIST=(15 75 150)
TX_LIST=(512 1024 2048 4096 8192 16384)

# Vòng lặp qua từng nhóm node
for NODES in "${NODE_LIST[@]}"; do
    CSV_FILE="${NODES}nodes_cross.csv"
    rm -f $CSV_FILE  # Xóa file cũ nếu có

    # Vòng lặp chạy từng giá trị giao dịch
    for TX in "${TX_LIST[@]}"; do
        echo "=== Running cross-chain test: $TX transactions, $NODES nodes ==="
        python3 benmark_cross.py --transactions $TX --nodes $NODES --export-csv $CSV_FILE
    done
done
