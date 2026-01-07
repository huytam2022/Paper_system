#!/bin/bash
# Kích hoạt venv trước (Linux/Mac)
source venv/bin/activate

# -------- Cấu hình --------
NODES=$1                      # Số node truyền từ tham số (ví dụ 50, 100, 250, 500)
CSV_FILE="${NODES}nodes_single.csv"  # Tên file CSV lưu kết quả
TX_LIST=(512 1024 2048 4096 8192 16384)  # Danh sách transaction

# Xóa CSV cũ nếu đã tồn tại
rm -f $CSV_FILE

# -------- Chạy lần lượt các testcase --------
for TX in "${TX_LIST[@]}"; do
    echo "=== Running test: $TX transactions, $NODES nodes ==="
    python3 benmark.py --transactions $TX --nodes $NODES --export-csv $CSV_FILE
done
