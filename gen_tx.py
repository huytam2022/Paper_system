import json
import random
import os
from blockchains.Source_Chains import SourceChain
from contracts.smart_contract import SmartContract

# Danh sách các zone của Smart Cities (theo hình)
TYPES = [
    "Smart Governance",
    "Smart Economy",
    "Smart Environment",
    "Smart People",
    "Smart Mobility",   # Zone này sẽ bị quá tải
    "Smart Living"
]

# Gán trọng số không đồng đều: Smart Mobility sẽ nhận tải cao hơn 10 lần
WEIGHTS = [0.05, 0.05, 0.05, 0.05, 0.70, 0.10]  # Tổng = 1.0

# Danh sách số lượng giao dịch cần tạo
sizes = [512, 1024, 2048, 4096, 8192, 16384]  # Duy trì tải cao trong thời gian dài

for num_tx in sizes:
    print(f"🚧 Đang tạo {num_tx} giao dịch không đồng đều...")

    source = SourceChain("SRC")
    contractA = SmartContract("SmartContractA")

    transactions = []

    for i in range(num_tx):
        tx_id = f"tx{i:05d}"
        zone = random.choices(TYPES, weights=WEIGHTS, k=1)[0]  # chọn theo trọng số
        payload = {
            "citizen_id": f"VN{i:05d}",
            "type": zone,
            "claim": "sufficient_income_and_no_debt"
        }
        tx_str = source.add_transaction(payload, zone, tx_id)
        contractA.store_tx(tx_id, payload)

        transactions.append({
            "tx_id": tx_id,
            "payload": payload
        })

    # Ghi ra file JSON
    filename = f"hetero_transactions_{num_tx}.json"
    with open(filename, "w") as f:
        json.dump({"transactions": transactions}, f, indent=2)

    print(f"✅ Đã lưu vào '{filename}' ({num_tx} giao dịch)")

print("\n🎉 Hoàn tất tạo tất cả file.")