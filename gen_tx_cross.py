import json
import random
import os

# Các loại giao dịch chính (mô phỏng các lĩnh vực của Smart City)
TYPES = [
    "Smart Governance",
    "Smart Economy",
    "Smart Environment",
    "Smart People",
    "Smart Mobility",
    "Smart Living"
]

def generate_cross_chain_transactions(num_tx: int, output_file: str):
    """Sinh ra num_tx giao dịch chính, kèm DST request (tổng gấp đôi), và lưu thành JSON."""
    transactions = []

    for i in range(num_tx):
        tx_id = f"tx_{i:06d}"
        tx_type = random.choice(TYPES)
        citizen_id = f"VN{i:06d}"

        # Giao dịch chính (SRC → DST)
        main_tx_payload = {
            "citizen_id": citizen_id,
            "type": tx_type,
            "action": "main_transfer"
        }

        # Giao dịch request (DST → SRC, mô phỏng handshake)
        request_tx_id = f"req_{i:06d}"
        request_tx_payload = {
            "citizen_id": citizen_id,
            "type": "verify_request",
            "origin": "DST",
            "target_tx": tx_id
        }

        # Đóng gói theo định dạng mà benmark_cross.py sử dụng
        transactions.append({
            "tx_id": tx_id,
            "payload": main_tx_payload
        })
        transactions.append({
            "tx_id": request_tx_id,
            "payload": request_tx_payload
        })

    # Ghi ra file JSON
    with open(output_file, "w") as f:
        json.dump({"transactions": transactions}, f, indent=2)

    print(f"✅ Generated {len(transactions)} transactions (including DST requests) to {output_file}")


if __name__ == "__main__":
    # Tự động sinh 6 dataset cho các mức transactions
    sizes = [512, 1024, 2048, 4096, 8192, 16384]
    for size in sizes:
        output_file = f"raw_transactions_cross_{size}.json"
        generate_cross_chain_transactions(size, output_file)

    print("\n🎯 All datasets generated successfully for sizes: 512, 1024, 2048, 4096, 8192, 16384.\n")
