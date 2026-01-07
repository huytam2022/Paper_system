import json
import time

class SmartContract:
    def __init__(self, name: str):
        self.name = name
        self.tx_records = []

    def store_tx(self, tx_id: str, payload: dict):
        record = {
            "tx_id": tx_id,
            "payload": payload,
            "timestamp": int(time.time())
        }
        self.tx_records.append(record)
        print(f"[{self.name}] ✅ Stored tx: {tx_id}")

    def get_tx(self, index: int) -> dict:
        if index < len(self.tx_records):
            return self.tx_records[index]
        raise IndexError("Invalid index")

    def get_all(self):
        return self.tx_records

    def get_count(self):
        return len(self.tx_records)

    def print_all(self):
        print(f"\n[{self.name}] 📄 All Transactions:")
        for i, tx in enumerate(self.tx_records):
            print(f"Tx {i}: {json.dumps(tx, indent=2)}")
