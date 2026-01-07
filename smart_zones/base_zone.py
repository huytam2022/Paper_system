class SmartZone:
    def __init__(self, name):
        self.name = name
        self.data_store = []

    def handle_tx(self, tx_id: str, tx_data: dict):
        """
        Xử lý giao dịch theo logic đặc thù zone.
        Mặc định chỉ lưu vào danh sách.
        """
        print(f"[{self.name}] 🔄 Handling tx: {tx_id}")
        self.data_store.append((tx_id, tx_data))

    def print_all(self):
        print(f"\n📁 [{self.name}] Stored TXs:")
        for i, (tx_id, data) in enumerate(self.data_store):
            print(f"{i+1}. {tx_id}: {data}")
