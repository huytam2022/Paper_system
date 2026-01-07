class Dispatcher:
    def __init__(self):
    
        # Khởi tạo route cho từng zone
        self.routes = {
            "Smart Economy": [],
            "Smart Governance": [],
            "Smart People": [],
            "Smart Mobility": [],
            "Smart Environment": [],
            "Smart Living": []
        }

    def route_tx(self, tx_id: str, tx_type: str):
        if tx_type == "verify_request":
            # Giao dịch handshake không cần zone, chỉ đếm là đã route
            print(f"📨 Routed handshake request {tx_id} (DST → SRC)")
            return
        """Định tuyến giao dịch theo zone dựa vào type (không cần payload)."""
        if tx_type not in self.routes:
            print(f"❌ Unknown tx type: {tx_type}")
            return
        self.routes[tx_type].append(tx_id)
        print(f"📬 Routed tx {tx_id} → zone [{tx_type}]")

    def print_all(self):
        """In tổng hợp tất cả các giao dịch đã định tuyến."""
        for t, tx_ids in self.routes.items():
            print(f"\n📂 Zone = {t} (Total: {len(tx_ids)})")
            for tx_id in tx_ids:
                print(f" - {tx_id}")
