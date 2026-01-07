from smart_zones.base_zone import SmartZone

class FinanceZone(SmartZone):
    def __init__(self):
        super().__init__("FinanceZone")

    def handle_tx(self, tx_id, tx_data):
        print(f"[FinanceZone] 💰 Processing tx: {tx_id}")
        super().handle_tx(tx_id, tx_data)
