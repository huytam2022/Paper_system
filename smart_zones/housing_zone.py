from smart_zones.base_zone import SmartZone

class HousingZone(SmartZone):
    def __init__(self):
        super().__init__("HousingZone")

    def handle_tx(self, tx_id, tx_data):
        print(f"[HousingZone] 🏘️ Processing housing tx: {tx_id}")
        super().handle_tx(tx_id, tx_data)
