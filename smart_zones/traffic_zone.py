from smart_zones.base_zone import SmartZone

class TrafficZone(SmartZone):
    def __init__(self):
        super().__init__("TrafficZone")

    def handle_tx(self, tx_id, tx_data):
        print(f"[TrafficZone] 🚦 Processing traffic tx: {tx_id}")
        super().handle_tx(tx_id, tx_data)
