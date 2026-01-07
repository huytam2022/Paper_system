from smart_zones.base_zone import SmartZone

class EnvironmentZone(SmartZone):
    def __init__(self):
        super().__init__("EnvironmentZone")

    def handle_tx(self, tx_id, tx_data):
        print(f"[EnvironmentZone] 🌿 Processing environment tx: {tx_id}")
        super().handle_tx(tx_id, tx_data)
