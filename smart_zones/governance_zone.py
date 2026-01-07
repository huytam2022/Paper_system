from smart_zones.base_zone import SmartZone

class GovernanceZone(SmartZone):
    def __init__(self):
        super().__init__("GovernanceZone")

    def handle_tx(self, tx_id, tx_data):
        print(f"[GovernanceZone] 🏛️ Processing governance tx: {tx_id}")
        super().handle_tx(tx_id, tx_data)
