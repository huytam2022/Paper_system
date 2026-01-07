from smart_zones.base_zone import SmartZone

class IdentityZone(SmartZone):
    def __init__(self):
        super().__init__("IdentityZone")

    def handle_tx(self, tx_id, tx_data):
        print(f"[IdentityZone] 🆔 Processing identity tx: {tx_id}")
        super().handle_tx(tx_id, tx_data)
