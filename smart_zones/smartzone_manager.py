from smart_zones.finance_zone import FinanceZone
from smart_zones.identity_zone import IdentityZone
from smart_zones.governance_zone import GovernanceZone
from smart_zones.traffic_zone import TrafficZone
from smart_zones.environment_zone import EnvironmentZone
from smart_zones.housing_zone import HousingZone

class SmartZoneManager:
    def __init__(self):
        self.zones = {
            "finance": FinanceZone(),
            "identity": IdentityZone(),
            "governance": GovernanceZone(),
            "traffic": TrafficZone(),
            "environment": EnvironmentZone(),
            "housing": HousingZone()
        }

    def dispatch(self, tx_id: str, tx_data: dict):
        tx_type = tx_data.get("type")
        zone = self.zones.get(tx_type)
        if zone:
            zone.handle_tx(tx_id, tx_data)
        else:
            print(f"❌ No zone for type: {tx_type}")

    def print_all(self):
        for zone in self.zones.values():
            zone.print_all()
