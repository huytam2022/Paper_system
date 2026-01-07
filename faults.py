# faults.py
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FaultConfig:
    # packet loss on message send
    packet_loss: float = 0.0        # e.g., 0.05
    # node churn: probability a node goes offline at a round boundary
    churn_rate: float = 0.0         # e.g., 0.10
    offline_rounds: int = 3         # offline duration when churn hits


class FaultInjector:
    """
    Central place to model:
    - packet loss/drop (intermittent connectivity)
    - churn (node goes offline for k rounds)
    """
    def __init__(self, cfg: FaultConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.offline_until: Dict[str, int] = {}  # node_id -> round index when it becomes online again

    def is_online(self, node_id: str, round_idx: int) -> bool:
        return self.offline_until.get(node_id, -1) <= round_idx

    def maybe_churn(self, node_id: str, round_idx: int) -> bool:
        """
        Call once per round per node.
        Returns True if node just went offline at this round.
        """
        if self.cfg.churn_rate <= 0:
            return False
        if not self.is_online(node_id, round_idx):
            return False
        if self.rng.random() < self.cfg.churn_rate:
            self.offline_until[node_id] = round_idx + max(1, int(self.cfg.offline_rounds))
            return True
        return False

    def should_drop(self) -> bool:
        if self.cfg.packet_loss <= 0:
            return False
        return self.rng.random() < self.cfg.packet_loss
