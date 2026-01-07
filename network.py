# network.py
from __future__ import annotations
import random
from dataclasses import dataclass

@dataclass
class NetConfig:
    loss: float = 0.0           # 0.0 -> 0.2
    base_delay_ms: float = 20.0 # baseline link delay
    jitter_ms: float = 10.0     # random jitter
    timeout_ms: float = 200.0   # retry timeout
    max_retries: int = 2        # don't explode

class Network:
    def __init__(self, cfg: NetConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = random.Random(seed)

    def sample_delay_ms(self) -> float:
        # non-negative delay
        d = self.cfg.base_delay_ms + self.rng.uniform(-self.cfg.jitter_ms, self.cfg.jitter_ms)
        return max(0.0, d)

    def should_drop(self) -> bool:
        return self.rng.random() < self.cfg.loss
