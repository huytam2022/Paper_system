# simclock.py
class SimClock:
    def __init__(self):
        self.t_ms = 0.0
    def advance(self, delta_ms: float):
        self.t_ms += max(0.0, float(delta_ms))
    def now_ms(self) -> float:
        return self.t_ms
