# experiments/run_sensitivity_failure.py
from __future__ import annotations
import argparse, csv, os
import numpy as np
import time
from collections import deque
import logging
logging.getLogger().setLevel(logging.WARNING)

from network import NetConfig, Network
from simclock import SimClock

# import/build your system here
from blockchains.node import Node
from consensus_layer import LightConsensus
from blockchains.Source_Chains import SourceChain

# --- BEGIN monkey patch: add network+clock send wrapper to Node ---
def _attach_network(self, net, clock):
    self._net = net
    self._clock = clock

def _send_with_retry(self, peer, msg: dict) -> bool:
    net = getattr(self, "_net", None)
    clock = getattr(self, "_clock", None)

    # fallback if not attached
    if net is None or clock is None:
        if hasattr(peer, "on_message"):
            peer.on_message(msg)
        return True

    for _ in range(net.cfg.max_retries + 1):
        clock.advance(net.sample_delay_ms())
        if net.should_drop():
            clock.advance(net.cfg.timeout_ms)
            continue
        if hasattr(peer, "on_message"):
            peer.on_message(msg)
        return True

    return False

def _broadcast(self, msg: dict) -> int:
    delivered = 0
    for p in getattr(self, "peers", []):
        if _send_with_retry(self, p, msg):
            delivered += 1
    return delivered

# Only add if missing (safe)
if not hasattr(Node, "attach_network"):
    Node.attach_network = _attach_network
if not hasattr(Node, "send_with_retry"):
    Node.send_with_retry = _send_with_retry
if not hasattr(Node, "broadcast"):
    Node.broadcast = _broadcast
# --- END monkey patch ---

def ensure_dir(p): os.makedirs(p, exist_ok=True)

from contextlib import contextmanager
import sys
import os

@contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

def sample_lognormal_ms(rng, mean_ms: float, sigma: float = 0.35, cap_mult: float = 5.0) -> float:
    mean_ms = max(1e-3, float(mean_ms))
    x = rng.lognormal(mean=np.log(mean_ms), sigma=sigma)
    return float(min(x, mean_ms * cap_mult))

def run_one(loss: float, seed: int, n_nodes: int, n_txs: int, log_every: int = 500):
    # EWMA for realistic ETA
    ewma_rate = None
    alpha = 0.2

    clock = SimClock()
    rng = np.random.default_rng(seed)

    # arrival rate (tx/s) in simulated time
    arrival_rate = 60.0  # tune: 40-90 realistic for edge gateway
    burst_prob = 0.03     # occasional burst
    burst_size = 30       # additional arrivals on burst

    # make it less ideal (recommended defaults)
    net = Network(
        NetConfig(loss=loss, base_delay_ms=35, jitter_ms=25, timeout_ms=350, max_retries=1),
        seed=seed
    )

    # build nodes
    nodes = [Node(node_id=f"n{i}", chain_class=SourceChain, malicious=False, reputation=1.0) for i in range(n_nodes)]
    for i, n in enumerate(nodes):
        for j in range(1, 4):
            n.connect(nodes[(i + j) % n_nodes])
        n.attach_network(net, clock)

    cons = LightConsensus(nodes=nodes)

    latencies = []
    committed = 0
    backlog = 0  # simple congestion proxy (optional but helps realism)

    t0 = time.time()     # wall-clock for progress only
    last_log = t0
    # window for realistic instantaneous throughput / eta stability
    win = 1000
    t_wall_win = deque(maxlen=win)     # wall timestamps
    t_sim_win  = deque(maxlen=win)     # sim timestamps
    comm_win   = deque(maxlen=win)     # committed count (cumulative)

    # submit a workload
    for k in range(n_txs):
        # arrivals based on simulated time increment of last step ~ use small dt
        # approximate dt= (queue + compute + network) will be added later,
        # so here inject arrivals using expected arrivals per step.
        expected = arrival_rate * 0.02  # assume ~20ms step baseline
        arrivals = rng.poisson(expected)
        
        if rng.random() < burst_prob:
            arrivals += burst_size
        
        backlog += arrivals
        
        if backlog > 0:
            tx = {"id": f"tx{k}", "t_submit_ms": clock.now_ms()}
            backlog -= 1
        else:
            # idle step: still advance small scheduler time
            clock.advance(2.0)
            continue
            
        proposer = cons.select_proposer(strategy="weighted")
        proposer.broadcast(tx, round_idx=0) if hasattr(proposer, "broadcast") else None

        with suppress_stdout():
            ok = cons.process_tx(tx) if hasattr(cons, "process_tx") else cons.confirm_block({"tx": tx})

        if ok:
            committed += 1
            latencies.append(clock.now_ms() - tx["t_submit_ms"])
            backlog = max(0, backlog - 1)
        else:
            backlog += 1

        # local processing cost (simulated time)
        # ---- stochastic cost model (REALISTIC) ----
        # coordination / voting
        clock.advance(sample_lognormal_ms(rng, 8.0, sigma=0.25))

        # verification / execution
        clock.advance(sample_lognormal_ms(rng, 12.0, sigma=0.35))

        # queueing delay (depends on backlog, always non-zero)
        base_q = 3.0 + 0.08 * backlog
        clock.advance(sample_lognormal_ms(rng, base_q, sigma=0.30, cap_mult=6.0))

        # ---- window tracking (AFTER one full step) ----
        t_wall_win.append(time.time())
        t_sim_win.append(clock.now_ms())
        comm_win.append(committed)

        # progress log
        if log_every > 0 and ((k + 1) % log_every == 0 or (k + 1) == n_txs):
            now = time.time()
            elapsed = now - t0
            done = k + 1
            pct = 100.0 * done / n_txs

            # ETA based on EWMA wall-clock rate (more realistic than done/elapsed)
            dt_wall = max(1e-6, now - last_log)
            inst_rate = log_every / dt_wall

            ewma_rate = inst_rate if ewma_rate is None else (
                alpha * inst_rate + (1 - alpha) * ewma_rate
            )

            eta = (n_txs - done) / ewma_rate if ewma_rate and ewma_rate > 0 else float("inf")


            # quick running stats (based on simulated time)
            sim_s = clock.now_ms() / 1000.0
            thr_now = 0.0
            if len(t_wall_win) >= 2:
                ds = (t_wall_win[-1] - t_wall_win[0])   # seconds (wall)
                dc = comm_win[-1] - comm_win[0]
                thr_now = (dc / ds) if ds > 0 else 0.0

            # avoid spamming logs too fast
            if now - last_log >= 0.2 or done == n_txs:
                print(
                    f"[loss={loss:.2f} seed={seed}] "
                    f"step {done}/{n_txs} ({pct:.1f}%) | "
                    f"wall {elapsed:.1f}s ETA {eta:.1f}s | "
                    f"sim {sim_s:.1f}s | committed {committed} | thr~{thr_now:.1f} tx/s | backlog {backlog}"
                )
                last_log = now

    sim_s = clock.now_ms() / 1000.0
    thr = committed / sim_s if sim_s > 0 else 0.0
    wall_s = time.time() - t0
    thr_wall = committed / wall_s if wall_s > 0 else 0.0
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    avg = float(np.mean(latencies)) if latencies else 0.0

    return dict(
        loss=loss, committed=committed,
        wall_s=wall_s, throughput_tps=thr_wall,
        sim_s=sim_s,
        lat_avg_ms=avg, lat_p50_ms=p50, lat_p95_ms=p95
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_sensitivity/failure_sensitivity.csv")
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--txs", type=int, default=5000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out))

    losses = [0.0, 0.05, 0.10, 0.20]
    rows = []
    for loss in losses:
        for r in range(args.repeats):
            seed = 1000 + r + int(loss*100)
            rows.append(run_one(loss, seed, args.nodes, args.txs, log_every=args.log_every))

    # write
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
