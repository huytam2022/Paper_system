# experiments/run_sensitivity_failure.py
from __future__ import annotations
import argparse, csv, os
import numpy as np
import time
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

def run_one(loss: float, seed: int, n_nodes: int, n_txs: int, log_every: int = 500):
    clock = SimClock()
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

    # submit a workload
    for k in range(n_txs):
        tx = {"id": f"tx{k}", "t_submit_ms": clock.now_ms()}

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
        clock.advance(2.0 + (seed % 3))   # gateway-ish
        clock.advance(4.0 + (seed % 5))   # verifier-ish
        clock.advance(min(50.0, backlog * 0.05))  # queueing delay

        # progress log
        if log_every > 0 and ((k + 1) % log_every == 0 or (k + 1) == n_txs):
            now = time.time()
            elapsed = now - t0
            done = k + 1
            pct = 100.0 * done / n_txs

            # ETA based on wall-clock speed
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (n_txs - done) / rate if rate > 0 else float("inf")

            # quick running stats (based on simulated time)
            sim_s = clock.now_ms() / 1000.0
            thr_now = committed / sim_s if sim_s > 0 else 0.0

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
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    avg = float(np.mean(latencies)) if latencies else 0.0

    return dict(
        loss=loss, committed=committed, sim_s=sim_s, throughput_tps=thr,
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
