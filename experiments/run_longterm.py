# experiments/run_longterm.py
from __future__ import annotations
import argparse
import csv
import os
from typing import List, Dict

from faults import FaultConfig, FaultInjector

# import objects from your repo
from blockchains.node import Node
from blockchains.Source_Chains import SourceChain
from consensus_layer import LightConsensus


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_snapshot_csv(path: str, round_idx: int, rep: Dict[str, float]):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["round", "node_id", "reputation"])
        for nid, r in rep.items():
            w.writerow([round_idx, nid, float(r)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=150)
    ap.add_argument("--rounds", type=int, default=1000)  # long-run
    ap.add_argument("--seed", type=int, default=42)

    # churn / packet loss
    ap.add_argument("--packet-loss", type=float, default=0.0)
    ap.add_argument("--churn-rate", type=float, default=0.0)
    ap.add_argument("--offline-rounds", type=int, default=3)

    # reputation smoothing/decay (eta) - you can wire into consensus if supported
    ap.add_argument("--eta", type=float, default=None)

    ap.add_argument("--outdir", type=str, default="results_longterm")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # build nodes
    nodes: List[Node] = []
    for i in range(args.nodes):
        n = Node(node_id=f"n{i}", chain_class=SourceChain, malicious=False, reputation=1.0)
        nodes.append(n)

    # connect peers (simple random-ish)
    for i, n in enumerate(nodes):
        # connect to next few nodes
        for j in range(1, 4):
            n.connect(nodes[(i + j) % len(nodes)])

    # consensus
    cons = LightConsensus(nodes=nodes)

    # optional: if your consensus has decay parameter, set it here
    if args.eta is not None and hasattr(cons, "set_decay"):
        cons.set_decay(args.eta)

    # attach consensus to node if needed by your code
    for n in nodes:
        n.consensus = cons

    # faults
    finj = FaultInjector(
        FaultConfig(
            packet_loss=args.packet_loss,
            churn_rate=args.churn_rate,
            offline_rounds=args.offline_rounds,
        ),
        seed=args.seed,
    )
    for n in nodes:
        n.attach_fault_injector(finj)

    snap_path = os.path.join(args.outdir, "reputation_snapshots.csv")

    # run long-term
    dropped_total = 0
    for t in range(args.rounds):
        # churn tick
        for n in nodes:
            n.tick_round(t)

        proposer = cons.select_proposer(strategy="weighted")

        # proposer makes a block (you can adapt to your chain/block format)
        blk = {
            "block_id": f"blk_{t}",
            "tx_count": 0,
            "is_valid": True,
            "proposer": proposer,
        }

        accepted = cons.confirm_block(block=blk)
        # snapshot
        rep = cons.get_reputation_snapshot()
        write_snapshot_csv(snap_path, t, rep)

        # (optional) track liveness
        if (t + 1) % 50 == 0:
            online = sum(1 for n in nodes if n.is_online(t))
            print(f"[t={t+1}] accepted={accepted} online={online}/{len(nodes)} dropped_total={dropped_total}")

    print("DONE:", snap_path)


if __name__ == "__main__":
    main()
