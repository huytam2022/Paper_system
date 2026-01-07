# plots/plot_longterm.py
from __future__ import annotations
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analysis.metrics import gini, top_share, lorenz_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)  # reputation_snapshots.csv
    ap.add_argument("--outdir", type=str, default="plots_out")
    ap.add_argument("--every", type=int, default=50)  # sample every k rounds for curves
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    # metrics per round
    rows = []
    for r, g in df.groupby("round"):
        rep = g["reputation"].to_numpy(dtype=float)
        rows.append({
            "round": int(r),
            "gini": gini(rep),
            "top1": top_share(rep, 0.01),
            "top5": top_share(rep, 0.05),
        })
    m = pd.DataFrame(rows).sort_values("round")
    m.to_csv(os.path.join(args.outdir, "longterm_metrics.csv"), index=False)

    # plot gini over rounds
    plt.figure()
    plt.plot(m["round"], m["gini"])
    plt.xlabel("Round")
    plt.ylabel("Gini (reputation concentration)")
    plt.title("Long-term Reputation Concentration (Gini)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_longterm_gini.png"), dpi=200)

    # plot top-1% share over rounds
    plt.figure()
    plt.plot(m["round"], m["top1"])
    plt.xlabel("Round")
    plt.ylabel("Top-1% reputation share")
    plt.title("Top-1% Share over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_longterm_top1.png"), dpi=200)

    # Lorenz curves at sampled rounds
    sampled = m["round"].iloc[::max(1, args.every)].tolist()
    plt.figure()
    for r in sampled:
        rep = df[df["round"] == r]["reputation"].to_numpy(dtype=float)
        p, L = lorenz_curve(rep)
        plt.plot(p, L, label=f"t={r}")
    plt.plot([0, 1], [0, 1], linestyle="--")  # equality line
    plt.xlabel("Cumulative share of nodes")
    plt.ylabel("Cumulative share of reputation")
    plt.title("Lorenz Curves (sampled rounds)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_longterm_lorenz.png"), dpi=200)


if __name__ == "__main__":
    main()
