# plots/plot_fig7_fig8.py
from __future__ import annotations
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_fig7(csv_path: str, out_path: str,
              col_nodes: str, col_thr_traffic: str, col_thr_energy: str, col_thr_env: str,
              col_eff: str):
    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df[col_nodes], df[col_thr_traffic], label="Traffic")
    plt.plot(df[col_nodes], df[col_thr_energy], label="Energy")
    plt.plot(df[col_nodes], df[col_thr_env], label="Environment")
    plt.xlabel("Number of nodes")
    plt.ylabel("Throughput (tx/s)")
    plt.title("Figure 7: Multi-domain stress throughput")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

    # dispatcher efficiency (optional separate plot if you prefer)
    plt.figure()
    plt.plot(df[col_nodes], df[col_eff])
    plt.xlabel("Number of nodes")
    plt.ylabel("Dispatcher efficiency (0-1)")
    plt.title("Figure 7 (right axis): Dispatcher efficiency")
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_eff.png"), dpi=200)


def plot_fig8a(csv_path: str, out_path: str,
               col_epoch: str, col_cpu: str, col_mem_mb: str):
    df = pd.read_csv(csv_path)
    plt.figure()
    plt.plot(df[col_epoch], df[col_cpu], label="CPU (%)")
    plt.plot(df[col_epoch], df[col_mem_mb], label="Memory (MB)")
    plt.xlabel("Epoch")
    plt.ylabel("Utilization / Memory")
    plt.title("Figure 8a: Gateway (Pi 4) resource profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_fig8b(csv_path: str, out_path: str,
               col_case: str, col_g16_prove_s: str, col_g16_verify_s: str,
               col_stark_prove_s: str, col_stark_verify_s: str):
    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df[col_case], df[col_g16_prove_s], label="Groth16 prove (s)")
    plt.plot(df[col_case], df[col_g16_verify_s], label="Groth16 verify (s)")
    plt.plot(df[col_case], df[col_stark_prove_s], label="STARK prove (s)")
    plt.plot(df[col_case], df[col_stark_verify_s], label="STARK verify (s)")
    plt.xlabel("Configuration / tx per block")
    plt.ylabel("Latency (s)")
    plt.title("Figure 8b: ZKP choice (Groth16 vs STARKs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="plots_out")
    ap.add_argument("--fig7_csv", default=None)
    ap.add_argument("--fig8a_csv", default=None)
    ap.add_argument("--fig8b_csv", default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # TODO: set these to match your CSV columns
    if args.fig7_csv:
        plot_fig7(
            args.fig7_csv, os.path.join(args.outdir, "fig7.png"),
            col_nodes="nodes",
            col_thr_traffic="thr_traffic_tps",
            col_thr_energy="thr_energy_tps",
            col_thr_env="thr_env_tps",
            col_eff="dispatcher_eff",
        )

    if args.fig8a_csv:
        plot_fig8a(
            args.fig8a_csv, os.path.join(args.outdir, "fig8a.png"),
            col_epoch="epoch",
            col_cpu="cpu_percent",
            col_mem_mb="mem_mb",
        )

    if args.fig8b_csv:
        plot_fig8b(
            args.fig8b_csv, os.path.join(args.outdir, "fig8b.png"),
            col_case="tx_per_block",
            col_g16_prove_s="groth16_prove_s",
            col_g16_verify_s="groth16_verify_s",
            col_stark_prove_s="stark_prove_s",
            col_stark_verify_s="stark_verify_s",
        )


if __name__ == "__main__":
    main()
