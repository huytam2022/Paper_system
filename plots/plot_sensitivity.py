# plots/plot_sensitivity.py
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="plots_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    g = df.groupby("loss").agg({
        "throughput_tps": ["mean","std"],
        "lat_p95_ms": ["mean","std"]
    }).reset_index()
    g.columns = ["loss","thr_mean","thr_std","p95_mean","p95_std"]

    # Throughput vs loss
    plt.figure()
    plt.errorbar(g["loss"]*100, g["thr_mean"], yerr=g["thr_std"], fmt="-o")
    plt.xlabel("Packet loss (%)")
    plt.ylabel("Throughput (tx/s)")
    plt.title("Sensitivity to Network Failure: Throughput")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "sens_throughput.png"), dpi=200)

    # Latency p95 vs loss
    plt.figure()
    plt.errorbar(g["loss"]*100, g["p95_mean"], yerr=g["p95_std"], fmt="-o")
    plt.xlabel("Packet loss (%)")
    plt.ylabel("Latency p95 (ms)")
    plt.title("Sensitivity to Network Failure: Latency (p95)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "sens_latency_p95.png"), dpi=200)

if __name__ == "__main__":
    main()
