# agent_light_client.py
# Simulates a lightweight IoT verifier node using Flask

import time, json, hashlib
from flask import Flask, request, jsonify
import psutil

app = Flask(__name__)
DEVICE_NAME = "default_device"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "cpu_percent": psutil.cpu_percent(0.2)})

@app.route("/run_task", methods=["POST"])
def run_task():
    req = request.get_json(force=True)
    num_proofs = int(req.get("num_proofs", 50))
    payload_size = int(req.get("payload_size", 1024))
    merkle_depth = int(req.get("merkle_depth", 8))

    cpu_before = psutil.cpu_percent(interval=0.4)
    mem_before = psutil.virtual_memory().used / (1024 * 1024)
    net_before = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv

    t0 = time.perf_counter()
    zk_times = []
    merkle_times = []

    leaf = b"L" * 32
    proof_list = [hashlib.sha256(f"p{i}".encode()).digest() for i in range(merkle_depth)]
    root = hashlib.sha256(leaf + b"".join(proof_list)).digest()

    for _ in range(num_proofs):
        data = (b"Z" * payload_size)

        # Simulate ZK proof verification
        z0 = time.perf_counter()
        hashlib.sha256(data).digest()
        z1 = time.perf_counter()
        zk_times.append((z1 - z0) * 1000.0)

        # Simulate Merkle proof verification
        m0 = time.perf_counter()
        h = hashlib.sha256(leaf).digest()
        for p in proof_list:
            h = hashlib.sha256(h + p).digest()
        m1 = time.perf_counter()
        merkle_times.append((m1 - m0) * 1000.0)

    t1 = time.perf_counter()
    cpu_after = psutil.cpu_percent(interval=0.4)
    mem_after = psutil.virtual_memory().used / (1024 * 1024)
    net_after = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv

    kbps = ((net_after - net_before) * 8.0 / 1000.0) / max(1e-6, t1 - t0)

    return jsonify({
        "device": DEVICE_NAME,
        "cpu_percent_before": cpu_before,
        "cpu_percent_after": cpu_after,
        "mem_used_mb_before": round(mem_before, 2),
        "mem_used_mb_after": round(mem_after, 2),
        "net_kbps": round(kbps, 2),
        "zk_verify_ms_avg": round(sum(zk_times) / len(zk_times), 3),
        "merkle_verify_ms_avg": round(sum(merkle_times) / len(merkle_times), 3),
        "elapsed_s": round(t1 - t0, 3)
    })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device-name", default="sim_device")
    args = parser.parse_args()

    DEVICE_NAME = args.device_name
    print(f"[Agent] Starting {DEVICE_NAME} on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
