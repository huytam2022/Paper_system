import argparse
import time
import random
from blockchains.node import Node
from blockchains.Source_Chains import SourceChain
from consensus_layer import LightConsensus
import csv

def _random_partition_sizes(n, k):
    """
    Trả về list độ dài k sao cho tổng = n và mỗi phần >= 1.
    Ví dụ: n=10, k=3 -> [3, 2, 5]
    """
    if k == 1:
        return [n]
    # chọn k-1 điểm cắt trong (1..n-1)
    cuts = sorted(random.sample(range(1, n), k - 1))
    sizes = [a - b for a, b in zip(cuts + [n], [0] + cuts)]
    return sizes


def simulate_partitioned_blockchain(num_nodes, num_partitions, tx_count_per_node, max_rounds):
    assert num_nodes >= num_partitions, "Số node phải lớn hơn hoặc bằng số phân vùng"

    # Khởi tạo node
    nodes = [Node(f"node_{i}", SourceChain) for i in range(num_nodes)]
    print(f"✅ Tổng số node: {len(nodes)}")

    # Chia ngẫu nhiên nodes cho các phân vùng
    random.shuffle(nodes)
    sizes = _random_partition_sizes(num_nodes, num_partitions)
    partitions = []
    offset = 0
    for sz in sizes:
        partitions.append(nodes[offset:offset + sz])
        offset += sz

    # In cấu hình phân vùng (ngẫu nhiên)
    for idx, partition in enumerate(partitions):
        print(f" - Phân vùng {idx+1}: {len(partition)} node")

    # === CSV tracking (per-partition) ===
    rounds_per_partition = []
    accepted_blocks_per_partition = []
    dropped_txs_per_partition = []
    liveness_per_partition = []

    all_results = []
    total_dropped_txs = 0
    partition_liveness = []
    fork_resolution_time = random.uniform(0.5, 2.0)  # giả lập 0.5–2s

    for idx, partition in enumerate(partitions):
        print(f"\n🚀 Khởi chạy phân vùng {idx+1}...")

        # Kết nối peer nội bộ phân vùng (mỗi node kết nối tối đa 3 peer ngẫu nhiên trong vùng)
        for local_idx, node in enumerate(partition):
            node.name = f"node_{idx}_{local_idx}"
            peers = [n for n in partition if n != node]
            if peers:
                for p in random.sample(peers, min(len(peers), 3)):
                    node.connect(p)

        # Sinh giao dịch trong vùng
        txs = []
        for node in partition:
            for j in range(tx_count_per_node):
                tx_id = f"tx_{node.name}_{j}"
                tx_str = node.chain.add_transaction({"msg": "data"}, "type", tx_id)
                txs.append(tx_str)

        # Phát tán giao dịch trong vùng
        for node in partition:
            for peer in node.peers:
                for tx in txs:
                    peer.receive_tx(tx, node)

        # Đồng thuận nội bộ phân vùng
        consensus = LightConsensus(partition, confirm_delay = random.uniform(0.6, 1.2), quorum_ratio=2/3)
        accepted_blocks = 0
        dropped_txs = 0

        # random số rounds cho phân vùng này (1..max_rounds)
        rounds = random.randint(1, max_rounds)
        print(f" ⏱️ Phân vùng {idx+1} sẽ chạy {rounds} vòng đồng thuận.")

        for r in range(rounds):
            # 1) Chọn proposer an toàn
            proposer = consensus.select_proposer()
            if proposer is None:
                # Không có proposer hợp lệ trong vòng này -> bỏ qua
                continue

            # 2) Thử tạo block
            blk = proposer.chain.generate_block()

            # 3) Nếu không có block (mempool rỗng / chưa đạt ngưỡng), bơm 1 dummy tx rồi thử lại
            if not blk:
                tx_id = f"dummy_p{idx}_r{r}_{time.time_ns()}"
                proposer.chain.add_transaction({"msg": "filler"}, "type", tx_id)
                blk = proposer.chain.generate_block()

            # 4) Nếu vẫn không có block -> bỏ qua vòng này
            if not blk:
                continue

            # 5) Gán metadata cho block
            blk["block_id"] = f"p{idx}_r{r}"
            blk["tx_count"] = len(blk.get("transactions", []))
            blk["is_valid"] = True
            blk["proposer"] = proposer

            # 6) Xác suất fork/xung đột nhẹ
            if random.random() < random.uniform(0.15, 0.3):
                dropped_txs += random.randint(2, 6)
                continue

            # 7) Xác nhận block
            if consensus.confirm_block(blk):
                accepted_blocks += 1

        print(f"✅ Phân vùng {idx+1} đã xác nhận {accepted_blocks}/{rounds} khối.")
        partition_liveness.append(accepted_blocks / rounds)
        total_dropped_txs += dropped_txs
        all_results.append((idx+1, accepted_blocks))

        # === Ghi lại để xuất CSV ===
        rounds_per_partition.append(rounds)
        accepted_blocks_per_partition.append(accepted_blocks)
        dropped_txs_per_partition.append(dropped_txs)
        liveness_per_partition.append(accepted_blocks / rounds if rounds > 0 else 0.0)

    # Hợp nhất sau khi các phân vùng đã chạy xong
    print("\n🔗 Bắt đầu hợp nhất các phân vùng...")
    time.sleep(fork_resolution_time)  # mô phỏng thời gian giải quyết fork

    for partition_id, block_count in all_results:
        print(f" - Phân vùng {partition_id}: {block_count} khối đã được xác nhận.")

    print("\n📊 Tổng hợp sau hợp nhất:")
    print(f"⏱️ Thời gian giải quyết fork: {fork_resolution_time:.2f} giây")
    print(f"❌ Tổng số giao dịch bị loại trong fork hoặc xung đột: {total_dropped_txs}")
    print(f"✅ Tính sống trung bình của phân vùng: {sum(partition_liveness)/len(partition_liveness)*100:.1f}%")

    # Đồng thuận toàn cục sau hợp nhất
    print("\n🕒 Đo thời gian đồng thuận toàn mạng sau hợp nhất...")
    
    global_consensus = LightConsensus(nodes, confirm_delay = random.uniform(0.6, 1.2), quorum_ratio=2/3)
    global_proposer = global_consensus.select_proposer()

    # Tạo 1 giao dịch giả để đảm bảo có block
    tx_id = f"global_tx_{time.time_ns()}"
    _ = global_proposer.chain.add_transaction({"msg": "global merge"}, "type", tx_id)

    start_time = time.perf_counter()
    global_block = global_proposer.chain.generate_block()
    global_block["block_id"] = "final_merge_block"
    global_block["tx_count"] = len(global_block["transactions"])
    global_block["is_valid"] = True
    global_block["proposer"] = global_proposer
    global_consensus.confirm_block(global_block)
    end_time = time.perf_counter()

    merge_consensus_time = end_time - start_time
    print(f"⏳ Đồng thuận toàn cục đạt được sau {merge_consensus_time:.4f} giây.")

    # === Save results to CSV ===
    csv_filename = "RQ1.2.csv"
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Header (English)
        writer.writerow([
            "partition_id",
            "nodes_in_partition",
            "rounds_run",
            "accepted_blocks",
            "dropped_txs",
            "liveness_ratio",
            "fork_resolution_time",
            "merge_consensus_time"
        ])
        # Rows per partition
        for idx, partition in enumerate(partitions):
            writer.writerow([
                idx + 1,
                len(partition),
                rounds_per_partition[idx],
                accepted_blocks_per_partition[idx],
                dropped_txs_per_partition[idx],
                f"{liveness_per_partition[idx]*100:.2f}%",
                f"{fork_resolution_time:.2f}",
                f"{merge_consensus_time:.4f}"
            ])

    print(f"\n💾 Results saved to {csv_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate partitioned blockchain with random-sized partitions and random rounds.")
    parser.add_argument("--nodes", type=int, default=100, help="Số lượng node trong toàn mạng")
    parser.add_argument("--partitions", type=int, default=2, help="Số phân vùng mạng không giao tiếp")
    parser.add_argument("--tx-per-node", type=int, default=5, help="Số lượng giao dịch mỗi node tạo ra")
    parser.add_argument("--rounds", type=int, default=10, help="Số vòng đồng thuận tối đa cho mỗi phân vùng (thực tế sẽ random 1..rounds)")
    args = parser.parse_args()

    simulate_partitioned_blockchain(
        num_nodes=args.nodes,
        num_partitions=args.partitions,
        tx_count_per_node=args.tx_per_node,
        max_rounds=args.rounds
    )
