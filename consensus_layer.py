# consensus_layer.py
import random
import time
from typing import List, Dict, Optional

class LightConsensus:
    """
    Đồng thuận nhẹ + voting (RQ1):
    - Chọn proposer: round_robin / random / weighted (reputation).
    - Voting với ngưỡng quorum (mặc định 2/3).
    - Cập nhật reputation mỗi vòng.
    """

    def __init__(self, nodes: list, confirm_delay: float = 0.05, quorum_ratio: float = 2/3):
        self.nodes: List = nodes
        self.confirm_delay = confirm_delay
        self.quorum_ratio = quorum_ratio
        self.current_proposer_idx = 0
        self.reputation: Dict[str, float] = {
            getattr(n, "id", f"node_{i}"): float(getattr(n, "reputation", 1.0))
            for i, n in enumerate(self.nodes)
        }

    # ---- proposer selection ----
    def select_proposer(self, strategy: str = "round_robin"):
        if not self.nodes:
            raise ValueError("No nodes registered in consensus.")

        if strategy == "random":
            return random.choice(self.nodes)

        if strategy == "weighted":
            weights = [max(0.0001, self.reputation[getattr(n, "id")]) for n in self.nodes]
            return random.choices(self.nodes, weights=weights, k=1)[0]

        # default round_robin
        proposer = self.nodes[self.current_proposer_idx]
        self.current_proposer_idx = (self.current_proposer_idx + 1) % len(self.nodes)
        return proposer

    # ---- block confirmation with voting ----
    def confirm_block(
        self,
        block: Optional[dict] = None,
        block_id: Optional[str] = None,
        tx_count: Optional[int] = None,
        is_valid: bool = True,
        proposer=None,
    ) -> bool:
        """
        Khuyến nghị dùng: confirm_block(block={"block_id","tx_count","is_valid","proposer"})
        Vẫn tương thích: confirm_block(block_id=..., tx_count=...) (coi hợp lệ).
        """
        if block is None:
            block = {
                "block_id": block_id if block_id is not None else f"blk_{int(time.time()*1000)}",
                "tx_count": int(tx_count) if tx_count is not None else 0,
                "is_valid": bool(is_valid),
                "proposer": proposer,
            }

        blk_id = block.get("block_id")
        blk_is_valid = bool(block.get("is_valid", True))
        blk_tx_count = int(block.get("tx_count", 0))
        blk_proposer = block.get("proposer", None)

        # voting
        votes = []
        for n in self.nodes:
            v = True
            if hasattr(n, "vote") and callable(getattr(n, "vote")):
                v = bool(n.vote(blk_is_valid))
            votes.append((n, v))

        approvals = sum(1 for _, v in votes if v)
        accepted = approvals >= int(self.quorum_ratio * len(self.nodes))

        time.sleep(self.confirm_delay)

        # update reputation: phiếu đúng +0.05, sai -0.10
        for n, v in votes:
            nid = getattr(n, "id")
            delta = 0.05 if (v == blk_is_valid) else -0.10
            self.reputation[nid] = max(0.0, self.reputation[nid] + delta)

        # đề xuất thưởng/phạt proposer
        if blk_proposer is not None:
            pid = getattr(blk_proposer, "id")
            if accepted and blk_is_valid:
                self.reputation[pid] = self.reputation[pid] + 0.10
            elif accepted and (not blk_is_valid):
                self.reputation[pid] = max(0.0, self.reputation[pid] - 0.20)
            elif (not accepted) and (not blk_is_valid):
                self.reputation[pid] = self.reputation[pid] + 0.02

        # sync vào node
        for n in self.nodes:
            nid = getattr(n, "id")
            if hasattr(n, "reputation"):
                n.reputation = self.reputation[nid]

        status = "ACCEPTED" if accepted else "REJECTED"
        validity = "VALID" if blk_is_valid else "INVALID"
        print(f"🧩 Block {blk_id} ({validity}) | tx={blk_tx_count} → {status} [{approvals}/{len(self.nodes)} approvals]")
        return accepted

    def get_reputation_snapshot(self) -> Dict[str, float]:
        return dict(self.reputation)
