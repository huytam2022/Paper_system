import hashlib
from blockchains.Source_Chains import SourceChain
from zkp_verifier.zk_simulator import verify_zk_proof
import json

def verify_merkle_proof(tx_hash: str, proof: list, root: str) -> bool:
    current_hash = tx_hash
    for p in proof:
        if p["position"] == "left":
            current_hash = hashlib.sha256((p["hash"] + current_hash).encode()).hexdigest()
        else:
            current_hash = hashlib.sha256((current_hash + p["hash"]).encode()).hexdigest()
    return current_hash == root

class DestinationChain(SourceChain):
    def __init__(self, chain_id: str):
        super().__init__(chain_id)

    def generate_block(self):
        # Nếu không có pending_tx, vẫn tạo block rỗng để tránh None
        if not self.pending_tx:
            block = {
                "block_id": f"blk_{len(self.chain)}",
                "height": self.block_height,
                "merkle_root": None,
                "transactions": []
            }
            self.chain.append(block)
            self.block_height += 1
            return block
        return super().generate_block()


    def receive_ctx(self, tx, tx_hash_id: str, tx_hash: str, merkle_root_rc: str, proof_merkle: list, proof_zk: str) -> bool:
        """
        Nhận giao dịch đã relay từ RC, xác minh:
        1. tx_hash nằm trong Merkle tree của RC (proof_merkle)
        2. proof_zk hợp lệ với (tx_id, citizen_id, root)
        """
        # Parse transaction nếu là chuỗi
        if isinstance(tx, str):
            tx = json.loads(tx)

        is_merkle_valid = verify_merkle_proof(tx_hash, proof_merkle, merkle_root_rc)
        if not is_merkle_valid:
            print("❌ Merkle proof invalid")
            return False

        citizen_id = tx["payload"].get("citizen_id")
        is_zk_valid = verify_zk_proof(proof_zk, citizen_id, tx_hash_id, merkle_root_rc)
        if not is_zk_valid:
            print("❌ zk proof invalid")
            return False

        self.pending_tx.append(json.dumps(tx))  # giữ tx lại dạng string
        return True