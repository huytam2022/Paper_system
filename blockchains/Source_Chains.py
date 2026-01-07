import hashlib
import json
from typing import List, Dict, Optional


def sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def build_merkle_tree(transactions: List[str]) -> List[List[str]]:
    if not transactions:
        return []

    layer = [sha256(tx) for tx in transactions]
    tree = [layer.copy()]

    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i+1] if i+1 < len(layer) else left
            combined = sha256(left + right)
            next_layer.append(combined)
        tree.insert(0, next_layer)
        layer = next_layer

    return tree


def get_merkle_proof(tree: List[List[str]], index: int) -> List[Dict[str, str]]:
    proof = []
    depth = len(tree)
    for level in range(depth - 1, 0, -1):
        layer = tree[level]
        is_right_node = index % 2
        sibling_index = index - 1 if is_right_node else index + 1
        if sibling_index >= len(layer):
            sibling_hash = layer[index]
        else:
            sibling_hash = layer[sibling_index]
        proof.append({
            "position": "left" if is_right_node else "right",
            "hash": sibling_hash
        })
        index //= 2
    return proof


class SourceChain:
    def __init__(self, chain_id: str):
        self.chain_id: str = chain_id
        self.chain: List[Dict] = []
        self.pending_tx: List[str] = []
        self.block_height = 0
        self.last_tree: Optional[List[List[str]]] = None

    def add_transaction(self, payload: Dict, tx_receiver: str, tx_id: str) -> str:
        tx = {
            "tx_id": tx_id,
            "chain_id": self.chain_id,
            "receiver": tx_receiver,
            "payload": payload
        }
        tx_str = json.dumps(tx, sort_keys=True)
        self.pending_tx.append(tx_str)
        return tx_str

    def generate_block(self) -> Optional[Dict]:
        if not self.pending_tx:
            return None

        self.last_tree = build_merkle_tree(self.pending_tx)
        root = self.get_merkle_root()
        block = {
            "block_id": f"blk_{len(self.chain)}",
            "height": self.block_height,
            "merkle_root": root,
            "transactions": self.pending_tx.copy()
        }
        self.chain.append(block)
        self.block_height += 1
        self.pending_tx.clear()
        return block

    def get_latest_block(self) -> Optional[Dict]:
        return self.chain[-1] if self.chain else None

    def get_merkle_root(self) -> Optional[str]:
        return self.last_tree[0][0] if self.last_tree else None

    def get_merkle_proof(self, tx: str) -> Optional[List[Dict[str, str]]]:
        block = self.get_latest_block()
        if not block or tx not in block["transactions"]:
            return None

        index = block["transactions"].index(tx)
        tree = build_merkle_tree(block["transactions"])
        return get_merkle_proof(tree, index)

    def get_block_header(self) -> Optional[Dict[str, object]]:
        block = self.get_latest_block()
        if block:
            return {
                "height": block["height"],
                "merkle_root": block["merkle_root"]
            }
        return None