# blockchains/node.py

class Node:
    def __init__(self, node_id: str, chain_class, consensus=None, malicious: bool = False, reputation: float = 1.0):
        self.id = node_id
        self.chain = chain_class(chain_id=node_id)
        self.peers = []
        self.consensus = consensus
        # RQ1 fields
        self.malicious = bool(malicious)
        self.reputation = float(reputation)

    def connect(self, peer_node):
        if peer_node not in self.peers:
            self.peers.append(peer_node)

    def broadcast_tx(self, payload, tx_receiver, tx_id):
        tx_str = self.chain.add_transaction(payload, tx_receiver, tx_id)
        for peer in self.peers:
            peer.receive_tx(tx_str, self)

    def receive_tx(self, tx_str, from_node):
        # đơn giản: thêm vào pending
        self.chain.pending_tx.append(tx_str)

    def generate_and_send_block(self):
        block = self.chain.generate_block()
        return block

    # ===== RQ1: voting behavior =====
    def vote(self, is_block_valid: bool) -> bool:
        """
        Honest: chỉ chấp thuận khối hợp lệ.
        Malicious: luôn chấp thuận (collusion) kể cả khối sai.
        """
        if self.malicious:
            return True
        return bool(is_block_valid)

    # blockchains/node.py (ADD inside class Node)

    def attach_fault_injector(self, injector):
        """
        injector: FaultInjector or None
        """
        self._fault = injector

    def is_online(self, round_idx: int) -> bool:
        if getattr(self, "_fault", None) is None:
            return True
        return self._fault.is_online(self.id, round_idx)

    def tick_round(self, round_idx: int):
        """
        Call at round boundary.
        """
        if getattr(self, "_fault", None) is None:
            return
        self._fault.maybe_churn(self.id, round_idx)

    def send_to_peer(self, peer, msg: dict, round_idx: int) -> bool:
        """
        Returns True if delivered, False if dropped/offline.
        """
        # if sender or receiver offline -> drop (connectivity)
        if not self.is_online(round_idx):
            return False
        if hasattr(peer, "is_online") and not peer.is_online(round_idx):
            return False

        # packet loss
        if getattr(self, "_fault", None) is not None and self._fault.should_drop():
            return False

        # "deliver" (in this simulator we just call a method if exists)
        if hasattr(peer, "on_message"):
            peer.on_message(msg)
        return True

    def broadcast(self, msg: dict, round_idx: int) -> int:
        delivered = 0
        for p in getattr(self, "peers", []):
            if self.send_to_peer(p, msg, round_idx):
                delivered += 1
        return delivered