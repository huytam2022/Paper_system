import hashlib

def sha256(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()

def generate_zk_proof(tx_id: str, citizen_id: str, verified_flag: bool, merkle_root: str) -> str:
    """
    Sinh proof giả lập nếu hồ sơ tài chính đã xác minh hợp lệ.
    """
    if verified_flag:
        return sha256(citizen_id + tx_id + merkle_root)
    else:
        return "invalid_proof"

def verify_zk_proof(proof: str, citizen_id: str, tx_id: str, merkle_root: str) -> bool:
    """
    Xác minh proof ZK bằng cách khớp lại hash.
    """
    expected = sha256(citizen_id + tx_id + merkle_root)
    return proof == expected