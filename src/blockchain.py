import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHAIN_FILE = DATA_DIR / "chain.json"

RPC_URL = os.getenv("RPC_URL", "").strip()
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "").strip()


def hash_record(record: dict) -> str:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_chain() -> list:
    if not CHAIN_FILE.exists():
        return []
    with open(CHAIN_FILE, "r") as f:
        return json.load(f)


def _save_chain(chain: list) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHAIN_FILE, "w") as f:
        json.dump(chain, f, indent=2)


def _block_hash(index: int, timestamp: float, data_hash: str, prev_hash: str) -> str:
    payload = f"{index}{timestamp}{data_hash}{prev_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def add_to_simulated_chain(record: dict) -> dict:
    chain = _load_chain()

    index = len(chain)
    timestamp = time.time()
    data_hash = hash_record(record)
    prev_hash = chain[-1]["block_hash"] if chain else "0" * 64

    block = {
        "index": index,
        "timestamp": timestamp,
        "data_hash": data_hash,
        "prev_hash": prev_hash,
        "record": record,
    }
    block["block_hash"] = _block_hash(index, timestamp, data_hash, prev_hash)

    chain.append(block)
    _save_chain(chain)

    return block


def verify_simulated_chain() -> bool:
    """Walk the whole chain and confirm every link and hash is intact."""
    chain = _load_chain()

    prev_hash = "0" * 64

    for block in chain:
        expected_data_hash = hash_record(block["record"])
        if block["data_hash"] != expected_data_hash:
            return False

        if block["prev_hash"] != prev_hash:
            return False

        expected_block_hash = _block_hash(
            block["index"], block["timestamp"], block["data_hash"], block["prev_hash"]
        )
        if block["block_hash"] != expected_block_hash:
            return False

        prev_hash = block["block_hash"]

    return True


def find_block_for_record(record: dict) -> dict | None:
    """Look up the on-chain block matching a given record's hash."""
    target_hash = hash_record(record)
    chain = _load_chain()

    for block in chain:
        if block["data_hash"] == target_hash:
            return block

    return None

def web3_enabled() -> bool:
    return bool(RPC_URL and PRIVATE_KEY)


def upload_to_web3(record: dict) -> dict | None:

    if not web3_enabled():
        return None

    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        raise ConnectionError(f"Could not connect to RPC_URL: {RPC_URL}")

    account = w3.eth.account.from_key(PRIVATE_KEY)
    data_hash = hash_record(record)

    tx = {
        "from": account.address,
        "to": account.address,  # self-send; the hash lives in tx input data
        "value": 0,
        "data": w3.to_bytes(hexstr=data_hash if data_hash.startswith("0x") else "0x" + data_hash),
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 60000,
        "gasPrice": w3.eth.gas_price,
        "chainId": w3.eth.chain_id,
    }

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    return {
        "tx_hash": tx_hash.hex(),
        "block_number": receipt["blockNumber"],
        "data_hash": data_hash,
        "chain_id": w3.eth.chain_id,
    }


def verify_web3_record(record: dict, tx_hash: str) -> bool:
    if not web3_enabled():
        return False

    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    tx = w3.eth.get_transaction(tx_hash)

    onchain_data = tx["input"]
    if isinstance(onchain_data, (bytes, bytearray)):
        onchain_hex = onchain_data.hex().lower().lstrip("0x")
    else:
        onchain_hex = onchain_data.lower().lstrip("0x")

    return onchain_hex == hash_record(record).lower()


if __name__ == "__main__":
    demo_record = {"title": "demo record", "source": "test", "url": "https://example.com"}
    block = add_to_simulated_chain(demo_record)
    print("Added block:", json.dumps(block, indent=2))
    print("Chain valid:", verify_simulated_chain())