import json
from pathlib import Path

from src.blockchain import (
    hash_record,
    verify_simulated_chain,
    find_block_for_record,
    web3_enabled,
    verify_web3_record,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LAST_RUN_FILE = DATA_DIR / "last_run.json"


def main():
    if not LAST_RUN_FILE.exists():
        print(f"No run found at {LAST_RUN_FILE}. Run `python -m src.main` first.")
        return

    with open(LAST_RUN_FILE, "r") as f:
        last_run = json.load(f)

    record = last_run["record"]
    fresh_hash = hash_record(record)

    print("=" * 60)
    print("RE-VERIFICATION")
    print("=" * 60)
    print("Record being verified:")
    print(json.dumps(record, indent=2))

    print(f"\nFreshly computed hash: {fresh_hash}")

    print("\n-- Simulated chain --")
    chain_ok = verify_simulated_chain()
    print(f"Chain integrity (all links/hashes intact): "
          f"{'PASSED' if chain_ok else 'FAILED - chain has been tampered with'}")

    block = find_block_for_record(record)
    if block is None:
        print("No matching block found for this record. FAILED.")
    else:
        match = block["data_hash"] == fresh_hash
        print(f"Record hash matches on-chain block {block['index']}: "
              f"{'PASSED' if match else 'FAILED'}")

    web3_result = last_run.get("web3_result")
    if web3_result:
        print("\n-- Live chain (web3) --")
        if not web3_enabled():
            print("RPC_URL/PRIVATE_KEY not set in this environment - "
                  "cannot re-check the live chain from here.")
        else:
            tx_hash = web3_result["tx_hash"]
            ok = verify_web3_record(record, tx_hash)
            print(f"tx_hash: {tx_hash}")
            print(f"On-chain data matches freshly computed hash: "
                  f"{'PASSED' if ok else 'FAILED'}")
    else:
        print("\n(No live-chain upload was made for this run - "
              "simulated chain only.)")


if __name__ == "__main__":
    main()