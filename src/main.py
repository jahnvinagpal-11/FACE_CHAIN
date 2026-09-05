import json
import shutil
import time
from pathlib import Path

from src.face_engine import get_embedding_from_file, compare_faces
from src.web_search import search_image, find_social_matches
from src.download_image import download_image
from src.blockchain import (
    add_to_simulated_chain,
    verify_simulated_chain,
    web3_enabled,
    upload_to_web3,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_IMAGE = DATA_DIR / "input.jpg"
CANDIDATE_IMAGE = DATA_DIR / "candidate.jpg"
BEST_MATCH_IMAGE = DATA_DIR / "best_match.jpg"
LAST_RUN_FILE = DATA_DIR / "last_run.json"

MATCH_THRESHOLD = 0.35  # cosine similarity threshold for insightface embeddings


def run_pipeline():
    print("=" * 60)
    print("STEP 1: Face detection on input scan")
    print("=" * 60)

    input_embedding = get_embedding_from_file(str(INPUT_IMAGE))
    if input_embedding is None:
        print("No face detected in input image. Aborting.")
        return
    print(f"Face detected. Embedding size: {len(input_embedding)}")

    print("\n" + "=" * 60)
    print("STEP 2: Web/social media search")
    print("=" * 60)

    results = search_image(str(INPUT_IMAGE))
    matches = find_social_matches(results)

    if not matches:
        print("No social media matches found. Aborting.")
        return

    print(f"Found {len(matches)} candidate social media result(s).")

    best_match = None
    best_score = -1.0

    for i, candidate in enumerate(matches[:10], start=1):
        thumb = candidate.get("thumbnail")
        if not thumb:
            continue

        try:
            download_image(thumb, str(CANDIDATE_IMAGE))
            candidate_embedding = get_embedding_from_file(str(CANDIDATE_IMAGE))
            score = compare_faces(input_embedding, candidate_embedding)
        except Exception as e:
            print(f"  [{i}] skipped ({e})")
            continue

        if score is None:
            print(f"  [{i}] no face found in candidate image, skipping")
            continue

        print(f"  [{i}] {candidate.get('source')} - {candidate.get('title')!r} "
              f"-> similarity {score:.4f}")

        if score > best_score:
            best_score = score
            best_match = candidate
            # candidate.jpg gets overwritten every loop iteration, so copy
            # off the current winner now, before the next candidate
            # downloads over it.
            shutil.copyfile(CANDIDATE_IMAGE, BEST_MATCH_IMAGE)

    if best_match is None or best_score < MATCH_THRESHOLD:
        print(f"\nNo confident match found (best score: {best_score:.4f}, "
              f"threshold: {MATCH_THRESHOLD}). Aborting.")
        return

    print(f"\nBest match: {best_match.get('source')} "
          f"({best_score:.4f} similarity)")
    print(f"  Saved winning image to: {BEST_MATCH_IMAGE}  <- open this to visually confirm")
    print(f"  Title: {best_match.get('title')}")
    print(f"  Link:  {best_match.get('link')}")

    print("\n" + "=" * 60)
    print("STEP 3: Blockchain upload / verification")
    print("=" * 60)

    record = {
        "title": best_match.get("title"),
        "source": best_match.get("source"),
        "link": best_match.get("link"),
        "thumbnail": best_match.get("thumbnail"),
        "similarity_score": round(best_score, 6),
        "discovered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    block = add_to_simulated_chain(record)
    print(f"Written to simulated chain at block index {block['index']}")
    print(f"  data_hash:  {block['data_hash']}")
    print(f"  block_hash: {block['block_hash']}")
    print(f"Chain integrity check: {'PASSED' if verify_simulated_chain() else 'FAILED'}")

    web3_result = None
    if web3_enabled():
        print("\nRPC_URL/PRIVATE_KEY detected -> also uploading to live chain...")
        web3_result = upload_to_web3(record)
        print(f"  tx_hash:      {web3_result['tx_hash']}")
        print(f"  block_number: {web3_result['block_number']}")
        print(f"  chain_id:     {web3_result['chain_id']}")
    else:
        print("\n(RPC_URL/PRIVATE_KEY not set - skipping real-chain upload, "
              "using simulated chain only. See README for how to enable it.)")

    last_run = {
        "record": record,
        "simulated_block": block,
        "web3_result": web3_result,
    }
    with open(LAST_RUN_FILE, "w") as f:
        json.dump(last_run, f, indent=2)

    print(f"\nSaved run summary to {LAST_RUN_FILE}")
    print("Done. Run `python -m src.verify` to independently re-verify this record.")


if __name__ == "__main__":
    run_pipeline()