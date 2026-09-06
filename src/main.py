"""
FACE_CHAIN end-to-end pipeline:

    face scan (data/input.jpg)
        -> reverse image / web search (SerpAPI + Google Lens)
        -> pick best matching social media post
        -> download the matched image, confirm it's the same face
        -> hash the discovered post's data and record it on-chain
           (simulated local chain, plus a real testnet/local node if
           RPC_URL + PRIVATE_KEY are set in .env)

Run with:  python -m src.main
(run from the repo root so the `src.` imports resolve)
"""

import json
import shutil
import time
from pathlib import Path

from PIL import Image
import imagehash

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

SUPPORTED_INPUT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# Filenames the pipeline generates itself -- never pick these as the input,
# even if they match a supported extension.
GENERATED_FILENAMES = {"candidate.jpg", "best_match.jpg"}


def find_input_image() -> Path:
    """
    Find the input photo in data/.

    Prefers a file explicitly named input.<ext>. If none exists, falls
    back to the most recently modified supported image file in data/
    (excluding files the pipeline itself generates), so you can just drop
    any photo in there without renaming it.
    """
    for ext in SUPPORTED_INPUT_EXTENSIONS:
        candidate = DATA_DIR / f"input{ext}"
        if candidate.exists():
            return candidate

    candidates = [
        p for p in DATA_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
        and p.name not in GENERATED_FILENAMES
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No input image found in {DATA_DIR}. Add a photo named "
            f"input.jpg (or .png/.webp/.jpeg), or any image file there."
        )

    chosen = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"(No file named 'input.<ext>' found -- using most recently "
          f"modified image instead: {chosen.name})")
    return chosen


INPUT_IMAGE = find_input_image()
CANDIDATE_IMAGE = DATA_DIR / "candidate.jpg"
BEST_MATCH_IMAGE = DATA_DIR / "best_match.jpg"
LAST_RUN_FILE = DATA_DIR / "last_run.json"
DEBUG_DIR = DATA_DIR / "debug_candidates"

MATCH_THRESHOLD = 0.55  # cosine similarity threshold for insightface embeddings
                         # (true matches in testing scored 0.85-0.88; raised
                         # from 0.35 after seeing false-accepts on non-face
                         # candidate images)

# Perceptual-hash near-duplicate detection. A pHash distance of 0 means
# pixel-identical; typical "same photo, different compression/crop"
# reposts land under ~10-12. This catches cases where the exact same
# photo was reposted but face-embedding similarity alone is noisy (e.g.
# due to a low-res thumbnail), giving a second, independent signal that
# it's a genuine repost of the same image.
PHASH_NEAR_DUPLICATE_THRESHOLD = 12
PHASH_CONFIDENCE_SCORE = 0.95  # score assigned when a near-duplicate is found


def phash_distance(path_a, path_b):
    """Perceptual-hash Hamming distance between two images (0 = identical)."""
    hash_a = imagehash.phash(Image.open(path_a).convert("RGB"))
    hash_b = imagehash.phash(Image.open(path_b).convert("RGB"))
    return hash_a - hash_b


def run_pipeline():
    print("=" * 60)
    print("STEP 1: Face detection on input scan")
    print("=" * 60)
    print(f"Using input image: {INPUT_IMAGE}")

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
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    best_match = None
    best_score = -1.0

    for i, candidate in enumerate(matches[:20], start=1):
        # Try the full-resolution "image" URL first (better quality than
        # the tiny compressed "thumbnail"), but some sources (notably
        # TikTok's CDN) block direct downloads of their "image" URL or
        # return non-image content. Fall back to "thumbnail" in that case
        # instead of skipping the candidate outright.
        candidates_to_try = [
            url for url in (candidate.get("image"), candidate.get("thumbnail"))
            if url
        ]

        if not candidates_to_try:
            continue

        score = None
        near_duplicate = False
        last_error = None

        for image_url in candidates_to_try:
            try:
                download_image(image_url, str(CANDIDATE_IMAGE))
                # Save a permanent copy for later visual inspection --
                # data/debug_candidates/01_instagram.jpg etc. -- since
                # CANDIDATE_IMAGE itself gets overwritten every iteration.
                safe_source = "".join(
                    c if c.isalnum() else "_" for c in str(candidate.get("source", "unknown"))
                )
                debug_path = DEBUG_DIR / f"{i:02d}_{safe_source}.jpg"
                shutil.copyfile(CANDIDATE_IMAGE, debug_path)

                candidate_embedding = get_embedding_from_file(str(CANDIDATE_IMAGE))
                face_score = compare_faces(input_embedding, candidate_embedding)

                # Second, independent signal: is this the *same photo*,
                # pixel-wise, regardless of whether face detection worked
                # well on a low-quality copy of it? Catches reposts that
                # face-embedding similarity alone scores unreliably.
                try:
                    distance = phash_distance(INPUT_IMAGE, CANDIDATE_IMAGE)
                except Exception:
                    distance = None

                if distance is not None and distance <= PHASH_NEAR_DUPLICATE_THRESHOLD:
                    near_duplicate = True
                    score = max(face_score or 0.0, PHASH_CONFIDENCE_SCORE)
                else:
                    score = face_score

                if score is not None:
                    break  # got a usable signal, no need to try the fallback URL
            except Exception as e:
                last_error = e
                continue

        if score is None:
            reason = f"({last_error})" if last_error else "(no face found in candidate image)"
            print(f"  [{i}] skipped {reason}")
            continue

        tag = "  [near-duplicate image detected]" if near_duplicate else ""
        print(f"  [{i}] {candidate.get('source')} - {candidate.get('title')!r} "
              f"-> similarity {score:.4f}{tag}  (see data/debug_candidates/{i:02d}_{safe_source}.jpg)")

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