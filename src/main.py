"""
FACE_CHAIN end-to-end pipeline:

    input image
        ->
    Google Lens / reverse image search
        ->
    social-media candidates
        ->
    face similarity
    + perceptual hash
    + SIFT feature matching
    + aspect-ratio comparison
        ->
    select the closest ORIGINAL/UNALTERED image candidate
        ->
    blockchain record

Run from repo root:

    python -m src.main
"""

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import imagehash

from src.face_engine import (
    get_embedding_from_file,
    compare_faces,
)

from src.web_search import (
    search_image,
    find_social_matches,
)

from src.download_image import download_image

from src.blockchain import (
    add_to_simulated_chain,
    verify_simulated_chain,
    web3_enabled,
    upload_to_web3,
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SUPPORTED_INPUT_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
)

GENERATED_FILENAMES = {
    "candidate.jpg",
    "best_match.jpg",
}


CANDIDATE_IMAGE = DATA_DIR / "candidate.jpg"

BEST_MATCH_IMAGE = DATA_DIR / "best_match.jpg"

LAST_RUN_FILE = DATA_DIR / "last_run.json"

DEBUG_DIR = DATA_DIR / "debug_candidates"


# ============================================================
# THRESHOLDS
# ============================================================

# Minimum face similarity required
MATCH_THRESHOLD = 0.55


# pHash:
#
# 0 = identical
# lower = more visually similar
#
PHASH_NEAR_DUPLICATE_THRESHOLD = 12


# ============================================================
# INPUT IMAGE
# ============================================================

def find_input_image() -> Path:

    """
    Find the user's input image.

    Priority:

        input.jpg
        input.jpeg
        input.png
        input.webp

    Otherwise use the most recently modified image.
    """

    for ext in SUPPORTED_INPUT_EXTENSIONS:

        candidate = DATA_DIR / f"input{ext}"

        if candidate.exists():

            return candidate


    candidates = [

        p

        for p in DATA_DIR.iterdir()

        if (
            p.is_file()
            and p.suffix.lower()
            in SUPPORTED_INPUT_EXTENSIONS
            and p.name not in GENERATED_FILENAMES
        )

    ]


    if not candidates:

        raise FileNotFoundError(

            f"No input image found in {DATA_DIR}."

        )


    chosen = max(

        candidates,

        key=lambda p: p.stat().st_mtime

    )


    print(

        "(No input.<ext> found -- "

        f"using most recently modified image: "

        f"{chosen.name})"

    )


    return chosen


INPUT_IMAGE = find_input_image()


# ============================================================
# pHASH
# ============================================================

def phash_distance(path_a, path_b):

    """
    Perceptual hash distance.

    Lower = more visually similar.

    0 = identical / extremely close.
    """

    hash_a = imagehash.phash(

        Image.open(path_a).convert("RGB")

    )

    hash_b = imagehash.phash(

        Image.open(path_b).convert("RGB")

    )

    return hash_a - hash_b


def phash_score(distance):

    """
    Convert pHash distance into 0-1 score.

    Lower distance = higher score.
    """

    if distance is None:

        return 0.0


    score = 1.0 - (distance / 32.0)

    return max(
        0.0,
        min(1.0, score)
    )


# ============================================================
# ASPECT RATIO
# ============================================================

def aspect_ratio_score(path_a, path_b):

    """
    Compare image aspect ratios.

    Same aspect ratio = 1.0

    Very different aspect ratio = lower score.

    This helps penalize reposts that have been turned into
    a different format with text added around the image.
    """

    try:

        with Image.open(path_a) as img_a:

            width_a, height_a = img_a.size


        with Image.open(path_b) as img_b:

            width_b, height_b = img_b.size


        ratio_a = width_a / height_a

        ratio_b = width_b / height_b


        difference = abs(
            ratio_a - ratio_b
        )


        # 0 difference -> 1.0
        # larger difference -> lower
        score = 1.0 - min(
            difference,
            1.0
        )


        return score


    except Exception:

        return 0.0


# ============================================================
# SIFT FEATURE MATCHING
# ============================================================

def sift_similarity(path_a, path_b):

    """
    Compare local visual features between two images.

    This is useful when:

        - image is resized
        - image is slightly cropped
        - compression changed
        - text was added around the image

    Returns a score between 0 and 1.
    """

    try:

        img_a = cv2.imread(
            str(path_a),
            cv2.IMREAD_GRAYSCALE
        )

        img_b = cv2.imread(
            str(path_b),
            cv2.IMREAD_GRAYSCALE
        )


        if img_a is None or img_b is None:

            return 0.0


        sift = cv2.SIFT_create(
            nfeatures=1000
        )


        keypoints_a, descriptors_a = sift.detectAndCompute(
            img_a,
            None
        )


        keypoints_b, descriptors_b = sift.detectAndCompute(
            img_b,
            None
        )


        if descriptors_a is None or descriptors_b is None:

            return 0.0


        if len(descriptors_a) < 2 or len(descriptors_b) < 2:

            return 0.0


        matcher = cv2.BFMatcher(
            cv2.NORM_L2
        )


        matches = matcher.knnMatch(

            descriptors_a,

            descriptors_b,

            k=2

        )


        good_matches = []


        for pair in matches:

            if len(pair) != 2:

                continue


            m, n = pair


            # Lowe's ratio test
            if m.distance < 0.70 * n.distance:

                good_matches.append(m)


        # Normalize number of good matches.
        #
        # 40+ good matches is considered extremely strong.
        #

        score = min(
            len(good_matches) / 40.0,
            1.0
        )


        return float(score)


    except Exception:

        return 0.0


# ============================================================
# IMAGE ALTERATION / OVERLAY HEURISTIC
# ============================================================

def overlay_penalty(path):

    """
    Look for suspicious added content near the edges/corners
    of an image.

    This is intentionally a soft penalty, NOT a hard decision.

    It helps with images such as:

        original photo

        vs

        original photo + fanpage caption/watermark

    Small corner watermarks (e.g. "SITE.COM" in a corner) get
    diluted to near-zero if you only average edge density over
    a full-width strip, so we also check each corner in
    isolation -- that's where logos/handles/watermarks usually
    sit.
    """

    try:

        img = cv2.imread(
            str(path)
        )


        if img is None:

            return 0.0


        height, width = img.shape[:2]


        if height < 50 or width < 50:

            return 0.0


        def edge_density(region):

            if region.size == 0:

                return 0.0

            gray = cv2.cvtColor(
                region,
                cv2.COLOR_BGR2GRAY
            )

            edges = cv2.Canny(
                gray,
                100,
                200
            )

            return np.mean(
                edges > 0
            )


        # Full-width strips -- catches centered captions/bars.
        bottom_strip = img[int(height * 0.85):height, :]
        top_strip = img[0:int(height * 0.10), :]

        strip_density = max(
            edge_density(top_strip),
            edge_density(bottom_strip),
        )


        # Individual corners -- catches small watermarks/handles
        # that a full-width average would dilute away.
        corner_h = max(int(height * 0.15), 1)
        corner_w = max(int(width * 0.25), 1)

        corners = [
            img[0:corner_h, 0:corner_w],                      # top-left
            img[0:corner_h, width - corner_w:width],          # top-right
            img[height - corner_h:height, 0:corner_w],        # bottom-left
            img[height - corner_h:height, width - corner_w:width],  # bottom-right
        ]

        corner_density = max(
            edge_density(c) for c in corners
        )


        # Whichever signal is stronger drives the penalty.
        suspicious = max(
            0.0,
            strip_density - 0.18,
            corner_density - 0.22,
        )


        return min(
            suspicious * 1.5,
            0.45
        )


    except Exception:

        return 0.0


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():

    # ========================================================
    # STEP 1
    # ========================================================

    print("=" * 60)

    print(
        "STEP 1: Face detection on input scan"
    )

    print("=" * 60)


    print(
        f"Using input image: {INPUT_IMAGE}"
    )


    input_embedding = get_embedding_from_file(

        str(INPUT_IMAGE)

    )


    if input_embedding is None:

        print(
            "No face detected in input image. Aborting."
        )

        return


    print(
        f"Face detected. Embedding size: "
        f"{len(input_embedding)}"
    )


    # ========================================================
    # STEP 2
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "STEP 2: Web/social media search"
    )

    print("=" * 60)


    results = search_image(

        str(INPUT_IMAGE)

    )


    matches = find_social_matches(
        results
    )


    if not matches:

        print(
            "No social media matches found. Aborting."
        )

        return


    print(
        f"Found {len(matches)} candidate "
        "social media result(s)."
    )


    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # BEST RESULT TRACKING
    # ========================================================

    best_match = None

    best_score = -1.0

    best_candidate_path = None


    # ========================================================
    # PROCESS CANDIDATES
    # ========================================================

    for i, candidate in enumerate(
        matches[:20],
        start=1
    ):

        print("\n" + "-" * 60)

        print(
            f"Candidate {i}: "
            f"{candidate.get('source')}"
        )

        print(
            f"Title: "
            f"{candidate.get('title')}"
        )


        # ----------------------------------------------------
        # Image URLs
        # ----------------------------------------------------

        urls_to_try = [

            url

            for url in [

                candidate.get("image"),

                candidate.get("thumbnail"),

            ]

            if url

        ]


        if not urls_to_try:

            print(
                "No image URL available."
            )

            continue


        processed = False

        last_error = None


        # ----------------------------------------------------
        # Download candidate
        # ----------------------------------------------------

        for image_url in urls_to_try:

            try:

                download_image(

                    image_url,

                    str(CANDIDATE_IMAGE)

                )


                # A download can "succeed" (no exception) while
                # what actually landed on disk isn't a decodable
                # image -- an HTML error page, a truncated file,
                # an unsupported/corrupt format, etc. Verify it
                # actually opens before trusting it.

                check = cv2.imread(str(CANDIDATE_IMAGE))

                if check is None:

                    raise ValueError(
                        "Downloaded file is not a readable image"
                    )


                processed = True

                break


            except Exception as e:

                last_error = e


        if not processed:

            print(
                f"Download failed: "
                f"{last_error}"
            )

            continue


        # ----------------------------------------------------
        # Save debug copy
        # ----------------------------------------------------

        safe_source = "".join(

            c if c.isalnum() else "_"

            for c in str(

                candidate.get(
                    "source",
                    "unknown"
                )

            )

        )


        debug_path = (

            DEBUG_DIR /

            f"{i:02d}_{safe_source}.jpg"

        )


        shutil.copyfile(

            CANDIDATE_IMAGE,

            debug_path

        )


        # ====================================================
        # FACE SCORE
        # ====================================================

        try:

            candidate_embedding = (

                get_embedding_from_file(

                    str(CANDIDATE_IMAGE)

                )

            )


        except Exception as e:

            print(

                f"Skipping candidate -- "
                f"could not process image: {e}"

            )

            continue


        face_score = compare_faces(

            input_embedding,

            candidate_embedding

        )


        if face_score is None:

            face_score = 0.0


        # ====================================================
        # PHASH
        # ====================================================

        try:

            distance = phash_distance(

                INPUT_IMAGE,

                CANDIDATE_IMAGE

            )


            image_score = phash_score(
                distance
            )


        except Exception:

            distance = None

            image_score = 0.0


        # ====================================================
        # SIFT
        # ====================================================

        sift_score = sift_similarity(

            INPUT_IMAGE,

            CANDIDATE_IMAGE

        )


        # ====================================================
        # ASPECT RATIO
        # ====================================================

        ratio_score = aspect_ratio_score(

            INPUT_IMAGE,

            CANDIDATE_IMAGE

        )


        # ====================================================
        # OVERLAY PENALTY
        # ========================================================

        penalty = overlay_penalty(

            CANDIDATE_IMAGE

        )


        # ====================================================
        # FINAL SCORE
        # ====================================================

        #
        # We DO NOT want face similarity to dominate.
        #
        # A fanpage repost can have:
        #
        #     face = 0.95
        #
        # while the actual source image also has:
        #
        #     face = 0.95
        #
        #
        # Therefore:
        #
        # pHash + SIFT + aspect ratio
        # are more important.
        #

        final_score = (

            (image_score * 0.35)

            +

            (sift_score * 0.35)

            +

            (face_score * 0.20)

            +

            (ratio_score * 0.10)

            -

            penalty

        )


        final_score = max(
            0.0,
            min(
                1.0,
                final_score
            )
        )


        # ====================================================
        # PRINT SCORES
        # ====================================================

        print(
            f"Face similarity: "
            f"{face_score:.4f}"
        )


        if distance is not None:

            print(
                f"pHash distance: "
                f"{distance}"
            )

        else:

            print(
                "pHash distance: N/A"
            )


        print(
            f"pHash similarity: "
            f"{image_score:.4f}"
        )


        print(
            f"SIFT similarity: "
            f"{sift_score:.4f}"
        )


        print(
            f"Aspect ratio similarity: "
            f"{ratio_score:.4f}"
        )


        print(
            f"Overlay penalty: "
            f"{penalty:.4f}"
        )


        print(
            f"FINAL SCORE: "
            f"{final_score:.4f}"
        )


        print(
            f"Debug image: "
            f"{debug_path}"
        )


        # ====================================================
        # BEST RESULT
        # ====================================================

        if final_score > best_score:

            best_score = final_score

            best_match = candidate

            best_candidate_path = debug_path

            shutil.copyfile(

                CANDIDATE_IMAGE,

                BEST_MATCH_IMAGE

            )


    # ========================================================
    # NO RESULT
    # ========================================================

    if best_match is None:

        print(
            "\nNo usable candidate found."
        )

        return


    # ========================================================
    # BEST RESULT
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "BEST MATCH"
    )

    print("=" * 60)


    print(
        f"Source: "
        f"{best_match.get('source')}"
    )


    print(
        f"Final score: "
        f"{best_score:.4f}"
    )


    print(
        f"Title: "
        f"{best_match.get('title')}"
    )


    print(
        f"Link: "
        f"{best_match.get('link')}"
    )


    print(
        f"Saved image: "
        f"{BEST_MATCH_IMAGE}"
    )


    print(
        "\nOpen best_match.jpg and visually confirm "
        "that it is the original/unmodified image."
    )


    # ========================================================
    # STEP 3: BLOCKCHAIN
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "STEP 3: Blockchain upload / verification"
    )

    print("=" * 60)


    record = {

        "title":
            best_match.get("title"),

        "source":
            best_match.get("source"),

        "link":
            best_match.get("link"),

        "thumbnail":
            best_match.get("thumbnail"),

        "similarity_score":
            round(best_score, 6),

        "discovered_at":
            time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime()
            ),

    }


    block = add_to_simulated_chain(

        record

    )


    print(
        f"Written to simulated chain "
        f"at block index "
        f"{block['index']}"
    )


    print(
        f"data_hash: "
        f"{block['data_hash']}"
    )


    print(
        f"block_hash: "
        f"{block['block_hash']}"
    )


    print(
        "Chain integrity check: "
        f"{'PASSED' if verify_simulated_chain() else 'FAILED'}"
    )


    # ========================================================
    # OPTIONAL REAL WEB3
    # ========================================================

    web3_result = None


    if web3_enabled():

        print(
            "\nRPC_URL/PRIVATE_KEY detected."
        )

        print(
            "Uploading to live blockchain..."
        )


        web3_result = upload_to_web3(
            record
        )


        print(
            f"tx_hash: "
            f"{web3_result['tx_hash']}"
        )


        print(
            f"block_number: "
            f"{web3_result['block_number']}"
        )


        print(
            f"chain_id: "
            f"{web3_result['chain_id']}"
        )


    else:

        print(
            "\n(RPC_URL/PRIVATE_KEY not set - "
            "using simulated chain only.)"
        )


    # ========================================================
    # SAVE LAST RUN
    # ========================================================

    last_run = {

        "record": record,

        "simulated_block": block,

        "web3_result": web3_result,

    }


    with open(
        LAST_RUN_FILE,
        "w"
    ) as f:

        json.dump(
            last_run,
            f,
            indent=2
        )


    print(
        f"\nSaved run summary to "
        f"{LAST_RUN_FILE}"
    )


    print(
        "\nDone."
    )


    print(
        "Run:"
    )

    print(
        "python -m src.verify"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_pipeline()