import os
import serpapi
from dotenv import load_dotenv


load_dotenv()


SOCIAL_SOURCES = {
    "Instagram",
    "Facebook",
    "X",
    "Twitter",
    "TikTok",
    "Pinterest",
}


def search_image(image_path):
    api_key = os.getenv("SERPAPI_KEY")

    if not api_key:
        raise ValueError("SERPAPI_KEY not found in .env")

    client = serpapi.Client(api_key=api_key)

    print("[1] Uploading image to Google Lens...")

    upload = client.upload_image(image_path)

    if "error" in upload:
        raise RuntimeError(upload["error"])

    image_id = upload["image_id"]

    print("[2] Image uploaded.")

    print("[3] Searching Google Lens...")

    results = client.search({
        "engine": "google_lens",
        "image_id": image_id,
    })

    if "error" in results:
        raise RuntimeError(results["error"])

    print("[4] Search completed.")

    return results


def find_social_matches(results):
    matches = results.get("visual_matches", [])

    social_matches = []

    for result in matches:
        source = result.get("source", "")

        if source in SOCIAL_SOURCES:
            social_matches.append(result)

    return social_matches


if __name__ == "__main__":
    results = search_image("data/input.jpg")

    matches = find_social_matches(results)

    print(f"\nSocial media matches found: {len(matches)}")

    for i, result in enumerate(matches[:10], start=1):
        print(f"\n--- Social Result {i} ---")
        print("Title:", result.get("title"))
        print("Source:", result.get("source"))
        print("Link:", result.get("link"))
        print("Image:", result.get("thumbnail"))
