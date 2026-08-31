import requests


def download_image(url, output_path):
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"Downloaded image to: {output_path}")


if __name__ == "__main__":
    url = input("Image URL: ").strip()

    try:
        download_image(url, "data/candidate.jpg")
    except Exception as e:
        print(f"Download failed: {e}")
