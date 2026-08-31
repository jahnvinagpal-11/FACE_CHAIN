import cv2
import insightface
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


_model = None


def load_model():
    global _model

    if _model is None:
        _model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        _model.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

    return _model


def get_embedding(image):
    model = load_model()

    faces = model.get(image)

    if not faces:
        return None

    # Use the largest detected face
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) *
                      (f.bbox[3] - f.bbox[1])
    )

    return face.embedding


def get_embedding_from_file(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return get_embedding(image)


def compare_faces(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return None

    score = cosine_similarity(
        [embedding1],
        [embedding2]
    )[0][0]

    return float(score)


if __name__ == "__main__":
    embedding = get_embedding_from_file("data/input.jpg")

    if embedding is None:
        print("No face detected.")
    else:
        print("Face detected!")
        print("Embedding dimensions:", len(embedding))
