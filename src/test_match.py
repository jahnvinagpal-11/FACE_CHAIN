from src.face_engine import (
    get_embedding_from_file,
    compare_faces
)


input_embedding = get_embedding_from_file(
    "data/input.jpg"
)

candidate_embedding = get_embedding_from_file(
    "data/candidate.jpg"
)


if input_embedding is None:
    print("No face found in input image.")
    exit()

if candidate_embedding is None:
    print("No face found in candidate image.")
    exit()


score = compare_faces(
    input_embedding,
    candidate_embedding
)


print("\nFACE MATCH TEST")
print("----------------")
print(f"Similarity score: {score:.4f}")
