from src.face_engine import (
    get_embedding_from_file,
    get_all_embeddings_from_file,
    compare_faces,
    compare_against_all
)

input_embedding = get_embedding_from_file(
    "data/input.jpg"
)

candidate_embeddings = get_all_embeddings_from_file(
    "data/candidate.jpg"
)

if input_embedding is None:
    print("No face found in input image.")
    exit()

if not candidate_embeddings:
    print("No faces found in candidate image.")
    exit()

print("\nFACE MATCH TEST")
print("----------------")
print(f"Faces found in candidate: {len(candidate_embeddings)}")

for i, embedding in enumerate(candidate_embeddings, start=1):
    score = compare_faces(input_embedding, embedding)

    print(
        f"Candidate Face {i}: "
        f"{score:.4f}"
    )

best_score = compare_against_all(
    input_embedding,
    candidate_embeddings
)

print("----------------")
print(f"BEST similarity score: {best_score:.4f}")