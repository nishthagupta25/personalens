from sentence_transformers import SentenceTransformer
import numpy as np

print("[embeddings] Loading SentenceTransformer...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("[embeddings] Embeddings model ready!")


def get_embedding(text: str) -> list:
    """
    Converts text into a 384-dimensional vector.
    Similar texts will have similar vectors.
    """
    embedding = model.encode(text)
    return embedding.tolist()


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Measures how similar two texts are.
    1.0 = identical meaning, 0.0 = completely different
    """
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_embedding_summary(text: str) -> dict:
    """
    Returns embedding + basic stats about it.
    We don't send the full 384 numbers to frontend
    — too large. Just store it for later use.
    """
    embedding = get_embedding(text)

    return {
        "dimensions": len(embedding),
        "preview": [round(x, 4) for x in embedding[:5]],
        "norm": round(float(np.linalg.norm(embedding)), 4)
    }