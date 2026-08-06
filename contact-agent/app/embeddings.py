import numpy as np
from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def contact_to_text(doc: dict) -> str:
    """
    Flatten a contact document into a text blob suitable for embedding.
    Edit this if your schema has different/extra fields.
    """
    parts = []
    for key in ["name", "company", "role", "email", "phone", "address", "notes", "tags"]:
        val = doc.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        parts.append(f"{key}: {val}")
    return " | ".join(parts)


def embed_text(text: str) -> list[float]:
    resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def embed_contact(doc: dict) -> list[float]:
    return embed_text(contact_to_text(doc))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
