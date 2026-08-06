from app.db import get_contacts
from app.embeddings import embed_text, cosine_similarity
from app.config import USE_ATLAS_VECTOR_SEARCH, ATLAS_VECTOR_INDEX_NAME

# Only these fields can be used in structured filters. Extend as needed,
# but never let the LLM pass arbitrary/raw Mongo operators outside this list.
ALLOWED_FIELDS = {"name", "company", "role", "email", "phone", "tags", "city", "country"}

# Mongo operators we allow inside a filter value. Blocks things like $where.
ALLOWED_OPERATORS = {"$regex", "$options", "$in", "$eq", "$ne", "$gte", "$lte", "$exists"}


def _sanitize_filter(filter_dict: dict) -> dict:
    """Whitelist-based sanitizer so the LLM can't inject arbitrary Mongo ops."""
    if not isinstance(filter_dict, dict):
        return {}
    clean = {}
    for field, value in filter_dict.items():
        if field not in ALLOWED_FIELDS:
            continue
        if isinstance(value, dict):
            clean_val = {op: v for op, v in value.items() if op in ALLOWED_OPERATORS}
            if clean_val:
                clean[field] = clean_val
        else:
            clean[field] = value
    return clean


def query_contacts(filter: dict = None, limit: int = 20):
    """Exact/structured lookup, e.g. {'company': 'Acme'} or {'name': {'$regex': 'john', '$options': 'i'}}."""
    filter = _sanitize_filter(filter or {})
    limit = max(1, min(int(limit or 20), 100))
    cursor = get_contacts().find(filter, {"_id": 0}).limit(limit)
    return list(cursor)


def count_contacts(filter: dict = None):
    filter = _sanitize_filter(filter or {})
    return {"count": get_contacts().count_documents(filter)}


def semantic_search_contacts(query: str, top_k: int = 5):
    """Fuzzy/semantic lookup over contact embeddings."""
    top_k = max(1, min(int(top_k or 5), 20))
    query_vec = embed_text(query)

    if USE_ATLAS_VECTOR_SEARCH:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": ATLAS_VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": max(100, top_k * 10),
                    "limit": top_k,
                }
            },
            {"$project": {"_id": 0, "embedding": 0}},
        ]
        return list(get_contacts().aggregate(pipeline))

    # Fallback: in-memory cosine similarity (fine for small/medium collections)
    docs = list(get_contacts().find({"embedding": {"$exists": True}}, {"_id": 0}))
    scored = []
    for doc in docs:
        vec = doc.get("embedding")
        if not vec:
            continue
        score = cosine_similarity(query_vec, vec)
        doc_no_vec = {k: v for k, v in doc.items() if k != "embedding"}
        scored.append((score, doc_no_vec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


# ---- OpenAI tool schema (function calling) ----
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_contacts",
            "description": (
                "Look up contacts with an EXACT/structured MongoDB filter. "
                f"Only these fields are allowed: {sorted(ALLOWED_FIELDS)}. "
                "Use $regex with $options:'i' for case-insensitive partial text match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {"type": "object", "description": "MongoDB filter object"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["filter"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_contacts",
            "description": "Count contacts matching a structured filter (same field rules as query_contacts).",
            "parameters": {
                "type": "object",
                "properties": {"filter": {"type": "object"}},
                "required": ["filter"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search_contacts",
            "description": (
                "Fuzzy/semantic search over contacts when the question isn't an exact field match, "
                "e.g. 'find someone who might know about marketing in Europe'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_IMPLS = {
    "query_contacts": lambda args: query_contacts(**args),
    "count_contacts": lambda args: count_contacts(**args),
    "semantic_search_contacts": lambda args: semantic_search_contacts(**args),
}
