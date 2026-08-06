import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB", "yourdb")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "contacts")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# OpenRouter (chat completions). When OPENROUTER_API_KEY is set it replaces
# OpenAI as the chat provider. Embeddings still use OPENAI_API_KEY because
# OpenRouter does not offer an embeddings endpoint.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "inclusionai/ling-3.0-flash:free")
OPENROUTER_REASONING = os.getenv("OPENROUTER_REASONING", "false").lower() == "true"

USE_ATLAS_VECTOR_SEARCH = os.getenv("USE_ATLAS_VECTOR_SEARCH", "false").lower() == "true"
ATLAS_VECTOR_INDEX_NAME = "vector_index"

RAG = os.getenv("RAG", "false").lower() == "true"
CONTACTS_API_URL = os.getenv("CONTACTS_API_URL", "http://localhost:8000/api/contacts")

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8001"))

if not MONGO_URI:
    raise RuntimeError("MONGO_URI is not set. Copy .env.example to .env and fill it in.")
if not OPENAI_API_KEY and not OPENROUTER_API_KEY:
    raise RuntimeError("Neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set. Copy .env.example to .env and fill one in.")
