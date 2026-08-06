# Contact Agent

An agentic chatbot over your MongoDB `contacts` collection. It uses OpenAI GPT
with function calling to decide whether to run a structured MongoDB query
(exact lookups) or a semantic vector search (fuzzy lookups), then answers
based on the real data returned.

Includes a MongoDB Change Stream listener that automatically keeps each
contact's vector embedding in sync whenever a document is inserted, updated,
or deleted — no manual/full rebuilds needed.

## 1. Setup

```bash
cd contact-agent
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# now edit .env and fill in MONGO_URI and OPENAI_API_KEY
```

## 2. Your MongoDB data

Your `contacts` collection can look like:

```json
{
  "name": "John Doe",
  "company": "Acme Corp",
  "role": "Sales Manager",
  "email": "john@acme.com",
  "phone": "+1 555 1234",
  "city": "New York",
  "country": "USA",
  "tags": ["vip", "marketing"]
}
```

Field names matter: if your schema differs, update `ALLOWED_FIELDS` in
`app/tools.py` and the field list in `contact_to_text()` in `app/embeddings.py`.

## 3. Backfill embeddings (run once for existing data)

```bash
python -m app.sync backfill
```

This embeds every existing contact that doesn't yet have an `embedding` field.
New/changed/deleted contacts after this point are handled automatically by
the change-stream listener that starts with the app (step 4) — you don't
need to re-run this unless you're bulk-importing data while the app is off.

> **Note:** Change Streams require MongoDB to run as a **replica set**.
> MongoDB Atlas clusters are replica sets by default, so this works
> out of the box. A local standalone `mongod` does NOT support change
> streams — either run it as a single-node replica set or just re-run
> the backfill command manually after bulk edits.

## 4. (Optional but recommended) Atlas Vector Search index

If your data is on Atlas, create a Vector Search index named `vector_index`
on the `contacts` collection:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ]
}
```

Then set `USE_ATLAS_VECTOR_SEARCH=true` in `.env`. Without this, the app
still works — it falls back to in-memory cosine similarity search, which is
fine for small/medium collections (a few thousand contacts) but won't scale
as well as native Atlas Vector Search.

## 5. Run the app

```bash
uvicorn app.main:app --reload --port 7000
```

On Windows PowerShell you can also run:

```powershell
.\run-backend.ps1
```

Open **http://localhost:7000** for the built-in chat UI (served from
`static/index.html`), or call the API directly:

```bash
curl -X POST http://localhost:7000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "who works at Acme Corp?"}'
```

## How it decides what to do

- **Exact questions** ("contacts at Acme Corp", "John's email") → the LLM
  calls `query_contacts` / `count_contacts`, which run a sanitized, whitelisted
  MongoDB filter.
- **Fuzzy/descriptive questions** ("someone who might know about marketing
  in Europe") → the LLM calls `semantic_search_contacts`, which embeds the
  question and finds the closest contacts by vector similarity.
- The LLM can chain multiple tool calls for questions that need more than
  one lookup (e.g. comparing two companies).

## Security notes

- `app/tools.py` whitelists which fields (`ALLOWED_FIELDS`) and Mongo
  operators (`ALLOWED_OPERATORS`) the LLM is allowed to use in a filter —
  it can never run arbitrary/raw Mongo queries (`$where`, etc. are blocked).
- Never commit your real `.env` file — only `.env.example` should be in
  version control.
- Add authentication in front of `/chat` before exposing this publicly.

## Extending it

- Add more tools in `app/tools.py` (e.g. `get_contact_by_email`,
  `list_companies`) — narrower tools tend to be more reliable than one
  generic query tool.
- Swap `gpt-4o` for `gpt-4o-mini` in `.env` for lower cost if accuracy stays
  acceptable for your data size.
- Add a conversation persistence layer (e.g. store `history` per session ID
  in Mongo/Redis) if you want multi-user, multi-session chat.
