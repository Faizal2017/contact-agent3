import threading
from urllib.parse import quote

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import run_agent
from app.config import APP_HOST, APP_PORT, RAG, CONTACTS_API_URL
from app.http_client import request as http_request, reset_agent, MudraIDError

app = FastAPI(title="Contact Agent")


class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None


class ChatResponse(BaseModel):
    answer: str
    history: list[dict]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = run_agent(req.message, req.history)
    return ChatResponse(**result)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/refresh-agent")
def refresh_agent():
    """Force the MudraID agent to re-initialize and pick up new platform grants.

    Call this after granting 'faizalplatform.mudraidtesting.online' (or any
    other platform) to the agent in the MudraID portal.
    """
    reset_agent()
    return {"status": "ok", "message": "MudraID agent reset. It will re-connect on the next request."}


def _host_request(method: str, path: str, body: dict | None = None):
    """Forward a request to the contacts host API (CONTACTS_API_URL)."""
    url = CONTACTS_API_URL.rstrip("/") + path
    try:
        resp = http_request(method, url, json=body, timeout=30)
    except (requests.RequestException, MudraIDError) as e:
        raise HTTPException(status_code=502, detail=f"Contacts API unreachable: {e}")
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    if not resp.content:
        return None
    try:
        return resp.json()
    except ValueError:
        return resp.text


@app.get("/api/contacts")
def list_contacts():
    return _host_request("GET", "")


@app.put("/api/contacts/{contact_id}")
def update_contact(contact_id: str, body: dict):
    return _host_request("PUT", f"/{quote(contact_id, safe='')}", body)


@app.delete("/api/contacts/{contact_id}")
def delete_contact(contact_id: str):
    return _host_request("DELETE", f"/{quote(contact_id, safe='')}")


@app.on_event("startup")
def startup():
    if not RAG:
        return
    # Runs the MongoDB change-stream listener in the background so the
    # vector embeddings stay in sync with inserts/updates/deletes.
    # Requires MongoDB to be a replica set (Atlas is by default).
    from app.sync import start_change_stream_listener
    t = threading.Thread(target=start_change_stream_listener, daemon=True)
    t.start()


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=APP_HOST, port=APP_PORT, reload=True)
