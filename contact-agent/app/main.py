import threading
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import run_agent
from app.config import APP_HOST, APP_PORT, RAG

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
