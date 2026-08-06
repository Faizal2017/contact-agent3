import json
from openai import OpenAI
from app.config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    RAG,
    CONTACTS_API_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_REASONING,
)
from app.http_client import request as http_request

# Chat goes through OpenRouter when its key is set, otherwise OpenAI.
USE_OPENROUTER = bool(OPENROUTER_API_KEY)

client = OpenAI(base_url=OPENROUTER_BASE_URL if USE_OPENROUTER else None,
                api_key=OPENROUTER_API_KEY or OPENAI_API_KEY)
CHAT_MODEL = OPENROUTER_MODEL if USE_OPENROUTER else OPENAI_CHAT_MODEL


def chat_kwargs():
    """extra_body for the OpenAI client. OpenRouter's reasoning feature is
    enabled via a provider-specific body field, which the SDK doesn't know."""
    if USE_OPENROUTER and OPENROUTER_REASONING:
        return {"extra_body": {"reasoning": {"enabled": True}}}
    return {}

SYSTEM_PROMPT = """You are an assistant that answers questions about contacts stored in a MongoDB database.
Always use the available tools to fetch real data before answering — never guess or make up contact details.
Use query_contacts / count_contacts for exact lookups (company name, email, role, etc).
Use semantic_search_contacts for fuzzy or descriptive questions.
You may call multiple tools, and call them more than once, if the question needs it (e.g. comparisons).
Once you have enough data, answer the user clearly and concisely, citing only what the tools returned.
If no contacts match, say so plainly instead of inventing data.
"""

NON_RAG_SYSTEM_PROMPT = """You are an assistant that answers questions about contacts.
The full contacts dataset is provided below in JSON. Base your answers ONLY on that data —
never guess or make up contact details.
If the data doesn't contain the answer, say so plainly instead of inventing it.
Answer clearly and concisely.
"""

MAX_TOOL_ROUNDS = 5


def fetch_contacts() -> list[dict]:
    """Fetch all contacts from CONTACTS_API_URL."""
    resp = http_request("GET", CONTACTS_API_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("contacts", "data", "results", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def run_agent_no_rag(user_message: str, history: list[dict] | None = None) -> dict:
    """Non-RAG mode: fetch all contacts and send them to the LLM directly."""
    try:
        contacts_data = fetch_contacts()
    except Exception as e:
        msg = f"I couldn't fetch the contacts from {CONTACTS_API_URL} ({e}). Make sure that backend is running."
        new_history = (history or []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": msg},
        ]
        return {"answer": msg, "history": new_history}

    payload = json.dumps(contacts_data, default=str, ensure_ascii=False)
    messages = [{"role": "system", "content": NON_RAG_SYSTEM_PROMPT + "\n\nContacts data:\n" + payload}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, **chat_kwargs())
    final_text = resp.choices[0].message.content or ""
    new_history = (history or []) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": final_text},
    ]
    return {"answer": final_text, "history": new_history}


def run_agent(user_message: str, history: list[dict] | None = None) -> dict:
    """
    history: list of {"role": "user"|"assistant", "content": str} from prior turns (optional).
    Returns {"answer": str, "history": updated_history}
    """
    if not RAG:
        return run_agent_no_rag(user_message, history)

    from app.tools import TOOL_SCHEMAS, TOOL_IMPLS

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            **chat_kwargs(),
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            final_text = msg.content or ""
            new_history = (history or []) + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": final_text},
            ]
            return {"answer": final_text, "history": new_history}

        messages.append(msg)
        for call in msg.tool_calls:
            fn_name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            impl = TOOL_IMPLS.get(fn_name)
            try:
                result = impl(args) if impl else {"error": f"unknown tool {fn_name}"}
            except Exception as e:
                result = {"error": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, default=str),
            })

    # Safety net if it loops too long without a final answer
    return {"answer": "I wasn't able to fully resolve that — could you rephrase or narrow the question?",
            "history": history or []}
