import json

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from src.api.auth import authenticate, create_access_token
from src.api.rate_limit import rate_limit
from src.config.settings import settings
from src.observability.instrument import log_event, new_request_id
from src.rag.pipeline import RagAssistant

app = FastAPI(title="RAG Logistics Assistant", version="0.1.0")


# --- Assistant as a lazily-created, injectable dependency ---
# We don't build the RagAssistant at import time (that would load the FAISS index
# on startup and make the module impossible to import without one). Instead it's
# created on first use and provided via Depends(), so tests can override it with a
# fake and exercise the API with no model or index.
_assistant = None


def get_assistant():
    global _assistant
    if _assistant is None:
        _assistant = RagAssistant(settings.index_path)
    return _assistant


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    """Liveness/readiness probe. Cheap on purpose: it does NOT call the model or
    load the index, so orchestrators can poll it frequently."""
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "chat_model": settings.chat_model,
    }


@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Demo login: exchange username/password for a JWT bearer token.

    In production this would be an OIDC flow against an identity provider; here
    it validates demo credentials and issues a short-lived signed JWT.
    """
    if not authenticate(form_data.username, form_data.password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(form_data.username)
    return {"access_token": token, "token_type": "bearer"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return a consistent JSON error shape and DON'T leak stack traces to
    clients. The real error is logged (with a request_id) for operators."""
    request_id = new_request_id()
    log_event(
        "unhandled_error",
        request_id,
        path=str(request.url.path),
        error=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "request_id": request_id},
    )


@app.post("/ask")
def ask_question(
    query: Query,
    assistant: RagAssistant = Depends(get_assistant),
    user: str = Depends(rate_limit),
):
    """Non-streaming endpoint: returns the whole answer in one JSON response.
    Requires a valid Bearer token and is rate-limited per user."""
    answer, citations, confidence = assistant.ask(query.question)
    return {"answer": answer, "citations": citations, "confidence": confidence}


@app.post("/ask/stream")
def ask_question_stream(
    query: Query,
    assistant: RagAssistant = Depends(get_assistant),
    user: str = Depends(rate_limit),
):
    """Streaming endpoint using Server-Sent Events (SSE).

    The pipeline yields structured event dicts; here we format each onto the wire
    as `data: <json>\\n\\n`, so the client sees tokens as they're produced.
    """

    def event_stream():
        for event in assistant.ask_stream(query.question):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
