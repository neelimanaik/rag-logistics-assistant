import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.rag.pipeline import RagAssistant

app = FastAPI()
assistant = RagAssistant("data/processed/index")


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask_question(query: Query):
    """Non-streaming endpoint: returns the whole answer in one JSON response."""
    answer, citations, confidence = assistant.ask(query.question)

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence
    }


@app.post("/ask/stream")
def ask_question_stream(query: Query):
    """Streaming endpoint using Server-Sent Events (SSE).

    The pipeline yields structured event dicts; here we format each one onto the
    wire as `data: <json>\\n\\n`, which is the SSE format browsers and HTTP
    clients understand. Because we stream, the client sees tokens as the model
    produces them instead of waiting for the full answer.
    """
    def event_stream():
        for event in assistant.ask_stream(query.question):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
