from fastapi import FastAPI
from pydantic import BaseModel
from src.rag.pipeline import RagAssistant

app = FastAPI()
assistant = RagAssistant("data/processed/index")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    answer, citations, confidence = assistant.ask(query.question)

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence
    }