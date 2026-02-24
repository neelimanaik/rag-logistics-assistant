from multiprocessing import context
from src.retrieval.retriever import Retriever
from src.llm.client import generate_answer
from src.llm.prompts import SYSTEM_PROMPT
from src.evaluation.confidence import compute_confidence
from src.rag.query_router import classify_query
from src.llm.client import stream_answer


class RagAssistant:

    def __init__(self, index_path):
        self.retriever = Retriever(index_path)

    def build_context(self, chunks):

        context = "\n\n".join(
            f"[Section: {c['metadata'].get('section')}]\n{c['text']}"
            for c in chunks
        )

        return SYSTEM_PROMPT + "\n\nContext:\n" + context

    def ask(self, question):

        #retrieved = self.retriever.query(question, k=5)
        filters = None

        # --- Simple intent routing ---
        #if any(word in question.lower() for word in ["how", "steps", "procedure"]):
         #   filters = {"document_type": "user_manual"}

        #if any(word in question.lower() for word in ["regulation", "law", "duty"]):
         #   filters = {"document_type": "customs_regulation"}

        filters = classify_query(question)

        retrieved = self.retriever.query(question, k=10, filters=filters)
        confidence = compute_confidence(retrieved)
        if confidence == "LOW":
            return "Insufficient evidence found in documents.", citations, confidence

        context = self.build_context(retrieved)

        #answer = generate_answer(context, question)
        print("\n===== STREAMING ANSWER =====\n")
        answer = stream_answer(context, question)


        # ----- Build citation list -----
        citations = []
        for c in retrieved:
            md = c["metadata"]

            citations.append({
                "document": md.get("source_file"),
                "section": md.get("section"),
                "pages": f"{md.get('page_start')} - {md.get('page_end')}"
            })

        return answer, citations, confidence


