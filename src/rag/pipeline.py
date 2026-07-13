from src.retrieval.retriever import Retriever
from src.llm.client import generate_answer
from src.llm.client import stream_answer_tokens
from src.llm.prompts import SYSTEM_PROMPT
from src.evaluation.confidence import compute_confidence
from src.rag.query_router import classify_query
from src.guardrails.validator import validate_query
from src.guardrails.grounding import check_grounding
from src.observability.instrument import new_request_id, timed_stage, log_event


class RagAssistant:

    def __init__(self, index_path):
        self.retriever = Retriever(index_path)

    def build_context(self, chunks):
        context = "\n\n".join(
            f"[Section: {c['metadata'].get('section')}]\n{c['text']}" for c in chunks
        )
        return SYSTEM_PROMPT + "\n\nContext:\n" + context

    def _build_citations(self, retrieved):
        """Turn retrieved chunks into the citation list returned to callers.
        Extracted so both ask() and ask_stream() build citations the same way."""
        citations = []
        for c in retrieved:
            md = c["metadata"]
            citations.append(
                {
                    "document": md.get("source_file"),
                    "section": md.get("section"),
                    "pages": f"{md.get('page_start')} - {md.get('page_end')}",
                }
            )
        return citations

    def _prepare(self, question, request_id):
        """Shared front half: guardrail -> routing -> retrieval -> confidence.

        The routing and retrieval stages are timed so their latency shows up in
        the logs. Returns (retrieved, confidence, early) where `early` is either
        None (proceed) or a ready (answer, citations, confidence) short-circuit.
        """
        is_allowed, rejection_message = validate_query(question)
        if not is_allowed:
            return None, "BLOCKED", (rejection_message, [], "BLOCKED")

        with timed_stage("route", request_id):
            filters = classify_query(question)

        with timed_stage("retrieve", request_id):
            retrieved = self.retriever.query(question, k=10, filters=filters)

        confidence = compute_confidence(retrieved)

        if confidence == "LOW":
            return (
                retrieved,
                confidence,
                ("Insufficient evidence found in documents.", [], confidence),
            )

        return retrieved, confidence, None

    def ask(self, question, request_id=None):
        """Non-streaming path: returns (answer, citations, confidence).

        request_id correlates every log line for this request; callers (the API)
        may pass their own, otherwise we generate one.
        """
        request_id = request_id or new_request_id()
        log_event("request_start", request_id, path="ask", question_chars=len(question))

        retrieved, confidence, early = self._prepare(question, request_id)
        if early is not None:
            log_event(
                "request_end", request_id, outcome="short_circuit", confidence=early[2]
            )
            return early

        with timed_stage("generate", request_id):
            context = self.build_context(retrieved)
            answer = generate_answer(context, question)

        # Output guardrail: is the answer grounded in the retrieved documents?
        retrieved_text = " ".join(c["text"] for c in retrieved)
        is_grounded, overlap = check_grounding(answer, retrieved_text)
        if not is_grounded:
            log_event(
                "request_end",
                request_id,
                outcome="ungrounded",
                confidence="LOW",
                grounding_overlap=round(overlap, 3),
            )
            return (
                "I could not find a well-grounded answer for this in the documents.",
                [],
                "LOW",
            )

        citations = self._build_citations(retrieved)
        log_event(
            "request_end",
            request_id,
            outcome="answered",
            confidence=confidence,
            grounding_overlap=round(overlap, 3),
        )
        return answer, citations, confidence

    def ask_stream(self, question, request_id=None):
        """Streaming path: yields structured events (metadata, token..., done).

        Formatting for the wire (SSE) is the API layer's job.
        """
        request_id = request_id or new_request_id()
        log_event(
            "request_start", request_id, path="ask_stream", question_chars=len(question)
        )

        retrieved, confidence, early = self._prepare(question, request_id)
        if early is not None:
            answer, citations, conf = early
            log_event(
                "request_end", request_id, outcome="short_circuit", confidence=conf
            )
            yield {
                "type": "message",
                "text": answer,
                "confidence": conf,
                "citations": citations,
            }
            return

        citations = self._build_citations(retrieved)
        # Metadata first: confidence + citations are known before generation.
        yield {"type": "metadata", "confidence": confidence, "citations": citations}

        context = self.build_context(retrieved)
        with timed_stage("generate_stream", request_id):
            for delta in stream_answer_tokens(context, question):
                yield {"type": "token", "text": delta}

        log_event(
            "request_end", request_id, outcome="answered_stream", confidence=confidence
        )
        yield {"type": "done"}
