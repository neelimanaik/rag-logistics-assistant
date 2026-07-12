"""Integration tests for RagAssistant.ask() control flow.

We bypass __init__ (so no FAISS index is needed) and monkeypatch the LLM-facing
helpers, so these run with no model and no network. They lock in the bug-1
(citation shape), bug-5 (guardrail), and B3 (grounding) behaviour.
"""
from src.rag import pipeline
from src.rag.pipeline import RagAssistant


class _FakeRetriever:
    def __init__(self, results):
        self._results = results

    def query(self, question, k=10, filters=None):
        return self._results


def _assistant(results):
    a = RagAssistant.__new__(RagAssistant)  # bypass __init__ (no FAISS load)
    a.retriever = _FakeRetriever(results)
    return a


def _allow_and_route(monkeypatch, confidence="HIGH"):
    monkeypatch.setattr(pipeline, "validate_query", lambda q: (True, None))
    monkeypatch.setattr(pipeline, "classify_query", lambda q: None)
    monkeypatch.setattr(pipeline, "compute_confidence", lambda results: confidence)


def test_blocked_query_short_circuits(monkeypatch):
    monkeypatch.setattr(pipeline, "validate_query", lambda q: (False, "out of scope"))
    a = _assistant([])
    answer, citations, confidence = a.ask("something blocked")
    assert answer == "out of scope"
    assert citations == []
    assert confidence == "BLOCKED"


def test_low_confidence_returns_insufficient(monkeypatch):
    _allow_and_route(monkeypatch, confidence="LOW")
    a = _assistant([{"metadata": {}, "text": "x", "score": 0.9}])
    answer, citations, confidence = a.ask("weak query")
    assert "Insufficient evidence" in answer
    assert citations == []
    assert confidence == "LOW"


def test_happy_path_builds_citations(monkeypatch):
    _allow_and_route(monkeypatch, confidence="HIGH")
    # The answer reuses the source vocabulary, so it passes the grounding check.
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        lambda context, question: "Duty exemption applies to certain imported goods.",
    )
    results = [{
        "metadata": {
            "source_file": "doc.pdf",
            "section": "FAQ",
            "page_start": 15,
            "page_end": 15,
        },
        "text": "Duty exemption certificate applies to certain imported goods.",
        "score": 0.4,
    }]
    a = _assistant(results)
    answer, citations, confidence = a.ask("good query")
    assert answer == "Duty exemption applies to certain imported goods."
    assert confidence == "HIGH"
    assert citations == [{"document": "doc.pdf", "section": "FAQ", "pages": "15 - 15"}]


def test_ungrounded_answer_is_refused(monkeypatch):
    _allow_and_route(monkeypatch, confidence="HIGH")
    # Answer shares no vocabulary with the retrieved context -> not grounded.
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        lambda context, question: "Bananas and zebras enjoy spaceship holidays.",
    )
    results = [{
        "metadata": {"source_file": "doc.pdf", "section": "FAQ", "page_start": 1, "page_end": 1},
        "text": "Duty exemption certificate applies to imported goods.",
        "score": 0.4,
    }]
    a = _assistant(results)
    answer, citations, confidence = a.ask("good query")
    assert "could not find a well-grounded answer" in answer
    assert citations == []
    assert confidence == "LOW"
