"""Unit tests for the fast rule-based query router (no LLM call)."""
from src.rag.query_router import classify_query


def test_functional_query_routes_to_user_manual():
    assert classify_query("How do I click the button to file an entry?") == {
        "document_type": "user_manual"
    }


def test_regulatory_query_routes_to_customs():
    assert classify_query("What are the HTS duty exemption regulations?") == {
        "document_type": "customs_regulation"
    }


def test_inconclusive_query_returns_no_filter():
    # No hints either way -> no filter (retriever will search everything).
    assert classify_query("hello there") is None
