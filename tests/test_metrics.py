"""Unit tests for retrieval metrics (pure functions, CI-safe)."""
from src.evaluation.metrics import precision_at_k, recall_at_k


def test_precision_all_relevant():
    retrieved = ["FAQ", "CODE PER PGA"]
    expected = ["faq", "code per pga"]
    assert precision_at_k(retrieved, expected) == 1.0


def test_precision_half_relevant():
    retrieved = ["FAQ", "Unrelated Section"]
    expected = ["faq"]
    assert precision_at_k(retrieved, expected) == 0.5


def test_precision_empty_retrieved_is_zero():
    assert precision_at_k([], ["faq"]) == 0.0


def test_recall_finds_all_expected():
    retrieved = ["FAQ page 15", "CODE PER PGA table"]
    expected = ["FAQ", "CODE PER PGA"]
    assert recall_at_k(retrieved, expected) == 1.0


def test_recall_misses_one():
    retrieved = ["FAQ page 15"]
    expected = ["FAQ", "Disclaim Reporting"]
    assert recall_at_k(retrieved, expected) == 0.5


def test_recall_empty_expected_is_zero():
    assert recall_at_k(["FAQ"], []) == 0.0


def test_substring_matching_is_case_insensitive():
    # "code per pga" expected label matches "CODE PER PGA table" retrieved.
    assert precision_at_k(["CODE PER PGA table"], ["code per pga"]) == 1.0
